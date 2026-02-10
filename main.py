import os
import io
import re
import json
import base64
import random
import logging
import difflib
import urllib.parse
import unicodedata
from datetime import datetime
from typing import List, Optional, Any, Dict, Union

# Frameworks e Utilitários
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, BeforeValidator, ConfigDict
from typing_extensions import Annotated

# Banco de Dados
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson.objectid import ObjectId
from pymongo.errors import PyMongoError

# IA e Processamento de Imagem
import google.generativeai as genai
from PIL import Image, ImageOps
import pillow_heif

# Geração de PDF
from fpdf import FPDF

# --- CONFIGURAÇÃO DE LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("TechnoBoltAPI")

# --- INICIALIZAÇÃO DE SUPORTE HEIC ---
pillow_heif.register_heif_opener()

# --- CONFIGURAÇÕES DE AMBIENTE (SETTINGS) ---
class Settings:
    """Centraliza as configurações da aplicação."""
    MONGO_USER = os.environ.get("MONGO_USER", "technobolt")
    MONGO_PASS = os.environ.get("MONGO_PASS", "tech@132")
    MONGO_HOST = os.environ.get("MONGO_HOST", "cluster0.zbjsvk6.mongodb.net")
    DB_NAME = "technoboltgym"
    API_TITLE = "TechnoBolt Gym Hub API"
    API_VERSION = "92.0-Enterprise-Opt"
    
    # Rotação de chaves de API para balanceamento de carga
    GEMINI_KEYS = [
        os.environ.get(f"GEMINI_CHAVE_{i}") 
        for i in range(1, 8) 
        if os.environ.get(f"GEMINI_CHAVE_{i}")
    ]

settings = Settings()

# --- MOTORES DE IA (MANTIDOS ESTRITAMENTE IGUAIS) ---
MOTORES_TECHNOBOLT = [
    "models/gemini-3-flash-preview", 
    "models/gemini-2.5-flash", 
    "models/gemini-2.0-flash", 
    "models/gemini-flash-latest"
]

# --- CAMADA DE DADOS: CONEXÃO MONGODB ---
class Database:
    client: MongoClient = None

    @classmethod
    def connect(cls):
        try:
            password = urllib.parse.quote_plus(settings.MONGO_PASS)
            uri = f"mongodb+srv://{settings.MONGO_USER}:{password}@{settings.MONGO_HOST}/?appName=Cluster0"
            cls.client = MongoClient(uri)
            # Teste de conexão (Ping)
            cls.client.admin.command('ping')
            logger.info("✅ Conexão com MongoDB estabelecida com sucesso.")
        except Exception as e:
            logger.critical(f"❌ Falha crítica ao conectar no MongoDB: {e}")
            raise e

    @classmethod
    def get_db(cls):
        if cls.client is None:
            cls.connect()
        return cls.client[settings.DB_NAME]

# Inicializa conexão
Database.connect()
db = Database.get_db()

# --- UTILITÁRIOS: PYOBJECTID (SERIALIZAÇÃO AUTOMÁTICA) ---
# Isso substitui a função 'preparar_resposta_frontend' de forma nativa no Pydantic
PyObjectId = Annotated[str, BeforeValidator(str)]

class MongoModel(BaseModel):
    """Classe base para modelos que usam ObjectId."""
    id: Optional[PyObjectId] = Field(alias="_id", default=None)

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

# --- MODELOS DE DADOS (SCHEMAS) ---
# Aumenta a robustez e documentação da API

class UserLogin(BaseModel):
    usuario: str
    senha: str

class UserRegister(BaseModel):
    usuario: str
    senha: str
    nome: str
    peso: float
    altura: float
    genero: str

class UserUpdate(BaseModel):
    usuario: str
    nome: Optional[str] = None
    peso: Optional[float] = None
    altura: Optional[float] = None
    genero: Optional[str] = None
    restricoes_alim: Optional[str] = None
    restricoes_fis: Optional[str] = None
    medicamentos: Optional[str] = None
    info_add: Optional[str] = None
    foto_perfil: Optional[str] = None

class SocialPostRequest(BaseModel):
    usuario: str
    post_id: str

class SocialCommentRequest(BaseModel):
    usuario: str
    post_id: str
    texto: str

class ChatMessageRequest(BaseModel):
    remetente: str
    destinatario: str
    texto: str

class AdminUserEdit(BaseModel):
    target_user: str
    status: Optional[str] = None
    creditos: Optional[int] = None

# --- BANCO DE EXERCÍCIOS ---
EXERCISE_DB = {}
EXERCISE_LIST_STRING = ""

def carregar_exercicios():
    global EXERCISE_DB, EXERCISE_LIST_STRING
    try:
        with open("exercises.json", "r", encoding="utf-8") as f:
            EXERCISE_DB = json.load(f)
            keys = list(EXERCISE_DB.keys())
            EXERCISE_LIST_STRING = ", ".join(keys)
        logger.info(f"✅ Banco de Exercícios Carregado: {len(EXERCISE_DB)} itens.")
    except FileNotFoundError:
        logger.warning("⚠️ Arquivo exercises.json não encontrado. Operando sem validação visual.")
    except Exception as e:
        logger.error(f"⚠️ Erro ao carregar exercises.json: {e}")

carregar_exercicios()

# --- SERVIÇOS (LÓGICA DE NEGÓCIO) ---

class AIService:
    """Gerencia interações com a API do Gemini."""
    
    @staticmethod
    def _get_api_key():
        if not settings.GEMINI_KEYS:
            logger.error("Nenhuma chave de API do Gemini configurada.")
            return None
        key = random.choice(settings.GEMINI_KEYS)
        return key

    @staticmethod
    def generate_content(prompt: str, image_bytes: Optional[bytes] = None) -> Optional[str]:
        """
        Executa uma requisição para a IA com retry e fallback de modelos.
        """
        api_key = AIService._get_api_key()
        if not api_key:
            return None

        img_blob = {"mime_type": "image/jpeg", "data": image_bytes} if image_bytes else None
        
        # Configura cliente
        genai.configure(api_key=api_key)
        
        for modelo in MOTORES_TECHNOBOLT:
            try:
                logger.info(f"🧠 Tentando inferência com modelo: {modelo}")
                model = genai.GenerativeModel(modelo)
                
                config = genai.types.GenerationConfig(
                    response_mime_type="application/json" if "json" in prompt.lower() else "text/plain",
                    max_output_tokens=8192,
                    temperature=0.7
                )
                
                inputs = [prompt, img_blob] if img_blob else [prompt]
                response = model.generate_content(inputs, generation_config=config)
                
                if response and response.text:
                    return response.text
                    
            except Exception as e:
                logger.warning(f"Falha no modelo {modelo}: {e}. Tentando próximo...")
                continue
                
        logger.error("❌ Todos os modelos de IA falharam.")
        return None

class ImageService:
    """Utilitários para processamento de imagem."""
    
    @staticmethod
    def optimize(file_bytes: bytes, quality: int = 70, size: tuple = (800, 800)) -> bytes:
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                # Corrige orientação baseada em EXIF
                img = ImageOps.exif_transpose(img)
                # Converte para RGB (remove alpha se houver)
                if img.mode != 'RGB':
                    img = img.convert("RGB")
                
                # Redimensiona mantendo proporção
                img.thumbnail(size)
                
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality, optimize=True)
                return output.getvalue()
        except Exception as e:
            logger.error(f"Erro ao otimizar imagem: {e}")
            return file_bytes # Retorna original em caso de erro

class PDFService(FPDF):
    """Gerador de relatórios PDF customizado."""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        # Paleta de Cores TechnoBolt
        self.col_fundo = (20, 20, 25)
        self.col_card = (35, 35, 40)
        self.col_azul = (59, 130, 246)
        self.col_texto = (230, 230, 230)
        self.col_destaque = (0, 255, 200)
        self.col_verde = (16, 185, 129)

    def sanitizar_texto(self, texto: Any) -> str:
        if not texto: return ""
        texto = str(texto)
        subs = {
            "🚀": ">>", "✅": "[OK]", "⚠️": "[!]", 
            "💊": "", "🥗": "", "🏋️": "", "📊": "",
            "**": "", "###": "", "##": "", "–": "-", 
            "“": '"', "”": '"'
        }
        for k, v in subs.items():
            texto = texto.replace(k, v)
        
        # Tratamento de encoding para FPDF (Latin-1)
        return texto.encode('latin-1', 'replace').decode('latin-1')

    def header(self):
        self.set_fill_color(*self.col_fundo)
        self.rect(0, 0, 210, 297, 'F')
        self.set_fill_color(10, 10, 15)
        self.rect(0, 0, 210, 35, 'F')
        self.set_xy(10, 10)
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(*self.col_azul)
        self.cell(60, 10, "TECHNOBOLT", 0, 0)
        self.set_text_color(255, 255, 255)
        self.cell(40, 10, "GYM HUB", 0, 1)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 150, 150)
        self.set_xy(10, 20)
        self.cell(0, 5, "RELATORIO DE ALTA PERFORMANCE | PHP PROTOCOL", 0, 1)
        self.set_draw_color(*self.col_destaque)
        self.set_line_width(0.5)
        self.line(10, 35, 200, 35)
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f'TechnoBolt Intelligence AI - Pagina {self.page_no()}', 0, 0, 'C')

    def draw_section_title(self, title, icon=">"):
        self.ln(5)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*self.col_destaque)
        self.cell(10, 10, icon, 0, 0)
        self.set_text_color(*self.col_azul)
        self.cell(0, 10, self.sanitizar_texto(title.upper()), 0, 1)
        self.set_draw_color(50, 50, 60)
        self.line(10, self.get_y(), 100, self.get_y())
        self.ln(5)

    def draw_card_text(self, label, content):
        self.set_fill_color(*self.col_card)
        self.set_text_color(*self.col_texto)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.col_azul)
        self.multi_cell(0, 6, self.sanitizar_texto(label), fill=True)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.col_texto)
        self.multi_cell(0, 6, self.sanitizar_texto(str(content)), fill=True)
        self.ln(2)

# --- APLICAÇÃO FASTAPI ---
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Backend de alta performance para a plataforma TechnoBolt Gym Hub.",
    openapi_tags=[
        {"name": "Auth", "description": "Autenticação e Registro"},
        {"name": "Perfil", "description": "Gestão de Usuários"},
        {"name": "Analise", "description": "Inteligência Artificial Generativa"},
        {"name": "Social", "description": "Feed, Likes e Comentários"},
        {"name": "Admin", "description": "Painel Administrativo"},
    ]
)

# Configuração de CORS (Permitir acesso do Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPERS DE LÓGICA ---

def normalizar_texto(texto: str) -> str:
    """Remove acentos e coloca em minúsculas para comparação."""
    if not texto: return ""
    if not isinstance(texto, str): texto = str(texto)
    return "".join(c for c in unicodedata.normalize('NFD', texto) 
                   if unicodedata.category(c) != 'Mn').lower().strip()

def validar_e_corrigir_exercicios(lista_exercicios: list) -> list:
    """Associa exercícios gerados pela IA com imagens do banco local."""
    if not lista_exercicios or not EXERCISE_DB: 
        return lista_exercicios
    
    base_url = "https://raw.githubusercontent.com/italoat/technobolt-backend/main/assets/exercises"
    
    # Mapas para busca O(1)
    db_map_norm = {normalizar_texto(k): v for k, v in EXERCISE_DB.items()}
    db_title_map = {normalizar_texto(k): k for k, v in EXERCISE_DB.items()}

    for ex in lista_exercicios:
        nome_ia = ex.get('nome', '')
        nome_ia_norm = normalizar_texto(nome_ia)
        
        pasta_github = None
        nome_final = str(nome_ia)

        # 1. Tentativa de Match Exato
        if nome_ia_norm in db_map_norm:
            pasta_github = db_map_norm[nome_ia_norm]
            nome_final = db_title_map[nome_ia_norm].title()
        else:
            # 2. Match por Similaridade (Difflib)
            matches = difflib.get_close_matches(nome_ia_norm, db_map_norm.keys(), n=1, cutoff=0.6)
            if matches:
                match_key = matches[0]
                pasta_github = db_map_norm[match_key]
                nome_final = db_title_map[match_key].title()
            else:
                # 3. Match por Substring (Busca parcial)
                melhor_candidato = None
                for key in db_map_norm.keys():
                    # Evita matches curtos demais (ex: "remada" vs "remada alta")
                    if (key in nome_ia_norm and len(key) > 4) or (nome_ia_norm in key and len(nome_ia_norm) > 4): 
                        melhor_candidato = key
                        break
                
                if melhor_candidato:
                    pasta_github = db_map_norm[melhor_candidato]
                    nome_final = db_title_map[melhor_candidato].title()
                else:
                    # 4. Fallback (Polichinelo)
                    fallback_key = "polichinelo" if "polichinelo" in db_map_norm else list(db_map_norm.keys())[0]
                    pasta_github = db_map_norm[fallback_key]
                    nome_final = f"{nome_ia} (Adaptado - Ver {db_title_map[fallback_key].title()})"

        ex['nome'] = str(nome_final)
        
        if pasta_github:
            ex['imagens_demonstracao'] = [
                f"{base_url}/{pasta_github}/0.jpg",
                f"{base_url}/{pasta_github}/1.jpg"
            ]
        else:
            ex['imagens_demonstracao'] = [] 

    return lista_exercicios

def calcular_medalha(username: str) -> str:
    """Calcula a medalha do usuário baseada em pontos (Gamification)."""
    try:
        user = db.usuarios.find_one({"usuario": username})
        if not user: return ""
        pontos = user.get('pontos', 0)
        if pontos > 1000: return "🥇"
        if pontos > 500: return "🥈"
        if pontos > 100: return "🥉"
        return ""
    except Exception:
        return ""

def limpar_e_parsear_json(texto_ia: str) -> dict:
    """Extrai e valida JSON de uma resposta de texto da IA."""
    try:
        # Regex para extrair apenas o objeto JSON {}
        match = re.search(r'\{.*\}', texto_ia, re.DOTALL)
        if match:
            texto_limpo = match.group(0)
        else:
            # Fallback de limpeza manual
            texto_limpo = texto_ia.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(texto_limpo)
        return parsed # Retorna tipos nativos (int, float, etc.)
        
    except json.JSONDecodeError as e:
        logger.error(f"Erro de parse JSON: {e}")
        # Retorna estrutura de segurança
        return {
            "avaliacao": {"insight": "Houve uma instabilidade na análise, mas geramos um protocolo base."},
            "dieta": [],
            "suplementacao": [],
            "treino": []
        }

# --- ROTAS: AUTH & PERFIL ---

@app.post("/auth/login", tags=["Auth"], response_model=Dict[str, Any])
def login(dados: UserLogin):
    """Autentica o usuário e retorna perfil completo."""
    user = db.usuarios.find_one({"usuario": dados.usuario, "senha": dados.senha})
    
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Credenciais inválidas")
    
    if user.get("status") != "ativo" and not user.get("is_admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Sua conta aguarda ativação.")
    
    # Monta resposta manualmente para garantir estrutura
    # O PyObjectId lida com a serialização do ID se necessário
    return {
        "sucesso": True,
        "dados": {
            "usuario": user['usuario'],
            "nome": user['nome'],
            "is_admin": user.get('is_admin', False),
            "peso": user.get('peso'),
            "altura": user.get('altura'),
            "genero": user.get('genero', 'Masculino'),
            "creditos": user.get('avaliacoes_restantes', 0),
            "pontos": user.get('pontos', 0),
            "foto_perfil": user.get('foto_perfil', None),
            "restricoes_alim": user.get('restricoes_alim', ''),
            "restricoes_fis": user.get('restricoes_fis', ''),
            "medicamentos": user.get('medicamentos', ''),
            "info_add": user.get('info_add', '')
        }
    }

@app.post("/auth/registro", tags=["Auth"])
def registrar(dados: UserRegister):
    """Cria um novo usuário no sistema."""
    if db.usuarios.find_one({"usuario": dados.usuario}):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Usuário já existe")
    
    novo_user = dados.model_dump()
    novo_user.update({
        "status": "pendente",
        "avaliacoes_restantes": 0,
        "pontos": 0,
        "historico_dossies": [],
        "is_admin": False
    })
    
    db.usuarios.insert_one(novo_user)
    return {"sucesso": True, "mensagem": "Cadastro realizado"}

@app.post("/perfil/atualizar", tags=["Perfil"])
def atualizar_perfil(dados: UserUpdate):
    """Atualiza informações cadastrais do usuário."""
    # Filtra apenas campos não nulos
    update_data = {k: v for k, v in dados.model_dump(exclude={'usuario'}).items() if v is not None}
    
    result = db.usuarios.update_one({"usuario": dados.usuario}, {"$set": update_data})
    
    if result.matched_count == 0:
        raise HTTPException(404, "Usuário não encontrado")
        
    return {"sucesso": True}

# --- ROTAS: ANÁLISE IA ---

@app.post("/analise/executar", tags=["Analise"])
async def executar_analise(
    usuario: str = Form(...),
    nome_completo: str = Form(...),
    peso: str = Form(...), 
    altura: str = Form(...), 
    objetivo: str = Form(...),
    genero: str = Form("Masculino"),
    observacoes: str = Form(""), 
    foto: UploadFile = File(...)
):
    """Gera um protocolo completo de treino e dieta usando IA."""
    
    # 1. Tratamento e Validação de Inputs
    try:
        peso_clean = str(peso).replace(',', '.')
        peso_float = float(peso_clean)
        
        altura_clean = str(altura).replace(',', '.').replace('cm', '').strip()
        altura_val = float(altura_clean)
        # Normaliza altura para cm
        if altura_val < 3.0: 
             altura_int = int(altura_val * 100)
        else:
             altura_int = int(altura_val)
             
    except ValueError:
        logger.warning(f"Erro ao converter medidas para user {usuario}. Usando padrão.")
        peso_float = 70.0 
        altura_int = 175

    # 2. Atualiza dados básicos no banco
    db.usuarios.update_one(
        {"usuario": usuario}, 
        {"$set": {
            "nome": nome_completo, 
            "peso": peso_float, 
            "altura": altura_int, 
            "genero": genero,
            "info_add": observacoes
        }}
    )

    # 3. Busca contexto existente
    user_data = db.usuarios.find_one({"usuario": usuario})
    if not user_data:
        raise HTTPException(404, "Usuário não encontrado após update.")

    r_a = user_data.get('restricoes_alim', 'Nenhuma')
    r_m = user_data.get('medicamentos', 'Nenhum')
    r_f = user_data.get('restricoes_fis', 'Nenhuma')
    info = observacoes 

    # 4. Processamento de Imagem
    content = await foto.read()
    img_otimizada = ImageService.optimize(content, quality=85, size=(800, 800))
    
    altura_m = altura_int / 100 if altura_int > 0 else 1.70
    imc = peso_float / (altura_m**2)
    
    # 5. Engenharia de Prompt (Otimizado)
    prompt_mestre = f"""
    VOCÊ É UM TREINADOR DE ELITE E NUTRICIONISTA PhD.
    SUA MISSÃO: CRIAR O PROTOCOLO PERFEITO PARA MAXIMIZAR RESULTADOS (Hipertrofia/Definição).
    
    PERFIL: {nome_completo}, {genero}, {peso_float}kg, {altura_int}cm, IMC {imc:.2f}.
    OBJETIVO: {objetivo}.
    RESTRIÇÕES: Alimentares: "{r_a}", Físicas: "{r_f}", Meds: "{r_m}", Obs: "{info}".

    REGRAS ESTRITAS DE OUTPUT (JSON APENAS):
    
    1. **TREINO (INDIVIDUALIZADO):**
       - Use APENAS exercícios desta lista: [ {EXERCISE_LIST_STRING} ]
       - Se o exercício ideal não estiver na lista, escolha a variação mais próxima da lista.
       - Estrutura: "dia", "foco", e lista de "exercicios".
       - DENTRO DE CADA EXERCÍCIO, adicione o campo "justificativa_individual": Explique de forma individualizada, biomecanicamente, porque este exercício foi escolhido para ESTA pessoa.
       - Campo "execucao": Detalhe técnico.
       - "treino_insight": Explique sobre o porque montou o treino em questão para a pessoa, focando na estratégia adotada.

    2. **DIETA (SUPERÁVIT CALÓRICO/BULKING LIMPO):**
       - Calcule o Gasto Energético Total (GET) estimado.
       - APLIQUE UM SUPERÁVIT CALÓRICO de +300 a +500 kcal (mostre a matemática no insight).
       - Preencha 'macros_totais' explicitando o superávit (ex: "3200kcal (Superávit) | P: 180g...").
       - "dieta_insight": Explique o cálculo do superávit para suportar o anabolismo, mostrando os números base.

    3. **RETORNO JSON PURO:** Sem markdown (```json), sem texto antes ou depois.

    ESTRUTURA JSON OBRIGATÓRIA:
    {{
      "avaliacao": {{
        "segmentacao": {{ "tronco": "...", "superior": "...", "inferior": "..." }},
        "dobras": {{ "abdominal": "...", "suprailiaca": "...", "peitoral": "..." }},
        "analise_postural": "...",
        "simetria": "...",
        "insight": "..."
      }},
      "dieta": [
        {{
            "dia": "Segunda-feira",
            "foco_nutricional": "...",
            "refeicoes": [ {{ "horario": "...", "nome": "...", "alimentos": "..." }} ],
            "macros_totais": "..."
        }}
      ],
      "dieta_insight": "...",
      "suplementacao": [ {{ "nome": "...", "dose": "...", "horario": "...", "motivo": "..." }} ],
      "suplementacao_insight": "...",
      "treino": [
        {{
          "dia": "Segunda-feira",
          "foco": "...",
          "exercicios": [
            {{ 
               "nome": "NOME EXATO DA LISTA", 
               "series_reps": "...",
               "execucao": "...",
               "justificativa_individual": "..." 
            }}
          ],
          "treino_alternativo": "...",
          "justificativa": "..."
        }}
      ],
      "treino_insight": "..."
    }}
    """
    
    resultado_raw = AIService.generate_content(prompt_mestre, img_otimizada)
    if not resultado_raw: 
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Serviço de IA Indisponível.")

    conteudo_json = limpar_e_parsear_json(resultado_raw)

    # 6. Pós-processamento e Validação de Exercícios
    if 'treino' in conteudo_json and isinstance(conteudo_json['treino'], list):
        for dia in conteudo_json['treino']:
            # Limpa nome do dia
            if 'dia' in dia:
                dia['dia'] = str(dia['dia']).split('-')[0].split(' ')[0].replace(',', '').strip()

            # Valida e anexa imagens aos exercícios
            if 'exercicios' in dia and isinstance(dia['exercicios'], list):
                dia['exercicios'] = validar_e_corrigir_exercicios(dia['exercicios'])

    # 7. Persistência
    dossie = {
        "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "peso_reg": peso_float,
        "conteudo_bruto": {
            "json_full": conteudo_json,
            "r1": str(conteudo_json.get('avaliacao', {}).get('insight', '')),
            "r2": str(conteudo_json.get('dieta_insight', '')),
            "r3": str(conteudo_json.get('suplementacao_insight', '')),
            "r4": str(conteudo_json.get('treino_insight', ''))
        }
    }
    
    # Atualiza saldo se não for admin
    update_query = {"$push": {"historico_dossies": dossie}}
    if not user_data.get('is_admin', False):
        update_query["$inc"] = {"avaliacoes_restantes": -1}
        
    db.usuarios.update_one({"usuario": usuario}, update_query)
    
    # Retorna tipos nativos (Pydantic/FastAPI serializa automaticamente)
    return {"sucesso": True, "resultado": dossie}

@app.post("/analise/regenerar-secao", tags=["Analise"])
def regenerar_secao(dados: dict = Body(...)):
    """Regenera uma parte específica do protocolo (ex: Dieta de Terça)."""
    usuario = dados.get("usuario")
    secao = dados.get("secao")
    dia_alvo = dados.get("dia") 
    
    if not usuario or secao not in ["dieta", "treino", "suplementacao", "avaliacao"]:
        return {"sucesso": False, "mensagem": "Parâmetros inválidos."}

    user_data = db.usuarios.find_one({"usuario": usuario})
    if not user_data: return {"sucesso": False, "mensagem": "Usuário não encontrado."}
    
    # Validação de Saldo
    creditos = user_data.get('avaliacoes_restantes', 0)
    is_admin = user_data.get('is_admin', False)

    if creditos <= 0 and not is_admin:
        return {"sucesso": False, "mensagem": "Saldo insuficiente."}

    if not user_data.get('historico_dossies'):
        return {"sucesso": False, "mensagem": "Sem histórico para regenerar."}

    ultimo_dossie = user_data['historico_dossies'][-1]
    
    # Contexto para IA
    r_a = user_data.get('restricoes_alim', 'Nenhuma')
    r_f = user_data.get('restricoes_fis', 'Nenhuma')
    obs = user_data.get('info_add', 'Nenhuma')
    nome = user_data.get('nome', 'Atleta')
    
    prompt_regeneracao = ""

    # Lógica de Prompt Específico
    if dia_alvo and secao in ["dieta", "treino"]:
        # --- REFRESH DE DIA ÚNICO ---
        if secao == "treino":
            prompt_regeneracao = f"""
            ATENÇÃO: Treinador de Elite TechnoBolt.
            TAREFA: Reescrever APENAS o dia '{dia_alvo}' da seção 'TREINO' para o atleta {nome}.
            CONTEXTO: Restrições: {r_f}. Obs: {obs}.
            
            REGRA CRÍTICA:
            1. USE **APENAS** EXERCÍCIOS DESTA LISTA: [ {EXERCISE_LIST_STRING} ]
            2. VOLUME ALTO: Gere entre **6 a 9 exercícios**.
            3. Título do dia: APENAS "{dia_alvo}".
            4. JUSTIFICATIVA INDIVIDUAL: Explique porque escolheu este exercício para a pessoa.
            
            RETORNE APENAS O JSON DO OBJETO DO DIA:
            {{ 
              "dia": "{dia_alvo}", 
              "foco": "...", 
              "exercicios": [
                {{ "nome": "...", "series_reps": "...", "execucao": "...", "justificativa_individual": "..." }}
              ], 
              "treino_alternativo": "...",
              "justificativa": "..."
            }}
            """
        else: # Dieta
             prompt_regeneracao = f"""
            ATENÇÃO: Nutricionista de Elite TechnoBolt.
            TAREFA: Reescrever a DIETA de '{dia_alvo}' para {nome}.
            CONTEXTO: {r_a}.
            REGRAS: 
            1. NÃO retorne campos nulos. 
            2. MANTENHA O SUPERÁVIT CALÓRICO calculado anteriormente.
            3. Preencha 'macros_totais'.
            
            RETORNE APENAS O JSON DO DIA:
            {{ "dia": "{dia_alvo}", "foco_nutricional": "...", "refeicoes": [...], "macros_totais": "P: Xg | C: Yg | G: Zg" }}
            """
    else:
        # Refresh Seção Completa
        prompt_regeneracao = f"""
        ATENÇÃO: Treinador de Elite TechnoBolt.
        TAREFA: Refresh COMPLETO da seção '{secao.upper()}' para {nome}.
        CONTEXTO: {r_f}, {r_a}, {obs}.
        LISTA DE EXERCÍCIOS PERMITIDOS: [ {EXERCISE_LIST_STRING if secao == 'treino' else ''} ]
        
        REGRAS: 
        - 7 Dias. 
        - VOLUME DE TREINO: 6 a 9 exercícios por dia.
        - Sem dados nulos. 
        - Títulos dos dias apenas o nome da semana.
        - Se for dieta, inclua 'macros_totais' com SUPERÁVIT CALÓRICO.
        - Se for treino, inclua 'justificativa_individual' nos exercícios.
        RETORNE JSON: {{ "{secao}": [ ... ] }}
        """

    resultado_texto = AIService.generate_content(prompt_regeneracao)
    if not resultado_texto: return {"sucesso": False, "mensagem": "Erro IA."}

    novo_dado_ia = limpar_e_parsear_json(resultado_texto)
    
    # Valida exercícios novamente se for treino
    if secao == "treino":
        if 'treino' in novo_dado_ia and isinstance(novo_dado_ia['treino'], list):
            for dia in novo_dado_ia['treino']:
                 if 'dia' in dia: dia['dia'] = str(dia['dia']).split('-')[0].split(' ')[0].replace(',', '').strip()
                 if 'exercicios' in dia: dia['exercicios'] = validar_e_corrigir_exercicios(dia['exercicios'])
        elif dia_alvo:
            obj_dia = novo_dado_ia.get('treino', novo_dado_ia)
            if isinstance(obj_dia, list) and len(obj_dia) > 0: obj_dia = obj_dia[0]
            
            if 'dia' in obj_dia: obj_dia['dia'] = dia_alvo 
            if 'exercicios' in obj_dia: obj_dia['exercicios'] = validar_e_corrigir_exercicios(obj_dia['exercicios'])
            
            novo_dado_ia = obj_dia

    # Lógica de atualização no Banco (Merge no JSON existente)
    updates = {}
    if dia_alvo and secao in ["dieta", "treino"]:
        lista_atual = ultimo_dossie.get('conteudo_bruto', {}).get('json_full', {}).get(secao, [])
        idx_alvo = -1
        # Busca índice do dia para update posicional
        for i, item in enumerate(lista_atual):
            if dia_alvo.lower() in str(item.get('dia', '')).lower():
                idx_alvo = i
                break
        
        if idx_alvo != -1:
            updates[f"historico_dossies.-1.conteudo_bruto.json_full.{secao}.{idx_alvo}"] = novo_dado_ia
    else:
        caminho_update = f"historico_dossies.-1.conteudo_bruto.json_full.{secao}"
        caminho_insight = f"historico_dossies.-1.conteudo_bruto.json_full.{secao}_insight"
        
        if secao in novo_dado_ia:
            updates[caminho_update] = novo_dado_ia[secao]
            
        if secao != "avaliacao":
             key_insight = f"{secao}_insight"
             if key_insight in novo_dado_ia:
                 updates[caminho_insight] = novo_dado_ia[key_insight]

    if updates:
        mongo_cmd = {"$set": updates}
        if not is_admin: mongo_cmd["$inc"] = {"avaliacoes_restantes": -1}
        db.usuarios.update_one({"usuario": usuario}, mongo_cmd)
        
        user_atualizado = db.usuarios.find_one({"usuario": usuario})
        return {
            "sucesso": True, 
            "resultado": user_atualizado['historico_dossies'][-1],
            "novo_saldo": user_atualizado.get('avaliacoes_restantes', 0)
        }
    
    return {"sucesso": False, "mensagem": "Não foi possível estruturar os dados da IA."}

@app.get("/historico/{usuario}", tags=["Perfil"])
def buscar_historico(usuario: str):
    """Retorna o histórico completo e perfil atualizado."""
    user = db.usuarios.find_one({"usuario": usuario})
    if not user: return {"sucesso": True, "historico": []}
    
    # Retorna tipos nativos (int, float) sem converter para string
    return {
        "sucesso": True, 
        "historico": user.get('historico_dossies', []), 
        "creditos": user.get('avaliacoes_restantes', 0), 
        "perfil": {
            "peso": user.get('peso'),
            "altura": user.get('altura'),
            "genero": user.get('genero', 'Masculino'),
            "restricoes_alim": user.get('restricoes_alim', ''),
            "restricoes_fis": user.get('restricoes_fis', ''),
            "medicamentos": user.get('medicamentos', ''),
            "info_add": user.get('info_add', ''),
            "creditos": user.get('avaliacoes_restantes', 0)
        }
    }

# --- ROTAS: SOCIAL E FEED ---

@app.get("/social/feed", tags=["Social"])
def get_feed():
    """Retorna o feed de postagens ordenado por data."""
    # O PyObjectId no Model (se implementado) ou conversão manual aqui
    posts = list(db.posts.find().sort("data", DESCENDING).limit(50))
    for p in posts: 
        p['_id'] = str(p['_id']) # Conversão explicita ainda necessária se não usar response_model list
        p['likes'] = p.get('likes', [])
        p['comentarios'] = p.get('comentarios', [])
        p['medalha'] = calcular_medalha(p.get('autor'))
    return {"sucesso": True, "feed": posts}

@app.post("/social/postar", tags=["Social"])
async def postar_feed(
    usuario: str = Form(...), 
    legenda: str = Form(...), 
    imagem: UploadFile = File(...)
):
    """Cria uma nova postagem com imagem."""
    content = await imagem.read()
    img_otimizada = ImageService.optimize(content, size=(600, 600))
    img_b64 = base64.b64encode(img_otimizada).decode('utf-8')

    post_doc = {
        "autor": usuario,
        "legenda": legenda,
        "imagem": img_b64,
        "data": datetime.now().isoformat(),
        "likes": [],
        "comentarios": []
    }
    
    post_id = db.posts.insert_one(post_doc).inserted_id

    # Comentário automático da IA
    prompt_comentario = f"Aja como um personal trainer da TechnoBolt. Comentário curto (máx 15 palavras) para foto com legenda: '{legenda}'"
    comentario_ia = AIService.generate_content(prompt_comentario, img_otimizada)
    
    if comentario_ia:
        db.posts.update_one(
            {"_id": post_id}, 
            {"$push": {"comentarios": {"autor": "TechnoBolt AI 🤖", "texto": comentario_ia}}}
        )

    return {"sucesso": True}

@app.post("/social/post/deletar", tags=["Social"])
def deletar_post_social(dados: SocialPostRequest):
    """Remove uma postagem (apenas o autor pode deletar)."""
    try:
        oid = ObjectId(dados.post_id)
        result = db.posts.delete_one({"_id": oid, "autor": dados.usuario})
        return {"sucesso": True} if result.deleted_count > 0 else {"sucesso": False, "mensagem": "Post não encontrado ou não autorizado."}
    except Exception as e:
        return {"sucesso": False, "mensagem": str(e)}
        
@app.post("/social/curtir", tags=["Social"])
def curtir_post(dados: SocialPostRequest):
    """Alterna like no post (Toggle)."""
    try:
        oid = ObjectId(dados.post_id)
        post = db.posts.find_one({"_id": oid})
        if not post: return {"sucesso": False, "mensagem": "Post não encontrado"}
        
        if dados.usuario in post.get("likes", []):
            db.posts.update_one({"_id": oid}, {"$pull": {"likes": dados.usuario}})
        else:
            db.posts.update_one({"_id": oid}, {"$addToSet": {"likes": dados.usuario}})
        return {"sucesso": True}
    except Exception as e:
        raise HTTPException(500, f"Erro ao processar like: {str(e)}")
        
@app.post("/social/comentar", tags=["Social"])
def postar_comentario(dados: SocialCommentRequest):
    """Adiciona um comentário a uma postagem."""
    try:
        oid = ObjectId(dados.post_id)
        comentario = {
            "autor": dados.usuario,
            "texto": dados.texto,
            "data": datetime.now().isoformat()
        }
        db.posts.update_one({"_id": oid}, {"$push": {"comentarios": comentario}})
        return {"sucesso": True}
    except Exception as e:
        raise HTTPException(500, f"Erro ao postar comentário: {str(e)}")

# --- ROTAS: GAMIFICATION & CHECKINS ---

@app.get("/social/ranking", tags=["Social"])
def get_ranking_global():
    """Retorna top 50 usuários por pontos."""
    users = list(db.usuarios.find(
        {"is_admin": False}, 
        {"nome": 1, "usuario": 1, "pontos": 1, "foto_perfil": 1, "_id": 0}
    ).sort("pontos", DESCENDING).limit(50))
    return {"sucesso": True, "ranking": users}

@app.get("/social/checkins", tags=["Social"])
def get_checkins(usuario: str):
    """Retorna calendário de checkins do mês atual."""
    now = datetime.now()
    start_date = datetime(now.year, now.month, 1).isoformat()
    checkins = list(db.checkins.find(
        {"usuario": usuario, "data": {"$gte": start_date}}
    ))
    formatted = {}
    for c in checkins:
        try:
            day = datetime.fromisoformat(c['data']).day
            formatted[day] = c['tipo']
        except:
            pass
    return {"sucesso": True, "checkins": formatted}

@app.post("/social/validar-conquista", tags=["Social"])
async def validar_conquista(
    usuario: str = Form(...),
    tipo: str = Form(...), 
    foto: UploadFile = File(...)
):
    """Valida um treino via IA (Computer Vision) para gamification."""
    now = datetime.now()
    today_start = datetime(now.year, now.month, now.day).isoformat()
    
    # Limite diário de 1 checkin
    ja_fez = db.checkins.find_one({
        "usuario": usuario, 
        "data": {"$gte": today_start}
    })
    
    if ja_fez:
        return {"sucesso": False, "mensagem": "Você já treinou hoje! Volte amanhã."}

    content = await foto.read()
    img_otimizada = ImageService.optimize(content, size=(800, 800))
    
    prompt_juiz = f"""
    ATUE COMO UM JUIZ RIGOROSO DE FITNESS.
    O usuário diz que fez um treino do tipo: '{tipo}' (gym=academia, home=casa, run=corrida).
    Analise a imagem anexada.
    - Se for 'gym', procure equipamentos de academia, espelhos, pesos, roupas de treino.
    - Se for 'home', procure tapete de yoga, pesos livres, roupa de ginástica em ambiente doméstico.
    - Se for 'run', procure ambiente externo (rua/parque), esteira, tênis de corrida, suor.
    IMPORTANTE: Selfies no espelho VALEM se o contexto bater.
    Responda APENAS: "APROVADO" ou "REPROVADO".
    """
    
    resultado_ia = AIService.generate_content(prompt_juiz, img_otimizada)
    
    if resultado_ia and "APROVADO" in resultado_ia.upper():
        pontos_ganhos = 50
        db.checkins.insert_one({
            "usuario": usuario,
            "tipo": tipo,
            "data": datetime.now().isoformat(),
            "pontos": pontos_ganhos
        })
        db.usuarios.update_one(
            {"usuario": usuario},
            {"$inc": {"pontos": pontos_ganhos}}
        )
        return {"sucesso": True, "aprovado": True, "pontos": pontos_ganhos}
    else:
        return {"sucesso": True, "aprovado": False, "mensagem": "A IA não identificou evidências claras do treino."}

# --- ROTAS: ADMINISTRAÇÃO ---

@app.get("/setup/criar-admin", tags=["Admin"])
def criar_admin_inicial():
    """Setup inicial de conta admin."""
    if db.usuarios.find_one({"usuario": "admin"}): 
        return {"sucesso": False, "mensagem": "Admin já existe!"}
    
    db.usuarios.insert_one({
        "usuario": "admin", 
        "senha": "123", 
        "nome": "Super Admin", 
        "is_admin": True, 
        "status": "ativo", 
        "avaliacoes_restantes": 9999, 
        "historico_dossies": [], 
        "peso": 80.0, 
        "altura": 180, 
        "genero": "Masculino"
    })
    return {"sucesso": True, "mensagem": "Admin criado"}

@app.get("/admin/listar", tags=["Admin"])
def listar_usuarios():
    """Lista todos os usuários (apenas Admin deve acessar, na prática)."""
    users = list(db.usuarios.find())
    for u in users: u['_id'] = str(u['_id'])
    return {"sucesso": True, "usuarios": users}

@app.post("/admin/editar", tags=["Admin"])
def editar_usuario(dados: AdminUserEdit):
    """Edita saldo ou status de um usuário."""
    update_fields = {}
    if dados.status: update_fields["status"] = dados.status
    if dados.creditos is not None: update_fields["avaliacoes_restantes"] = dados.creditos
    
    if update_fields:
        db.usuarios.update_one({"usuario": dados.target_user}, {"$set": update_fields})
    return {"sucesso": True}

@app.post("/admin/excluir", tags=["Admin"])
def excluir_usuario(dados: AdminUserEdit):
    """Remove um usuário do sistema."""
    db.usuarios.delete_one({"usuario": dados.target_user})
    return {"sucesso": True}

# --- ROTAS: EXPORTAÇÃO (PDF) ---

@app.get("/analise/baixar-pdf/{usuario}", tags=["Analise"])
def baixar_pdf_completo(usuario: str):
    """Gera e retorna PDF do último protocolo."""
    try:
        user = db.usuarios.find_one({"usuario": usuario})
        if not user or not user.get('historico_dossies'): 
            raise HTTPException(404, "Sem relatório.")
        
        dossie = user['historico_dossies'][-1]
        raw = dossie.get('conteudo_bruto', {})
        json_data = raw.get('json_full', {}) if isinstance(raw.get('json_full'), dict) else {}

        pdf = PDFService()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, pdf.sanitizar_texto(f"ATLETA: {user.get('nome', 'N/A').upper()}"), ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(180, 180, 180)
        pdf.cell(0, 5, f"DATA: {dossie.get('data', 'N/A')}", ln=True)
        pdf.cell(0, 5, f"OBJETIVO: {pdf.sanitizar_texto(json_data.get('avaliacao', {}).get('insight', 'Alta Performance')[:50])}...", ln=True)
        pdf.ln(10)

        # Seção 1: Avaliação
        pdf.draw_section_title("1. ANALISE CORPORAL COMPLETA", icon="O")
        av = json_data.get('avaliacao', {})
        seg = av.get('segmentacao', {})
        dob = av.get('dobras', {})
        pdf.draw_card_text("Segmentacao:", f"- Tronco: {seg.get('tronco','')}\n- Sup: {seg.get('superior','')}\n- Inf: {seg.get('inferior','')}")
        pdf.ln(2)
        pdf.draw_card_text("Dobras:", f"- Abd: {dob.get('abdominal','')}\n- Supra: {dob.get('suprailiaca','')}\n- Peit: {dob.get('peitoral','')}")
        pdf.ln(2)
        pdf.set_text_color(0, 255, 200)
        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(0, 6, pdf.sanitizar_texto(f">> INSIGHT: {av.get('insight', '')}"))

        # Seção 2: Dieta
        pdf.add_page()
        pdf.draw_section_title("2. PROTOCOLO NUTRICIONAL", icon="U")
        dieta = json_data.get('dieta', [])
        if isinstance(dieta, list):
            for dia in dieta:
                pdf.ln(3)
                pdf.set_fill_color(50, 50, 60)
                pdf.set_text_color(16, 185, 129)
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 8, pdf.sanitizar_texto(f"{dia.get('dia', '').upper()} | {dia.get('foco_nutricional', '').upper()}"), 0, 1, 'L', True)
                pdf.set_fill_color(35, 35, 40)
                pdf.set_text_color(230, 230, 230)
                for ref in dia.get('refeicoes', []):
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.cell(30, 6, f"{pdf.sanitizar_texto(ref.get('horario',''))}:", 0, 0, 'L', True)
                    pdf.set_font("Helvetica", "", 9)
                    pdf.multi_cell(0, 6, pdf.sanitizar_texto(ref.get('alimentos','')), fill=True)
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(150, 150, 150)
                pdf.cell(0, 6, pdf.sanitizar_texto(f"Macros: {dia.get('macros_totais', '')}"), 0, 1, 'R', True)
                pdf.set_draw_color(30, 30, 30)
                pdf.line(10, pdf.get_y()+2, 200, pdf.get_y()+2)
                pdf.ln(2)

        # Seção 3: Treino
        pdf.add_page()
        pdf.draw_section_title("3. PLANILHA DE TREINO", icon="X")
        treino = json_data.get('treino', [])
        if isinstance(treino, list):
            for item in treino:
                pdf.ln(3)
                pdf.set_fill_color(50, 50, 60)
                pdf.set_text_color(0, 255, 200)
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 8, pdf.sanitizar_texto(f"{item.get('dia','').upper()} - {item.get('foco','').upper()}"), 0, 1, 'L', True)
                pdf.set_fill_color(35, 35, 40)
                pdf.set_text_color(230, 230, 230)
                pdf.set_font("Helvetica", "", 9)
                for ex in item.get('exercicios', []):
                    nome_ex = pdf.sanitizar_texto(ex.get('nome', ''))
                    series = pdf.sanitizar_texto(ex.get('series_reps', ''))
                    execucao = pdf.sanitizar_texto(ex.get('execucao', ''))
                    pdf.cell(0, 5, f"  > {nome_ex} [{series}]", 0, 1, 'L', True)
                    if execucao:
                        pdf.set_font("Helvetica", "I", 8)
                        pdf.set_text_color(180, 180, 180)
                        pdf.multi_cell(0, 4, f"     {execucao}", fill=True)
                        pdf.set_font("Helvetica", "", 9)
                        pdf.set_text_color(230, 230, 230)
                pdf.ln(1)
                pdf.set_draw_color(30, 30, 30)
                pdf.line(10, pdf.get_y()+2, 200, pdf.get_y()+2)
                pdf.ln(2)

        # Seção 4: Suplementos
        pdf.add_page()
        pdf.draw_section_title("4. SUPLEMENTACAO", icon="+")
        suple = json_data.get('suplementacao', [])
        if isinstance(suple, list):
            for item in suple:
                pdf.draw_card_text(item.get('nome',''), f"Dose: {item.get('dose','')} | {item.get('horario','')}\nMotivo: {item.get('motivo','')}")

        pdf_buffer = io.BytesIO()
        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, str): pdf_buffer.write(pdf_output.encode('latin-1'))
        else: pdf_buffer.write(pdf_output)
        pdf_buffer.seek(0)
        
        headers = {'Content-Disposition': f'attachment; filename="TechnoBolt_Protocolo.pdf"'}
        return StreamingResponse(pdf_buffer, media_type="application/pdf", headers=headers)

    except Exception as e:
        logger.error(f"ERRO CRÍTICO PDF: {e}")
        raise HTTPException(500, f"Erro PDF: {str(e)}")

# --- ROTAS: CHAT ---

@app.get("/chat/usuarios", tags=["Chat"])
def listar_usuarios_chat(usuario_atual: str):
    """Lista usuários disponíveis para chat (exceto o próprio)."""
    users = list(db.usuarios.find(
        {"usuario": {"$ne": usuario_atual}}, 
        {"usuario": 1, "nome": 1, "_id": 0}
    ))
    return {"sucesso": True, "usuarios": users}

@app.get("/chat/mensagens", tags=["Chat"])
def pegar_mensagens(user1: str, user2: str):
    """Recupera histórico de mensagens entre dois usuários."""
    msgs = list(db.chat.find({
        "$or": [
            {"remetente": user1, "destinatario": user2}, 
            {"remetente": user2, "destinatario": user1}
        ]
    }).sort("timestamp", ASCENDING))
    
    # Conversão manual de ID necessária aqui pois não usamos Pydantic response model no list
    for m in msgs: m['_id'] = str(m['_id'])
    
    return {"sucesso": True, "mensagens": msgs}

@app.post("/chat/enviar", tags=["Chat"])
def enviar_mensagem(dados: ChatMessageRequest):
    """Envia uma nova mensagem no chat."""
    db.chat.insert_one({
        "remetente": dados.remetente,
        "destinatario": dados.destinatario,
        "texto": dados.texto,
        "timestamp": datetime.now().isoformat()
    })
    return {"sucesso": True}
