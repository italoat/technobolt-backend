from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pymongo import MongoClient
from bson.objectid import ObjectId
import google.generativeai as genai
from PIL import Image, ImageOps
import io
import os
import re
import json
import urllib.parse
from datetime import datetime
import base64
import random
import pillow_heif
from fpdf import FPDF
import unicodedata
import difflib

# --- INICIALIZAÇÃO DE SUPORTE HEIC ---
pillow_heif.register_heif_opener()

app = FastAPI(title="TechnoBolt Gym Hub API", version="85.0-Elite-Full-Fixed")

# --- CARREGAMENTO DO BANCO DE EXERCÍCIOS (JSON EXTERNO) ---
# Carrega as chaves em memória para validação rápida e blindagem
EXERCISE_KEYS = []
EXERCISE_DB = {}

try:
    with open("exercises.json", "r", encoding="utf-8") as f:
        EXERCISE_DB = json.load(f)
        # Normaliza as chaves para facilitar a busca (remove acentos e lowercase)
        # Isso cria um índice em memória para busca rápida
        for k, v in EXERCISE_DB.items():
            key_norm = "".join(c for c in unicodedata.normalize('NFD', k) if unicodedata.category(c) != 'Mn').lower().strip()
            EXERCISE_KEYS.append(key_norm)
            
    print(f"✅ Banco de Exercícios Carregado e Indexado: {len(EXERCISE_DB)} itens.")
except Exception as e:
    print(f"⚠️ AVISO CRÍTICO: Erro ao carregar exercises.json. A API funcionará, mas sem imagens. Erro: {e}")

# --- MOTORES DE IA ---
MOTORES_TECHNOBOLT = [
    "models/gemini-3-flash-preview", 
    "models/gemini-2.5-flash", 
    "models/gemini-2.0-flash", 
    "models/gemini-flash-latest"
]

# --- CONEXÃO BANCO ---
def get_database():
    try:
        user = os.environ.get("MONGO_USER", "technobolt")
        password = urllib.parse.quote_plus(os.environ.get("MONGO_PASS", "tech@132"))
        host = os.environ.get("MONGO_HOST", "cluster0.zbjsvk6.mongodb.net")
        uri = f"mongodb+srv://{user}:{password}@{host}/?appName=Cluster0"
        return MongoClient(uri).technoboltgym
    except Exception as e:
        print(f"Erro Mongo: {e}")
        return None

db = get_database()

# --- FUNÇÕES UTILITÁRIAS GERAIS ---

def rodar_ia(prompt, imagem_bytes=None):
    chaves = [os.environ.get(f"GEMINI_CHAVE_{i}") for i in range(1, 8) if os.environ.get(f"GEMINI_CHAVE_{i}")]
    img_blob = {"mime_type": "image/jpeg", "data": imagem_bytes} if imagem_bytes else None
    random.shuffle(chaves)
    
    for chave in chaves:
        genai.configure(api_key=chave)
        for modelo in MOTORES_TECHNOBOLT:
            try:
                model = genai.GenerativeModel(modelo)
                generation_config = {
                    "response_mime_type": "application/json" if "json" in prompt.lower() else "text/plain",
                    "max_output_tokens": 8192, 
                    "temperature": 0.7
                }
                inputs = [prompt, img_blob] if img_blob else [prompt]
                response = model.generate_content(inputs, generation_config=generation_config)
                if response and response.text:
                    return response.text
            except Exception as e:
                print(f"Erro IA ({modelo}): {e}")
                continue 
    return None

def limpar_e_parsear_json(texto_ia):
    try:
        texto_limpo = texto_ia.replace("```json", "").replace("```", "").strip()
        return json.loads(texto_limpo)
    except Exception as e:
        print(f"Erro ao parsear JSON da IA: {e}")
        return {
            "avaliacao": {"insight": "Erro na leitura da IA"},
            "dieta": [],
            "suplementacao": [],
            "treino": []
        }

def otimizar_imagem(file_bytes, quality=70, size=(800, 800)):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail(size)
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality)
        return output.getvalue()
    except Exception:
        return file_bytes

# --- [CORE] VALIDAÇÃO E BLINDAGEM DE EXERCÍCIOS ---

def normalizar_texto(texto):
    if not texto: return ""
    return "".join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn').lower().strip()

def validar_e_corrigir_exercicios(lista_exercicios):
    """
    Função Sênior de Blindagem:
    1. Recebe a lista suja da IA.
    2. Tenta Match Exato no Banco JSON.
    3. Tenta Fuzzy Match (correção de digitação).
    4. Tenta Substring Match (achar 'Supino' dentro de 'Supino Inclinado 45').
    5. Se tudo falhar, usa um Fallback Seguro (Polichinelo) para garantir imagem.
    6. Retorna a lista corrigida com URLs válidas do GitHub.
    """
    if not lista_exercicios or not EXERCISE_DB: return lista_exercicios
    
    base_url = "https://raw.githubusercontent.com/italoat/technobolt-backend/main/assets/exercises"
    
    # Mapeamento reverso temporário para recuperar a chave original (Case Sensitive) a partir da normalizada
    # Isso é necessário porque EXERCISE_KEYS só tem keys normalizadas
    db_map_norm = {normalizar_texto(k): v for k, v in EXERCISE_DB.items()}
    # Mapeamento para recuperar o Título (Nome bonito) a partir da normalizada
    db_title_map = {normalizar_texto(k): k for k, v in EXERCISE_DB.items()}

    for ex in lista_exercicios:
        nome_ia = ex.get('nome', '')
        nome_ia_norm = normalizar_texto(nome_ia)
        
        pasta_github = None
        nome_corrigido = None

        # 1. Match Exato
        if nome_ia_norm in db_map_norm:
            pasta_github = db_map_norm[nome_ia_norm]
            nome_corrigido = db_title_map[nome_ia_norm].title()
        
        else:
            # 2. Fuzzy Match (Correção de pequenos erros)
            matches = difflib.get_close_matches(nome_ia_norm, EXERCISE_KEYS, n=1, cutoff=0.6)
            if matches:
                match_key = matches[0]
                pasta_github = db_map_norm[match_key]
                nome_corrigido = db_title_map[match_key].title()
            
            else:
                # 3. Substring Match (Inteligência Semântica)
                # Tenta encontrar uma palavra chave do banco DENTRO do nome da IA ou vice-versa
                melhor_candidato = None
                
                # Estratégia: Se a IA mandou "Agachamento com Salto e Carga", procura "Agachamento com Salto" no banco
                for key in EXERCISE_KEYS:
                    # Se a chave do banco está contida no nome da IA (Ex: "Supino" em "Supino Reto Articulado")
                    if key in nome_ia_norm and len(key) > 4: 
                        melhor_candidato = key
                        # Prioriza matches maiores (mais específicos)
                        break 
                    
                    # Se o nome da IA está contido na chave do banco (Ex: "Flexão" em "Flexão de Braços")
                    if nome_ia_norm in key and len(nome_ia_norm) > 4:
                        melhor_candidato = key
                        break
                
                if melhor_candidato:
                    pasta_github = db_map_norm[melhor_candidato]
                    # Mantemos o nome da IA se for mais específico, mas usamos a foto do banco
                    # Ou forçamos o nome do banco para garantir consistência visual. Vamos forçar o banco.
                    nome_corrigido = db_title_map[melhor_candidato].title()
                else:
                    # 4. Fallback Seguro (Último Recurso)
                    # Se não achou nada, define um exercício padrão funcional para não quebrar a UI
                    # Tenta achar "Polichinelo" ou "Burpee"
                    fallback_key = "polichinelo" if "polichinelo" in db_map_norm else list(db_map_norm.keys())[0]
                    pasta_github = db_map_norm[fallback_key]
                    nome_corrigido = f"{nome_ia} (Adaptado)" # Avisa que foi adaptado

        # Aplica as correções
        if nome_corrigido:
            ex['nome'] = nome_corrigido
            
        if pasta_github:
            ex['imagens_demonstracao'] = [
                f"{base_url}/{pasta_github}/0.jpg",
                f"{base_url}/{pasta_github}/1.jpg"
            ]
        else:
            # Caso raríssimo onde o DB está vazio ou corrompido
            ex['imagens_demonstracao'] = []

    return lista_exercicios

# --- PDF GENERATOR ---

def sanitizar_texto(texto):
    if not texto: return ""
    if not isinstance(texto, str): texto = str(texto)
    texto = texto.replace("🚀", ">>").replace("✅", "[OK]").replace("⚠️", "[!]")
    texto = texto.replace("💊", "").replace("🥗", "").replace("🏋️", "").replace("📊", "")
    texto = texto.replace("**", "").replace("###", "").replace("##", "")
    texto = texto.replace("–", "-").replace("“", '"').replace("”", '"')
    return texto.encode('latin-1', 'replace').decode('latin-1')

class ModernPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.col_fundo = (20, 20, 25)
        self.col_card = (35, 35, 40)
        self.col_azul = (59, 130, 246)
        self.col_texto = (230, 230, 230)
        self.col_destaque = (0, 255, 200)
        self.col_verde = (16, 185, 129)

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
        self.cell(0, 10, sanitizar_texto(title.upper()), 0, 1)
        self.set_draw_color(50, 50, 60)
        self.line(10, self.get_y(), 100, self.get_y())
        self.ln(5)

    def draw_card_text(self, label, content):
        self.set_fill_color(*self.col_card)
        self.set_text_color(*self.col_texto)
        self.set_font("Helvetica", "", 10)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.col_azul)
        self.multi_cell(0, 6, sanitizar_texto(label), fill=True)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.col_texto)
        self.texto = sanitizar_texto(str(content))
        self.multi_cell(0, 6, self.texto, fill=True)
        self.ln(2)

    def draw_table_row(self, col1, col2, col3=None):
        self.set_fill_color(*self.col_card)
        self.set_text_color(*self.col_texto)
        self.set_font("Helvetica", "", 9)
        self.set_draw_color(*self.col_fundo) 
        self.set_line_width(0.3)
        h = 8 
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*self.col_destaque)
        self.cell(40, h, sanitizar_texto(col1), 1, 0, 'L', True)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*self.col_texto)
        if col3:
            self.cell(100, h, sanitizar_texto(col2), 1, 0, 'L', True)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(50, h, sanitizar_texto(col3), 1, 1, 'L', True)
        else:
            self.cell(150, h, sanitizar_texto(col2), 1, 1, 'L', True)

# --- FUNÇÕES AUXILIARES DE NEGÓCIO ---

def calcular_medalha(username):
    try:
        desafios = list(db.desafios.find({"participantes": username}))
        if not desafios: return ""
        melhor_nivel = 4 
        for d in desafios:
            ranking = d.get('ranking', {})
            if not ranking: continue
            ordenados = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
            total_participantes = len(ordenados)
            if total_participantes == 0: continue
            try:
                posicao = [i for i, (u, s) in enumerate(ordenados, 1) if u == username][0]
                percentual = posicao / total_participantes
                if percentual <= 0.20: melhor_nivel = min(melhor_nivel, 1)
                elif percentual <= 0.30: melhor_nivel = min(melhor_nivel, 2)
                else: melhor_nivel = min(melhor_nivel, 3)
            except IndexError:
                continue
        mapeamento = {1: "🥇", 2: "🥈", 3: "🥉"}
        return mapeamento.get(melhor_nivel, "")
    except Exception:
        return ""

# --- ENDPOINTS: AUTH & PERFIL ---

@app.post("/auth/login")
def login(dados: dict):
    user = db.usuarios.find_one({"usuario": dados['usuario'], "senha": dados['senha']})
    if not user: raise HTTPException(401, "Credenciais inválidas")
    if user.get("status") != "ativo" and not user.get("is_admin"):
        raise HTTPException(403, "Sua conta está aguardando ativação pelo administrador.")
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
            "restricoes_alim": user.get('restricoes_alim', ''),
            "restricoes_fis": user.get('restricoes_fis', ''),
            "medicamentos": user.get('medicamentos', ''),
            "info_add": user.get('info_add', '')
        }
    }

@app.post("/auth/registro")
def registrar(dados: dict):
    if db.usuarios.find_one({"usuario": dados['usuario']}):
        raise HTTPException(400, "Usuário já existe")
    novo_user = {
        **dados,
        "status": "pendente",
        "avaliacoes_restantes": 0,
        "historico_dossies": [],
        "is_admin": False
    }
    db.usuarios.insert_one(novo_user)
    return {"sucesso": True, "mensagem": "Cadastro realizado"}
    
@app.post("/perfil/atualizar")
def atualizar_perfil(dados: dict):
    db.usuarios.update_one(
        {"usuario": dados['usuario']},
        {"$set": {
            "nome": dados.get('nome'),
            "peso": dados.get('peso'),
            "altura": dados.get('altura'),
            "restricoes_alim": dados.get('restricoes_alim'),
            "restricoes_fis": dados.get('restricoes_fis'),
            "medicamentos": dados.get('medicamentos'),
            "info_add": dados.get('info_add')
        }}
    )
    return {"sucesso": True}

# --- ENDPOINTS: ANÁLISE (ESTRUTURADA PARA JSON) ---

@app.post("/analise/executar")
async def executar_analise(
    usuario: str = Form(...),
    nome_completo: str = Form(...),
    peso: float = Form(...),
    altura: int = Form(...),
    objetivo: str = Form(...),
    genero: str = Form("Masculino"),
    observacoes: str = Form(""), 
    foto: UploadFile = File(...)
):
    # Atualiza dados básicos e as observações (info_add)
    db.usuarios.update_one(
        {"usuario": usuario}, 
        {"$set": {
            "nome": nome_completo, 
            "peso": peso, 
            "altura": altura, 
            "genero": genero,
            "info_add": observacoes
        }}
    )

    user_data = db.usuarios.find_one({"usuario": usuario})
    r_a = user_data.get('restricoes_alim', 'Nenhuma')
    r_m = user_data.get('medicamentos', 'Nenhum')
    r_f = user_data.get('restricoes_fis', 'Nenhuma')
    info = user_data.get('info_add', '') 

    content = await foto.read()
    img_otimizada = otimizar_imagem(content, quality=85, size=(800, 800))
    imc = peso / ((altura/100)**2)
    
    # [PROMPT MESTRE - BLINDADO CONTRA NULL]
    prompt_mestre = f"""
    VOCÊ É A IA DE ELITE DA TECHNOBOLT. 
    DADOS: {nome_completo}, IMC {imc:.2f}, Objetivo: {objetivo}.
    RESTRIÇÕES: Alimentares: "{r_a}", Físicas: "{r_f}", Meds: "{r_m}", Obs: "{info}".

    REGRAS CRÍTICAS DE OUTPUT (SIGA RIGOROSAMENTE):
    1. PROIBIDO RETORNAR NULL ou "Sem dados". Se faltar info, ESTIME com base no IMC e Objetivo.
    2. AVALIAÇÃO: Gere estimativas realistas de dobras e postura.
    3. DIETA: OBRIGATÓRIO preencher 'macros_totais' (ex: "P: 180g | C: 200g | G: 60g"). 
       - Crie cardápios completos para 7 dias.
    4. SUPLEMENTAÇÃO: Se o aluno não tiver restrição médica grave, sugira o básico (Creatina, Whey, Multivitamínico). Preencha dose/horário.
    5. TREINO:
       - Título do dia: APENAS O NOME DO DIA DA SEMANA (Ex: "Segunda-feira"). Coloque o grupo muscular no campo 'foco'.
       - Gere para os 7 dias.
       - Use NOMES CLÁSSICOS DE ACADEMIA em Português.
       - OBRIGATÓRIO: campo 'execucao' detalhado sem emojis.
    
    RETORNE APENAS JSON VÁLIDO:
    {{
      "avaliacao": {{
        "segmentacao": {{ "tronco": "...", "superior": "...", "inferior": "..." }},
        "dobras": {{ "abdominal": "X mm", "suprailiaca": "X mm", "peitoral": "X mm" }},
        "analise_postural": "...",
        "simetria": "...",
        "insight": "..."
      }},
      "dieta": [
        {{
            "dia": "Segunda-feira",
            "foco_nutricional": "...",
            "refeicoes": [
                {{ "horario": "08:00", "nome": "Café", "alimentos": "..." }}
            ],
            "macros_totais": "P: Xg | C: Yg | G: Zg"
        }},
        ...
      ],
      "dieta_insight": "...",
      "suplementacao": [
        {{ "nome": "Creatina", "dose": "5g", "horario": "Pós-treino", "motivo": "Força" }}
      ],
      "suplementacao_insight": "...",
      "treino": [
        {{
          "dia": "Segunda-feira",
          "foco": "Peito e Tríceps",
          "exercicios": [
            {{ 
               "nome": "Supino Reto com Barra", 
               "series_reps": "4x10",
               "execucao": "..."
            }}
          ],
          "treino_alternativo": "...",
          "justificativa": "..."
        }},
        ...
      ],
      "treino_insight": "..."
    }}
    """
    
    resultado_raw = rodar_ia(prompt_mestre, img_otimizada)
    if not resultado_raw: raise HTTPException(503, "IA Indisponível.")

    conteudo_json = limpar_e_parsear_json(resultado_raw)

    # [BLINDAGEM] Validação e Correção Automática dos Exercícios para garantir imagens
    if 'treino' in conteudo_json and isinstance(conteudo_json['treino'], list):
        for dia in conteudo_json['treino']:
            # Força o título do dia a ser limpo se a IA falhar
            if 'dia' in dia:
                dia['dia'] = dia['dia'].split('-')[0].split(' ')[0].replace(',', '').strip()

            if 'exercicios' in dia and isinstance(dia['exercicios'], list):
                # Aqui aplicamos a correção de nomes e geração de links
                dia['exercicios'] = validar_e_corrigir_exercicios(dia['exercicios'])

    dossie = {
        "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "peso_reg": peso,
        "conteudo_bruto": {
            "json_full": conteudo_json,
            "r1": conteudo_json.get('avaliacao', {}).get('insight', ''),
            "r2": conteudo_json.get('dieta_insight', ''),
            "r3": conteudo_json.get('suplementacao_insight', ''),
            "r4": conteudo_json.get('treino_insight', '')
        }
    }
    
    if not user_data.get('is_admin', False):
        db.usuarios.update_one({"usuario": usuario}, {"$push": {"historico_dossies": dossie}, "$inc": {"avaliacoes_restantes": -1}})
    else:
        db.usuarios.update_one({"usuario": usuario}, {"$push": {"historico_dossies": dossie}})
    
    return {"sucesso": True, "resultado": dossie}

@app.post("/analise/regenerar-secao")
def regenerar_secao(dados: dict):
    usuario = dados.get("usuario")
    secao = dados.get("secao")
    dia_alvo = dados.get("dia") 
    
    if not usuario or secao not in ["dieta", "treino", "suplementacao", "avaliacao"]:
        return {"sucesso": False, "mensagem": "Seção inválida."}

    user_data = db.usuarios.find_one({"usuario": usuario})
    if not user_data: return {"sucesso": False, "mensagem": "Usuário não encontrado."}
    
    creditos = user_data.get('avaliacoes_restantes', 0)
    is_admin = user_data.get('is_admin', False)

    if creditos <= 0 and not is_admin:
        return {"sucesso": False, "mensagem": "Saldo insuficiente."}

    if not user_data.get('historico_dossies'):
        return {"sucesso": False, "mensagem": "Sem histórico."}

    ultimo_dossie = user_data['historico_dossies'][-1]
    
    r_a = user_data.get('restricoes_alim', 'Nenhuma')
    r_f = user_data.get('restricoes_fis', 'Nenhuma')
    obs = user_data.get('info_add', 'Nenhuma')
    nome = user_data.get('nome', 'Atleta')
    
    if dia_alvo and secao in ["dieta", "treino"]:
        # --- REFRESH DE DIA ÚNICO ---
        prompt_regeneracao = f"""
        ATENÇÃO: TechnoBolt IA.
        TAREFA: Reescrever APENAS o dia '{dia_alvo}' da seção '{secao.upper()}' para o atleta {nome}.
        CONTEXTO: {r_f}, {r_a}, {obs}.
        
        REGRA CRÍTICA:
        1. MANTENHA A ESTRUTURA DE OBJETO ÚNICO DO DIA.
        2. Título do dia: APENAS "{dia_alvo}".
        3. Preencha TODOS os campos (nada de null).
        
        RETORNE APENAS O JSON DO OBJETO DO DIA:
        {{ 
          "dia": "{dia_alvo}", 
          "foco": "...", 
          "exercicios": [
            {{ "nome": "...", "series_reps": "...", "execucao": "..." }}
          ], 
          "treino_alternativo": "...",
          "justificativa": "..."
        }}
        """
    else:
        prompt_regeneracao = f"""
        ATENÇÃO: TechnoBolt IA.
        TAREFA: Refresh COMPLETO da seção '{secao.upper()}' para {nome}.
        CONTEXTO: {r_f}, {r_a}, {obs}.
        REGRAS: 7 Dias. Sem dados nulos. Títulos dos dias simples.
        RETORNE JSON: {{ "{secao}": [ ... ] }}
        """

    resultado_texto = rodar_ia(prompt_regeneracao)
    if not resultado_texto: return {"sucesso": False, "mensagem": "Erro IA."}

    novo_dado_ia = limpar_e_parsear_json(resultado_texto)
    
    # [BLINDAGEM] Aplica correção de exercícios se for treino
    if secao == "treino":
        if 'treino' in novo_dado_ia and isinstance(novo_dado_ia['treino'], list):
            for dia in novo_dado_ia['treino']:
                 if 'dia' in dia: dia['dia'] = dia['dia'].split('-')[0].split(' ')[0].replace(',', '').strip()
                 if 'exercicios' in dia: dia['exercicios'] = validar_e_corrigir_exercicios(dia['exercicios'])
        elif dia_alvo:
            # Refresh único
            obj_dia = novo_dado_ia.get('treino', novo_dado_ia)
            if isinstance(obj_dia, list) and len(obj_dia) > 0: obj_dia = obj_dia[0]
            
            if 'dia' in obj_dia: obj_dia['dia'] = dia_alvo # Garante o nome certo
            if 'exercicios' in obj_dia: obj_dia['exercicios'] = validar_e_corrigir_exercicios(obj_dia['exercicios'])
            
            novo_dado_ia = obj_dia

    # Lógica de atualização no Banco (Mantida intacta)
    updates = {}
    if dia_alvo and secao in ["dieta", "treino"]:
        lista_atual = ultimo_dossie.get('conteudo_bruto', {}).get('json_full', {}).get(secao, [])
        idx_alvo = -1
        # Busca flexível pelo dia
        for i, item in enumerate(lista_atual):
            if dia_alvo.lower() in item.get('dia', '').lower():
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
    
    return {"sucesso": False, "mensagem": "Estrutura inválida."}

@app.get("/historico/{usuario}")
def buscar_historico(usuario: str):
    user = db.usuarios.find_one({"usuario": usuario})
    if not user: return {"sucesso": True, "historico": []}
    return {"sucesso": True, "historico": user.get('historico_dossies', [])}

# --- ENDPOINTS: SOCIAL E DESAFIOS ---

@app.get("/social/feed")
def get_feed():
    posts = list(db.posts.find().sort("data", -1).limit(50))
    for p in posts: 
        p['_id'] = str(p['_id'])
        p['likes'] = p.get('likes', [])
        p['comentarios'] = p.get('comentarios', [])
        p['medalha'] = calcular_medalha(p.get('autor'))
    return {"sucesso": True, "feed": posts}

@app.post("/social/postar")
async def postar_feed(usuario: str = Form(...), legenda: str = Form(...), imagem: UploadFile = File(...)):
    content = await imagem.read()
    img_otimizada = otimizar_imagem(content, size=(600, 600))
    img_b64 = base64.b64encode(img_otimizada).decode('utf-8')

    post_id = db.posts.insert_one({
        "autor": usuario,
        "legenda": legenda,
        "imagem": img_b64,
        "data": datetime.now().isoformat(),
        "likes": [],
        "comentarios": []
    }).inserted_id

    prompt_comentario = f"Aja como um personal trainer da TechnoBolt. Comentário curto (máx 15 palavras) para foto com legenda: '{legenda}'"
    comentario_ia = rodar_ia(prompt_comentario, img_otimizada)
    
    if comentario_ia:
        db.posts.update_one({"_id": post_id}, {"$push": {"comentarios": {"autor": "TechnoBolt AI 🤖", "texto": comentario_ia}}})

    return {"sucesso": True}

@app.post("/social/post/deletar")
def deletar_post_social(dados: dict):
    try:
        post_id = dados.get("post_id")
        usuario = dados.get("usuario")
        if not post_id or not usuario: return {"sucesso": False, "mensagem": "Dados incompletos"}
        try: oid = ObjectId(post_id)
        except: return {"sucesso": False, "mensagem": "ID inválido"}
        result = db.posts.delete_one({"_id": oid, "autor": usuario})
        return {"sucesso": True} if result.deleted_count > 0 else {"sucesso": False}
    except Exception as e: return {"sucesso": False, "mensagem": str(e)}
        
@app.post("/social/curtir")
def curtir_post(dados: dict):
    try:
        post_id = dados.get("post_id")
        usuario = dados.get("usuario")
        post = db.posts.find_one({"_id": ObjectId(post_id)})
        if not post: return {"sucesso": False, "mensagem": "Post não encontrado"}
        if usuario in post.get("likes", []):
            db.posts.update_one({"_id": ObjectId(post_id)}, {"$pull": {"likes": usuario}})
        else:
            db.posts.update_one({"_id": ObjectId(post_id)}, {"$addToSet": {"likes": usuario}})
        return {"sucesso": True}
    except Exception as e:
        raise HTTPException(500, f"Erro ao processar like: {str(e)}")
        
@app.post("/social/comentar")
def postar_comentario(dados: dict):
    try:
        post_id = dados.get("post_id")
        usuario = dados.get("usuario")
        texto = dados.get("texto")
        db.posts.update_one({"_id": ObjectId(post_id)}, {"$push": {"comentarios": {"autor": usuario, "texto": texto, "data": datetime.now().isoformat()}}})
        return {"sucesso": True}
    except Exception as e:
        raise HTTPException(500, f"Erro ao postar comentário: {str(e)}")

@app.post("/social/desafio/criar")
def criar_desafio(dados: dict):
    prompt_validacao = f"Analise se este desafio é relacionado a saúde/fitness: '{dados['titulo']} - {dados.get('descricao')}'. Responda APENAS 'SIM' ou 'NAO'."
    res = rodar_ia(prompt_validacao)
    if not res or "SIM" not in res.upper(): return {"sucesso": False, "mensagem": "A IA detectou que este desafio não é focado em saúde."}
    novo_desafio = {**dados, "criador": dados['usuario'], "participantes": [dados['usuario']], "ranking": {dados['usuario']: 0}, "status": "ativo"}
    db.desafios.insert_one(novo_desafio)
    return {"sucesso": True}

@app.get("/social/desafios")
def listar_desafios_disponiveis(usuario: str):
    desafios = list(db.desafios.find({"participantes": {"$ne": usuario}}).sort("_id", -1))
    for d in desafios: d['_id'] = str(d['_id'])
    return {"sucesso": True, "desafios": desafios}

@app.post("/social/desafio/participar")
def participar_desafio(dados: dict):
    db.desafios.update_one({"_id": ObjectId(dados.get("id_desafio"))}, {"$addToSet": {"participantes": dados.get("usuario")}, "$set": {f"ranking.{dados.get('usuario')}": 0}})
    return {"sucesso": True}

@app.get("/social/meus-desafios")
def listar_meus_desafios(usuario: str):
    desafios = list(db.desafios.find({"participantes": usuario}))
    for d in desafios:
        d['_id'] = str(d['_id'])
        if 'ranking' not in d: d['ranking'] = {usuario: 0}
        d['dias_concluidos_atleta'] = d.get(f"progresso_{usuario}", [])
    return {"sucesso": True, "meus_desafios": desafios}

@app.post("/social/desafio/validar-ia")
async def validar_desafio(usuario: str = Form(...), id_desafio: str = Form(...), foto_prova: UploadFile = File(...)):
    content = await foto_prova.read()
    img = otimizar_imagem(content)
    prompt = "Aja como juiz fitness. Na imagem, o usuario esta treinando ou comendo saudavel? Responda 'SIM' ou 'NAO'. Se NAO, curto motivo."
    res = rodar_ia(prompt, img)
    aprovado = res and "SIM" in res.upper()
    pontos = 10 if aprovado else 0
    motivo = res if not aprovado else "Desafio validado!"

    if aprovado:
        db.desafios.update_one({"_id": ObjectId(id_desafio)}, {"$inc": {f"ranking.{usuario}": pontos}, "$addToSet": {f"progresso_{usuario}": datetime.now().day}})
        img_b64 = base64.b64encode(img).decode('utf-8')
        db.posts.insert_one({"autor": usuario, "legenda": f"🔥 Validou o dia no desafio! (+{pontos} pts)", "imagem": img_b64, "data": datetime.now().isoformat(), "tipo": "prova_desafio", "likes": [], "comentarios": [{"autor": "TechnoBolt 🤖", "texto": "Excelente forma! Continue assim."}]})

    return {"sucesso": True, "aprovado": aprovado, "pontos": pontos, "motivo": motivo}

# --- ENDPOINTS: ADMIN ---

@app.get("/setup/criar-admin")
def criar_admin_inicial():
    if db.usuarios.find_one({"usuario": "admin"}): return {"sucesso": False, "mensagem": "Admin já existe!"}
    db.usuarios.insert_one({"usuario": "admin", "senha": "123", "nome": "Super Admin", "is_admin": True, "status": "ativo", "avaliacoes_restantes": 9999, "historico_dossies": [], "peso": 80.0, "altura": 180, "genero": "Masculino"})
    return {"sucesso": True, "mensagem": "Admin criado"}

@app.get("/admin/listar")
def listar_usuarios():
    users = list(db.usuarios.find())
    for u in users: u['_id'] = str(u['_id'])
    return {"sucesso": True, "usuarios": users}

@app.post("/admin/editar")
def editar_usuario(dados: dict):
    db.usuarios.update_one({"usuario": dados['target_user']}, {"$set": {"status": dados.get('status'), "avaliacoes_restantes": int(dados.get('creditos', 0))}})
    return {"sucesso": True}

@app.post("/admin/excluir")
def excluir_usuario(dados: dict):
    db.usuarios.delete_one({"usuario": dados['target_user']})
    return {"sucesso": True}

# --- ENDPOINT: BAIXAR PDF ---

@app.get("/analise/baixar-pdf/{usuario}")
def baixar_pdf_completo(usuario: str):
    try:
        user = db.usuarios.find_one({"usuario": usuario})
        if not user or not user.get('historico_dossies'): raise HTTPException(404, "Sem relatório.")
        dossie = user['historico_dossies'][-1]
        raw = dossie.get('conteudo_bruto', {})
        json_data = raw.get('json_full', {}) if isinstance(raw.get('json_full'), dict) else {}

        pdf = ModernPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, sanitizar_texto(f"ATLETA: {user.get('nome', 'N/A').upper()}"), ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(180, 180, 180)
        pdf.cell(0, 5, f"DATA: {dossie.get('data', 'N/A')}", ln=True)
        pdf.cell(0, 5, f"OBJETIVO: {sanitizar_texto(json_data.get('avaliacao', {}).get('insight', 'Alta Performance')[:50])}...", ln=True)
        pdf.ln(10)

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
        pdf.multi_cell(0, 6, sanitizar_texto(f">> INSIGHT: {av.get('insight', '')}"))

        pdf.add_page()
        pdf.draw_section_title("2. PROTOCOLO NUTRICIONAL", icon="U")
        dieta = json_data.get('dieta', [])
        if isinstance(dieta, list):
            for dia in dieta:
                pdf.ln(3)
                pdf.set_fill_color(50, 50, 60)
                pdf.set_text_color(16, 185, 129)
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 8, sanitizar_texto(f"{dia.get('dia', '').upper()} | {dia.get('foco_nutricional', '').upper()}"), 0, 1, 'L', True)
                pdf.set_fill_color(35, 35, 40)
                pdf.set_text_color(230, 230, 230)
                for ref in dia.get('refeicoes', []):
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.cell(30, 6, f"{sanitizar_texto(ref.get('horario',''))}:", 0, 0, 'L', True)
                    pdf.set_font("Helvetica", "", 9)
                    pdf.multi_cell(0, 6, sanitizar_texto(ref.get('alimentos','')), fill=True)
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(150, 150, 150)
                pdf.cell(0, 6, sanitizar_texto(f"Macros: {dia.get('macros_totais', '')}"), 0, 1, 'R', True)
                pdf.set_draw_color(30, 30, 30)
                pdf.line(10, pdf.get_y()+2, 200, pdf.get_y()+2)
                pdf.ln(2)

        pdf.add_page()
        pdf.draw_section_title("3. PLANILHA DE TREINO", icon="X")
        treino = json_data.get('treino', [])
        if isinstance(treino, list):
            for item in treino:
                pdf.ln(3)
                pdf.set_fill_color(50, 50, 60)
                pdf.set_text_color(0, 255, 200)
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 8, sanitizar_texto(f"{item.get('dia','').upper()} - {item.get('foco','').upper()}"), 0, 1, 'L', True)
                pdf.set_fill_color(35, 35, 40)
                pdf.set_text_color(230, 230, 230)
                pdf.set_font("Helvetica", "", 9)
                for ex in item.get('exercicios', []):
                    nome_ex = sanitizar_texto(ex.get('nome', ''))
                    series = sanitizar_texto(ex.get('series_reps', ''))
                    execucao = sanitizar_texto(ex.get('execucao', ''))
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
        print(f"ERRO CRÍTICO PDF: {e}")
        raise HTTPException(500, f"Erro PDF: {str(e)}")

# --- CHAT ---
@app.get("/chat/usuarios")
def listar_usuarios_chat(usuario_atual: str):
    users = list(db.usuarios.find({"usuario": {"$ne": usuario_atual}}, {"usuario": 1, "nome": 1, "_id": 0}))
    return {"sucesso": True, "usuarios": users}

@app.get("/chat/mensagens")
def pegar_mensagens(user1: str, user2: str):
    msgs = list(db.chat.find({"$or": [{"remetente": user1, "destinatario": user2}, {"remetente": user2, "destinatario": user1}]}).sort("timestamp", 1))
    for m in msgs: m['_id'] = str(m['_id'])
    return {"sucesso": True, "mensagens": msgs}

@app.post("/chat/enviar")
def enviar_mensagem(dados: dict):
    db.chat.insert_one({"remetente": dados['remetente'], "destinatario": dados['destinatario'], "texto": dados['texto'], "timestamp": datetime.now().isoformat()})
    return {"sucesso": True}
