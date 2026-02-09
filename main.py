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

app = FastAPI(title="TechnoBolt Gym Hub API", version="89.4-Elite-Stable-Fix-Types")

# --- CARREGAMENTO DO BANCO DE EXERCÍCIOS (JSON EXTERNO) ---
EXERCISE_KEYS = []
EXERCISE_DB = {}
EXERCISE_LIST_STRING = ""

try:
    with open("exercises.json", "r", encoding="utf-8") as f:
        EXERCISE_DB = json.load(f)
        EXERCISE_KEYS = list(EXERCISE_DB.keys())
        EXERCISE_LIST_STRING = ", ".join([k for k in EXERCISE_KEYS])
        
    print(f"✅ Banco de Exercícios Carregado e Indexado: {len(EXERCISE_DB)} itens.")
except Exception as e:
    print(f"⚠️ AVISO CRÍTICO: Erro ao carregar exercises.json. A API funcionará, mas sem imagens. Erro: {e}")

# --- MOTORES DE IA (LISTA ATUALIZADA) ---
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
        parsed = json.loads(texto_limpo)
        # Aplica a sanitização recursiva de forma obrigatória
        return sanitizar_json_ia(parsed)
    except Exception as e:
        print(f"Erro ao parsear JSON da IA: {e}")
        return {
            "avaliacao": {"insight": "Erro na leitura da IA"},
            "dieta": [],
            "suplementacao": [],
            "treino": []
        }

# Função crítica para evitar o erro 'double is not subtype of String'
def sanitizar_json_ia(data):
    """
    Percorre o JSON recursivamente e converte números (int, float) em strings 
    para garantir que o Flutter (fortemente tipado) não quebre ao tentar
    renderizar um Double em um Text() widget.
    """
    if isinstance(data, dict):
        return {k: sanitizar_json_ia(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitizar_json_ia(v) for v in data]
    elif isinstance(data, (int, float)):
        # Converte número forçadamente para String
        return str(data)
    else:
        return data

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
    # Proteção extra: garante que é string antes de normalizar
    if not isinstance(texto, str): texto = str(texto)
    return "".join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn').lower().strip()

def validar_e_corrigir_exercicios(lista_exercicios):
    if not lista_exercicios or not EXERCISE_DB: return lista_exercicios
    
    base_url = "https://raw.githubusercontent.com/italoat/technobolt-backend/main/assets/exercises"
    
    db_map_norm = {normalizar_texto(k): v for k, v in EXERCISE_DB.items()}
    db_title_map = {normalizar_texto(k): k for k, v in EXERCISE_DB.items()}

    for ex in lista_exercicios:
        nome_ia = ex.get('nome', '')
        nome_ia_norm = normalizar_texto(nome_ia)
        
        pasta_github = None
        nome_final = str(nome_ia) # Força string

        if nome_ia_norm in db_map_norm:
            pasta_github = db_map_norm[nome_ia_norm]
            nome_final = db_title_map[nome_ia_norm].title()
        else:
            matches = difflib.get_close_matches(nome_ia_norm, db_map_norm.keys(), n=1, cutoff=0.6)
            if matches:
                match_key = matches[0]
                pasta_github = db_map_norm[match_key]
                nome_final = db_title_map[match_key].title()
            else:
                melhor_candidato = None
                for key in db_map_norm.keys():
                    if (key in nome_ia_norm and len(key) > 4) or (nome_ia_norm in key and len(nome_ia_norm) > 4): 
                        melhor_candidato = key
                        break
                
                if melhor_candidato:
                    pasta_github = db_map_norm[melhor_candidato]
                    nome_final = db_title_map[melhor_candidato].title()
                else:
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
        user = db.usuarios.find_one({"usuario": username})
        if not user: return ""
        pontos = user.get('pontos', 0)
        if pontos > 1000: return "🥇"
        if pontos > 500: return "🥈"
        if pontos > 100: return "🥉"
        return ""
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
            "pontos": user.get('pontos', 0),
            "foto_perfil": user.get('foto_perfil', None),
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
        "pontos": 0,
        "historico_dossies": [],
        "is_admin": False
    }
    db.usuarios.insert_one(novo_user)
    return {"sucesso": True, "mensagem": "Cadastro realizado"}
    
@app.post("/perfil/atualizar")
def atualizar_perfil(dados: dict):
    update_data = {
        "nome": dados.get('nome'),
        "peso": dados.get('peso'),
        "altura": dados.get('altura'),
        "genero": dados.get('genero'),
        "restricoes_alim": dados.get('restricoes_alim'),
        "restricoes_fis": dados.get('restricoes_fis'),
        "medicamentos": dados.get('medicamentos'),
        "info_add": dados.get('info_add'),
    }
    
    if 'foto_perfil' in dados:
        update_data['foto_perfil'] = dados.get('foto_perfil')

    db.usuarios.update_one(
        {"usuario": dados['usuario']},
        {"$set": update_data}
    )
    return {"sucesso": True}

# --- ENDPOINTS: ANÁLISE (ESTRUTURADA PARA JSON) ---

@app.post("/analise/executar")
async def executar_analise(
    usuario: str = Form(...),
    nome_completo: str = Form(...),
    peso: str = Form(...), # [SENIOR] Alterado para str para tratar vírgula
    altura: int = Form(...),
    objetivo: str = Form(...),
    genero: str = Form("Masculino"),
    observacoes: str = Form(""), 
    foto: UploadFile = File(...)
):
    # Tratamento da String de Peso (Aceitar vírgula e ponto)
    try:
        peso_limpo = peso.replace(',', '.')
        peso_float = float(peso_limpo)
    except ValueError:
        raise HTTPException(400, "Formato de peso inválido. Use números (ex: 80.5 ou 80,5)")

    # Atualiza dados no banco
    db.usuarios.update_one(
        {"usuario": usuario}, 
        {"$set": {
            "nome": nome_completo, 
            "peso": peso_float, 
            "altura": altura, 
            "genero": genero,
            "info_add": observacoes
        }}
    )

    user_data = db.usuarios.find_one({"usuario": usuario})
    r_a = user_data.get('restricoes_alim', 'Nenhuma')
    r_m = user_data.get('medicamentos', 'Nenhum')
    r_f = user_data.get('restricoes_fis', 'Nenhuma')
    info = observacoes 

    content = await foto.read()
    img_otimizada = otimizar_imagem(content, quality=85, size=(800, 800))
    imc = peso_float / ((altura/100)**2)
    
    # [PROMPT MESTRE - ATUALIZADO PARA SUPERÁVIT CALÓRICO]
    prompt_mestre = f"""
    VOCÊ É UM TREINADOR DE ELITE (PhD em Biomecânica e Nutrição). 
    SUA MISSÃO: CRIAR O PROTOCOLO PERFEITO E ÚNICO PARA O ALUNO, MAXIMIZANDO RESULTADOS.
    
    PERFIL: {nome_completo}, {genero}, IMC {imc:.2f}, OBJETIVO: {objetivo}.
    RESTRIÇÕES: Alimentares: "{r_a}", Físicas: "{r_f}", Meds: "{r_m}", Obs: "{info}".

    REGRAS CRÍTICAS DE OUTPUT (SIGA RIGOROSAMENTE):
    1. **TREINO (O MAIS IMPORTANTE)**: 
       - **USE APENAS EXERCÍCIOS DESTA LISTA ABAIXO (COPIE O NOME EXATO):**
       [ {EXERCISE_LIST_STRING} ]
       - Se você escolher um exercício fora dessa lista, o aplicativo quebrará. SEJA ESTRITO.
       - Título do dia: APENAS O NOME DO DIA DA SEMANA (Ex: "Segunda-feira").
       - Coloque o foco muscular no campo 'foco' (Ex: "Peito e Tríceps").
       - **VOLUME ALTO DE TREINO:** Para maximizar resultados, gere entre **8 a 12 exercícios por sessão**.
       - Gere para os 7 dias.
       - OBRIGATÓRIO: campo 'execucao' detalhado sem emojis.

    2. PROIBIDO RETORNAR NULL ou "Sem dados". Se faltar info, ESTIME com base no IMC e Objetivo.
    
    3. **ANÁLISE CORPORAL E BIOMETRIA:**
       - Na seção 'avaliacao', analise a composição corporal atual considerando que o objetivo requer construção muscular.
       - Mencione o planejamento de Bulking/Manutenção no 'insight'.

    4. **DIETA (SUPERÁVIT CALÓRICO):**
       - **CALCULE O GASTO ENERGÉTICO TOTAL (GET)** do aluno.
       - **OBRIGATÓRIO:** APLIQUE UM **SUPERÁVIT CALÓRICO (BULKING LIMPO)** de aproximadamente **+300 a +500 kcal** sobre o GET para garantir o ganho de massa, a não ser que o objetivo seja explicitamente 'Perda de Peso' ou 'Cutting'.
       - Preencha 'macros_totais' refletindo esse superávit (Ex: Proteína alta ~2g/kg, Carbo alto para energia).
       - Crie cardápios completos para 7 dias.

    5. SUPLEMENTAÇÃO: Se o aluno não tiver restrição médica grave, sugira o básico (Creatina, Whey, Multivitamínico). Preencha dose/horário.

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
            "macros_totais": "P: Xg | C: Yg | G: Zg (Superávit Aplicado)"
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
            }},
            ... (Insira de 8 a 12 exercícios aqui)
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

    if 'treino' in conteudo_json and isinstance(conteudo_json['treino'], list):
        for dia in conteudo_json['treino']:
            if 'dia' in dia:
                dia['dia'] = str(dia['dia']).split('-')[0].split(' ')[0].replace(',', '').strip()

            if 'exercicios' in dia and isinstance(dia['exercicios'], list):
                dia['exercicios'] = validar_e_corrigir_exercicios(dia['exercicios'])

    dossie = {
        "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
        # [SENIOR FIX] Garante que peso_reg seja sempre string na saída
        "peso_reg": str(peso_float), 
        "conteudo_bruto": {
            "json_full": conteudo_json,
            "r1": str(conteudo_json.get('avaliacao', {}).get('insight', '')),
            "r2": str(conteudo_json.get('dieta_insight', '')),
            "r3": str(conteudo_json.get('suplementacao_insight', '')),
            "r4": str(conteudo_json.get('treino_insight', ''))
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
        if secao == "treino":
            prompt_regeneracao = f"""
            ATENÇÃO: Treinador de Elite TechnoBolt.
            TAREFA: Reescrever APENAS o dia '{dia_alvo}' da seção 'TREINO' para o atleta {nome}.
            CONTEXTO: Restrições: {r_f}. Obs: {obs}.
            
            REGRA CRÍTICA:
            1. USE **APENAS** EXERCÍCIOS DESTA LISTA: [ {EXERCISE_LIST_STRING} ]
            2. VOLUME ALTO: Gere entre **6 a 9 exercícios**.
            3. Título do dia: APENAS "{dia_alvo}".
            
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
        else: # Dieta
             prompt_regeneracao = f"""
            ATENÇÃO: Nutricionista de Elite TechnoBolt.
            TAREFA: Reescrever a DIETA de '{dia_alvo}' para {nome}.
            CONTEXTO: {r_a}.
            REGRAS: 
            1. NÃO retorne campos nulos. 
            2. MANTENHA O SUPERÁVIT CALÓRICO calculado anteriormente (Bulking), exceto se solicitado o contrário.
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
        - Se for dieta, inclua 'macros_totais' com SUPERÁVIT CALÓRICO para ganho de massa.
        - Se for suplementacao, inclua dose e horario.
        RETORNE JSON: {{ "{secao}": [ ... ] }}
        """

    resultado_texto = rodar_ia(prompt_regeneracao)
    if not resultado_texto: return {"sucesso": False, "mensagem": "Erro IA."}

    novo_dado_ia = limpar_e_parsear_json(resultado_texto)
    
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

    # Lógica de atualização no Banco
    updates = {}
    if dia_alvo and secao in ["dieta", "treino"]:
        lista_atual = ultimo_dossie.get('conteudo_bruto', {}).get('json_full', {}).get(secao, [])
        idx_alvo = -1
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
    
    return {"sucesso": False, "mensagem": "Estrutura inválida."}

@app.get("/historico/{usuario}")
def buscar_historico(usuario: str):
    user = db.usuarios.find_one({"usuario": usuario})
    if not user: return {"sucesso": True, "historico": []}
    
    # Perfil sanitizado para evitar nulos ou tipos errados no front
    perfil = {
        "peso": str(user.get('peso', '')), 
        "altura": str(user.get('altura', '')), 
        "genero": str(user.get('genero', 'Masculino')),
        "restricoes_alim": str(user.get('restricoes_alim', '')),
        "restricoes_fis": str(user.get('restricoes_fis', '')),
        "medicamentos": str(user.get('medicamentos', '')),
        "info_add": str(user.get('info_add', '')),
        "creditos": user.get('avaliacoes_restantes', 0)
    }
    
    return {"sucesso": True, "historico": user.get('historico_dossies', []), "creditos": user.get('avaliacoes_restantes', 0), "perfil": perfil}

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

# --- ENDPOINTS: CONQUISTA & RANKING (GAMIFICATION) ---

@app.get("/social/ranking")
def get_ranking_global():
    users = list(db.usuarios.find(
        {"is_admin": False}, 
        {"nome": 1, "usuario": 1, "pontos": 1, "foto_perfil": 1, "_id": 0}
    ).sort("pontos", -1).limit(50))
    return {"sucesso": True, "ranking": users}

@app.get("/social/checkins")
def get_checkins(usuario: str):
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

@app.post("/social/validar-conquista")
async def validar_conquista(
    usuario: str = Form(...),
    tipo: str = Form(...), 
    foto: UploadFile = File(...)
):
    now = datetime.now()
    today_start = datetime(now.year, now.month, now.day).isoformat()
    
    ja_fez = db.checkins.find_one({
        "usuario": usuario, 
        "data": {"$gte": today_start}
    })
    
    if ja_fez:
        return {"sucesso": False, "mensagem": "Você já treinou hoje! Volte amanhã."}

    content = await foto.read()
    img_otimizada = otimizar_imagem(content, size=(800, 800))
    
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
    
    resultado_ia = rodar_ia(prompt_juiz, img_otimizada)
    
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
