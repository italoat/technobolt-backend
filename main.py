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

# --- INICIALIZAÇÃO DE SUPORTE HEIC ---
pillow_heif.register_heif_opener()

app = FastAPI(title="TechnoBolt Gym Hub API", version="77.0-Elite-Granular-Refresh")

# --- MOTORES DE IA ---
# [AJUSTE] Lista de modelos atualizada conforme solicitado
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
    # Recupera chaves do ambiente
    chaves = [os.environ.get(f"GEMINI_CHAVE_{i}") for i in range(1, 8) if os.environ.get(f"GEMINI_CHAVE_{i}")]
    img_blob = {"mime_type": "image/jpeg", "data": imagem_bytes} if imagem_bytes else None
    
    # Embaralha para balanceamento de carga
    random.shuffle(chaves)
    
    for chave in chaves:
        genai.configure(api_key=chave)
        for modelo in MOTORES_TECHNOBOLT:
            try:
                model = genai.GenerativeModel(modelo)
                # Configuração para forçar JSON e AUMENTAR O LIMITE DE TOKENS DE RESPOSTA
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
    """Limpa blocos de código markdown e converte para dict"""
    try:
        texto_limpo = texto_ia.replace("```json", "").replace("```", "").strip()
        return json.loads(texto_limpo)
    except Exception as e:
        print(f"Erro ao parsear JSON da IA: {e}")
        # Retorna estrutura vazia para não quebrar o app
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

# --- PDF GENERATOR PREMIUM (DARK MODE) ---

def sanitizar_texto(texto):
    """Remove emojis e caracteres incompatíveis com Latin-1 do PDF"""
    if not texto: return ""
    if not isinstance(texto, str): texto = str(texto)
    texto = texto.replace("🚀", ">>").replace("✅", "[OK]").replace("⚠️", "[!]")
    texto = texto.replace("💊", "").replace("🥗", "").replace("🏋️", "").replace("📊", "")
    texto = texto.replace("**", "").replace("###", "").replace("##", "")
    # Substituições comuns de aspas e travessões
    texto = texto.replace("–", "-").replace("“", '"').replace("”", '"')
    return texto.encode('latin-1', 'replace').decode('latin-1')

class ModernPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        # Cores da Marca (Dark Mode)
        self.col_fundo = (20, 20, 25)      # Preto Suave (Background)
        self.col_card = (35, 35, 40)       # Cinza Card
        self.col_azul = (59, 130, 246)     # Azul TechnoBolt
        self.col_texto = (230, 230, 230)   # Branco Gelo
        self.col_destaque = (0, 255, 200)  # Ciano Neon
        self.col_verde = (16, 185, 129)    # Verde Sucesso

    def header(self):
        # Fundo total da página
        self.set_fill_color(*self.col_fundo)
        self.rect(0, 0, 210, 297, 'F')
        
        # Barra superior
        self.set_fill_color(10, 10, 15)
        self.rect(0, 0, 210, 35, 'F')
        
        # Logo (Texto Estilizado)
        self.set_xy(10, 10)
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(*self.col_azul)
        self.cell(60, 10, "TECHNOBOLT", 0, 0)
        self.set_text_color(255, 255, 255)
        self.cell(40, 10, "GYM HUB", 0, 1)
        
        # Subtitulo
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 150, 150)
        self.set_xy(10, 20)
        self.cell(0, 5, "RELATORIO DE ALTA PERFORMANCE | PHP PROTOCOL", 0, 1)
        
        # Linha decorativa
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
        
        # Linha sutil abaixo do título
        self.set_draw_color(50, 50, 60)
        self.line(10, self.get_y(), 100, self.get_y())
        self.ln(5)

    def draw_card_text(self, label, content):
        """Desenha um bloco de texto com fundo diferenciado"""
        self.set_fill_color(*self.col_card)
        self.set_text_color(*self.col_texto)
        self.set_font("Helvetica", "", 10)
        
        # Título do bloco
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.col_azul)
        self.multi_cell(0, 6, sanitizar_texto(label), fill=True)
        
        # Conteúdo
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.col_texto)
        texto = sanitizar_texto(str(content))
        self.multi_cell(0, 6, texto, fill=True)
        self.ln(2)

    def draw_table_row(self, col1, col2, col3=None):
        """Linha de tabela personalizada"""
        self.set_fill_color(*self.col_card)
        self.set_text_color(*self.col_texto)
        self.set_font("Helvetica", "", 9)
        self.set_draw_color(*self.col_fundo) 
        self.set_line_width(0.3)
        
        h = 8 
        
        # Coluna 1
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*self.col_destaque)
        self.cell(40, h, sanitizar_texto(col1), 1, 0, 'L', True)
        
        # Coluna 2
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
    
    # [PROMPT MESTRE]
    prompt_mestre = f"""
    VOCÊ É A IA DE ELITE DA TECHNOBOLT. SUA MISSÃO É CRIAR UM PROTOCOLO DE ALTA PERFORMANCE.
    
    DADOS DO ATLETA:
    - Nome: {nome_completo} ({genero})
    - IMC: {imc:.2f}
    - Objetivo: {objetivo} (FOCO TOTAL EM RESULTADO MÁXIMO)
    
    RESTRIÇÕES E OBSERVAÇÕES (OBRIGATÓRIO RESPEITAR):
    - Alimentares: {r_a}
    - Físicas: {r_f}
    - Medicamentos: {r_m}
    - Observações Extras: {info}

    ===================================================================================
    REGRAS CRÍTICAS DE GERAÇÃO (LEIA COM ATENÇÃO):
    1. NÃO SEJA PREGUIÇOSO. Você DEVE gerar o plano COMPLETO para os 7 DIAS DA SEMANA (Segunda a Domingo).
    2. DIETA: Gere cardápios DIFERENTES ou CICLOS para TODOS OS 7 DIAS. Nada de "Repetir dia anterior".
    3. TREINO: O usuário pediu para MAXIMIZAR o resultado.
       - Crie treinos de ALTO VOLUME.
       - Mínimo de 8 a 12 EXERCÍCIOS por sessão de treino.
       - Cubra TODOS OS GRUPOS MUSCULARES divididos durante a semana (ABC, ABCD, ou ABCDE).
       - Se for dia de descanso, especifique cardio leve ou recuperação ativa.
    4. RESPEITE LESÕES: Se houver lesão citada, adapte o treino (ex: lesão no joelho = sem impacto).
    ===================================================================================
    
    RETORNE APENAS JSON VÁLIDO (SEM MARKDOWN) NESTE FORMATO EXATO:

    {{
      "avaliacao": {{
        "segmentacao": {{
          "tronco": "Análise detalhada...",
          "superior": "Análise detalhada...",
          "inferior": "Análise detalhada..."
        }},
        "dobras": {{
          "abdominal": "Estimativa...",
          "suprailiaca": "Estimativa...",
          "peitoral": "Estimativa..."
        }},
        "analise_postural": "Texto sobre postura...",
        "simetria": "Texto sobre simetria...",
        "insight": "Insight principal."
      }},
      "dieta": [
        {{
            "dia": "Segunda-feira",
            "foco_nutricional": "Ex: High Carb",
            "refeicoes": [
                {{ "horario": "08:00", "nome": "Café", "alimentos": "..." }},
                {{ "horario": "12:00", "nome": "Almoço", "alimentos": "..." }},
                {{ "horario": "15:00", "nome": "Lanche", "alimentos": "..." }},
                {{ "horario": "18:00", "nome": "Pré-treino", "alimentos": "..." }},
                {{ "horario": "21:00", "nome": "Jantar", "alimentos": "..." }}
            ],
            "macros_totais": "P: 200g | C: 300g..."
        }},
        {{ "dia": "Terça-feira", "foco_nutricional": "...", "refeicoes": [...], "macros_totais": "..." }},
        {{ "dia": "Quarta-feira", "foco_nutricional": "...", "refeicoes": [...], "macros_totais": "..." }},
        {{ "dia": "Quinta-feira", "foco_nutricional": "...", "refeicoes": [...], "macros_totais": "..." }},
        {{ "dia": "Sexta-feira", "foco_nutricional": "...", "refeicoes": [...], "macros_totais": "..." }},
        {{ "dia": "Sábado", "foco_nutricional": "...", "refeicoes": [...], "macros_totais": "..." }},
        {{ "dia": "Domingo", "foco_nutricional": "...", "refeicoes": [...], "macros_totais": "..." }}
      ],
      "dieta_insight": "Estratégia nutricional detalhada.",
      "suplementacao": [
        {{ "nome": "Creatina", "dose": "5g", "horario": "Manhã", "motivo": "Força" }},
        {{ "nome": "Whey", "dose": "30g", "horario": "Pós-treino", "motivo": "Recuperação" }}
      ],
      "suplementacao_insight": "Estratégia de suplementação.",
      "treino": [
        {{
          "dia": "Segunda-feira",
          "foco": "Costas e Bíceps (Volume Alto)",
          "exercicios": [
            {{ "nome": "Puxada Alta", "series_reps": "4x12" }},
            {{ "nome": "Remada Curvada", "series_reps": "4x10" }},
            {{ "nome": "Remada Baixa", "series_reps": "3x12" }},
            {{ "nome": "Pulldown", "series_reps": "3x15" }},
            {{ "nome": "Crucifixo Inverso", "series_reps": "4x12" }},
            {{ "nome": "Rosca Direta", "series_reps": "4x10" }},
            {{ "nome": "Rosca Martelo", "series_reps": "3x12" }},
            {{ "nome": "Rosca Scott", "series_reps": "3x15" }}
          ],
          "treino_alternativo": "Opção com halteres",
          "justificativa": "Foco em densidade dorsal."
        }},
        {{ "dia": "Terça-feira", "foco": "...", "exercicios": [LISTA LONGA...], "..." : "..." }},
        {{ "dia": "Quarta-feira", "foco": "...", "exercicios": [LISTA LONGA...], "..." : "..." }},
        {{ "dia": "Quinta-feira", "foco": "...", "exercicios": [LISTA LONGA...], "..." : "..." }},
        {{ "dia": "Sexta-feira", "foco": "...", "exercicios": [LISTA LONGA...], "..." : "..." }},
        {{ "dia": "Sábado", "foco": "...", "exercicios": [LISTA LONGA...], "..." : "..." }},
        {{ "dia": "Domingo", "foco": "...", "exercicios": [LISTA LONGA...], "..." : "..." }}
      ],
      "treino_insight": "Explicação da periodização."
    }}
    """
    
    resultado_raw = rodar_ia(prompt_mestre, img_otimizada)
    
    if not resultado_raw:
        raise HTTPException(503, "IA Indisponível.")

    # Parseia o JSON
    conteudo_json = limpar_e_parsear_json(resultado_raw)

    dossie = {
        "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "peso_reg": peso,
        "conteudo_bruto": {
            # JSON puro para o novo Flutter
            "json_full": conteudo_json,
            # Fallback para versões antigas
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

# --- ENDPOINT ATUALIZADO: REGENERAR SEÇÃO OU DIA ESPECÍFICO COM CUSTO ---
@app.post("/analise/regenerar-secao")
def regenerar_secao(dados: dict):
    # dados esperados: 
    # {
    #   "usuario": "...", 
    #   "secao": "dieta" | "treino" | "suplementacao" | "avaliacao",
    #   "dia": "Quinta-feira" (Opcional - Se enviado, regenera apenas este dia)
    # }
    usuario = dados.get("usuario")
    secao = dados.get("secao")
    dia_alvo = dados.get("dia") # Parâmetro opcional para granularidade
    
    if not usuario or secao not in ["dieta", "treino", "suplementacao", "avaliacao"]:
        return {"sucesso": False, "mensagem": "Seção inválida ou usuário não informado."}

    user_data = db.usuarios.find_one({"usuario": usuario})
    if not user_data:
        return {"sucesso": False, "mensagem": "Usuário não encontrado."}
    
    # [AJUSTE] Verificação de Créditos
    creditos = user_data.get('avaliacoes_restantes', 0)
    is_admin = user_data.get('is_admin', False)

    if creditos <= 0 and not is_admin:
        return {"sucesso": False, "mensagem": "Saldo de créditos insuficiente para regenerar."}

    if not user_data.get('historico_dossies'):
        return {"sucesso": False, "mensagem": "Nenhum histórico encontrado para basear a regeneração."}

    # Pega o último dossiê para contexto
    ultimo_dossie = user_data['historico_dossies'][-1]
    
    # Dados de contexto
    r_a = user_data.get('restricoes_alim', 'Nenhuma')
    r_f = user_data.get('restricoes_fis', 'Nenhuma')
    obs = user_data.get('info_add', 'Nenhuma')
    nome = user_data.get('nome', 'Atleta')
    
    # --- LÓGICA DE PROMPT (DIA ESPECÍFICO VS SEÇÃO COMPLETA) ---
    if dia_alvo and secao in ["dieta", "treino"]:
        # MODO: REFRESH DE DIA ÚNICO
        prompt_regeneracao = f"""
        ATENÇÃO: Você é um especialista da TechnoBolt.
        TAREFA: Reescrever APENAS o dia '{dia_alvo}' da seção '{secao.upper()}' para o atleta {nome}.
        
        O usuário quer mudar especificamente a rotina deste dia. Mantenha a intensidade alta.
        
        CONTEXTO:
        - Restrições: {r_f} (Físicas), {r_a} (Alimentares)
        - Obs: {obs}
        
        RETORNE APENAS UM JSON COM O OBJETO DESTE DIA.
        Exemplo para Dieta: {{ "dia": "{dia_alvo}", "foco_nutricional": "...", "refeicoes": [...], "macros_totais": "..." }}
        Exemplo para Treino: {{ "dia": "{dia_alvo}", "foco": "...", "exercicios": [...], "treino_alternativo": "...", "justificativa": "..." }}
        """
    else:
        # MODO: REFRESH DE SEÇÃO COMPLETA (Mantido lógica anterior)
        prompt_regeneracao = f"""
        ATENÇÃO: Você é um especialista da TechnoBolt.
        TAREFA: Reescrever COMPLETAMENTE a seção de '{secao.upper()}' para o atleta {nome}.
        O usuário pediu um "REFRESH" nesta seção completa.
        
        CONTEXTO ATUALIZADO:
        - Restrições Físicas: {r_f}
        - Restrições Alimentares: {r_a}
        - Observações: {obs}
        
        REGRAS:
        1. GERE PARA OS 7 DIAS (Segunda a Domingo) se for Treino/Dieta.
        2. Mantenha volume ALTO (8-12 exercícios) para treino.
        
        RETORNE APENAS UM JSON VÁLIDO com a chave correspondente à seção. Exemplo: {{ "{secao}": [ ... ] }}
        """

    # Roda a IA
    resultado_texto = rodar_ia(prompt_regeneracao)
    
    if not resultado_texto:
        return {"sucesso": False, "mensagem": "Erro na IA ao regenerar."}

    novo_dado_ia = limpar_e_parsear_json(resultado_texto)
    
    # --- LÓGICA DE ATUALIZAÇÃO NO BANCO ---
    updates = {}
    
    if dia_alvo and secao in ["dieta", "treino"]:
        # Atualização de Dia Específico (Mais complexo: precisa achar o índice no array)
        lista_atual = ultimo_dossie.get('conteudo_bruto', {}).get('json_full', {}).get(secao, [])
        
        # Encontra o índice do dia alvo (busca case-insensitive parcial)
        idx_alvo = -1
        for i, item in enumerate(lista_atual):
            if dia_alvo.lower() in item.get('dia', '').lower():
                idx_alvo = i
                break
        
        # O objeto retornado pela IA para um dia único geralmente não vem encapsulado na chave da seção
        # Mas as vezes a IA coloca. Vamos normalizar.
        objeto_dia = novo_dado_ia.get(secao, novo_dado_ia) # Tenta pegar dentro da chave, se não, é o próprio dict
        if isinstance(objeto_dia, list): objeto_dia = objeto_dia[0] # Se veio lista de 1 item

        if idx_alvo != -1:
            # Substitui no índice específico
            updates[f"historico_dossies.-1.conteudo_bruto.json_full.{secao}.{idx_alvo}"] = objeto_dia
        else:
            # Se não achou o dia (estranho), adiciona no final ou retorna erro. Vamos tentar adicionar.
            # (Na prática, é melhor retornar erro se o dia não existe no array original, mas vamos ser permissivos)
            # updates -> push (complexo com $set). Vamos simplificar: se não achou, ignora.
            pass
            
    else:
        # Atualização de Seção Completa
        caminho_update = f"historico_dossies.-1.conteudo_bruto.json_full.{secao}"
        caminho_insight = f"historico_dossies.-1.conteudo_bruto.json_full.{secao}_insight"
        
        if secao in novo_dado_ia:
            updates[caminho_update] = novo_dado_ia[secao]
            
        if secao != "avaliacao":
             key_insight = f"{secao}_insight"
             if key_insight in novo_dado_ia:
                 updates[caminho_insight] = novo_dado_ia[key_insight]

    # Aplica as atualizações e cobra o crédito
    if updates:
        # Prepara o comando de update ($set para dados, $inc para créditos)
        mongo_cmd = {"$set": updates}
        if not is_admin:
            mongo_cmd["$inc"] = {"avaliacoes_restantes": -1}
            
        db.usuarios.update_one({"usuario": usuario}, mongo_cmd)
        
        # Retorna o dossiê completo atualizado e o novo saldo
        user_atualizado = db.usuarios.find_one({"usuario": usuario})
        return {
            "sucesso": True, 
            "resultado": user_atualizado['historico_dossies'][-1],
            "novo_saldo": user_atualizado.get('avaliacoes_restantes', 0)
        }
    
    return {"sucesso": False, "mensagem": "Falha ao processar estrutura da resposta da IA."}

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
        
        if not post_id or not usuario:
            return {"sucesso": False, "mensagem": "Dados incompletos"}

        try:
            oid = ObjectId(post_id)
        except Exception:
            return {"sucesso": False, "mensagem": "ID de post inválido"}

        result = db.posts.delete_one({"_id": oid, "autor": usuario})

        if result.deleted_count > 0:
            return {"sucesso": True, "mensagem": "Post deletado"}
        else:
            return {"sucesso": False, "mensagem": "Post não encontrado ou sem permissão."}
            
    except Exception as e:
        print(f"Erro delete: {e}")
        return {"sucesso": False, "mensagem": f"Erro interno: {str(e)}"}
        
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

        db.posts.update_one(
            {"_id": ObjectId(post_id)},
            {"$push": {"comentarios": {"autor": usuario, "texto": texto, "data": datetime.now().isoformat()}}}
        )
        return {"sucesso": True}
    except Exception as e:
        raise HTTPException(500, f"Erro ao postar comentário: {str(e)}")

@app.post("/social/desafio/criar")
def criar_desafio(dados: dict):
    prompt_validacao = f"Analise se este desafio é relacionado a saúde/fitness: '{dados['titulo']} - {dados.get('descricao')}'. Responda APENAS 'SIM' ou 'NAO'."
    res = rodar_ia(prompt_validacao)
    
    if not res or "SIM" not in res.upper():
        return {"sucesso": False, "mensagem": "A IA detectou que este desafio não é focado em saúde."}

    novo_desafio = {
        **dados, 
        "criador": dados['usuario'], 
        "participantes": [dados['usuario']], 
        "ranking": {dados['usuario']: 0}, 
        "status": "ativo"
    }
    db.desafios.insert_one(novo_desafio)
    return {"sucesso": True}

@app.get("/social/desafios")
def listar_desafios_disponiveis(usuario: str):
    desafios = list(db.desafios.find({"participantes": {"$ne": usuario}}).sort("_id", -1))
    for d in desafios: 
        d['_id'] = str(d['_id'])
    return {"sucesso": True, "desafios": desafios}

@app.post("/social/desafio/participar")
def participar_desafio(dados: dict):
    usuario = dados.get("usuario")
    id_desafio = dados.get("id_desafio")
    
    db.desafios.update_one(
        {"_id": ObjectId(id_desafio)},
        {
            "$addToSet": {"participantes": usuario},
            "$set": {f"ranking.{usuario}": 0}
        }
    )
    return {"sucesso": True}

@app.get("/social/meus-desafios")
def listar_meus_desafios(usuario: str):
    desafios = list(db.desafios.find({"participantes": usuario}))
    for d in desafios:
        d['_id'] = str(d['_id'])
        if 'ranking' not in d: d['ranking'] = {usuario: 0}
        
        campo_progresso = f"progresso_{usuario}"
        d['dias_concluidos_atleta'] = d.get(campo_progresso, [])
    
    return {"sucesso": True, "meus_desafios": desafios}

@app.post("/social/desafio/validar-ia")
async def validar_desafio(usuario: str = Form(...), id_desafio: str = Form(...), foto_prova: UploadFile = File(...)):
    content = await foto_prova.read()
    img = otimizar_imagem(content)
    
    prompt = "Aja como juiz fitness. Na imagem, o usuario esta treinando ou comendo saudavel? Responda 'SIM' ou 'NAO'. Se NAO, curto motivo."
    res = rodar_ia(prompt, img)
    
    aprovado = res and "SIM" in res.upper()
    pontos = 10 if aprovado else 0
    motivo = res if not aprovado else "Desafio validado com sucesso!"

    if aprovado:
        campo_progresso = f"progresso_{usuario}"
        db.desafios.update_one(
            {"_id": ObjectId(id_desafio)},
            {
                "$inc": {f"ranking.{usuario}": pontos},
                "$addToSet": {campo_progresso: datetime.now().day} 
            }
        )
        
        img_b64 = base64.b64encode(img).decode('utf-8')
        db.posts.insert_one({
            "autor": usuario,
            "legenda": f"🔥 Validou o dia no desafio! (+{pontos} pts)",
            "imagem": img_b64,
            "data": datetime.now().isoformat(),
            "tipo": "prova_desafio",
            "likes": [],
            "comentarios": [{"autor": "TechnoBolt 🤖", "texto": "Excelente forma! Continue assim."}]
        })

    return {"sucesso": True, "aprovado": aprovado, "pontos": pontos, "motivo": motivo}

# --- ENDPOINTS: ADMIN ---

@app.get("/setup/criar-admin")
def criar_admin_inicial():
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
        "peso": 80.0, "altura": 180, "genero": "Masculino"
    })
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

# --- ENDPOINT: BAIXAR PDF (VERSÃO FINAL COM DARK MODE E JSON) ---

@app.get("/analise/baixar-pdf/{usuario}")
def baixar_pdf_completo(usuario: str):
    try:
        user = db.usuarios.find_one({"usuario": usuario})
        if not user or not user.get('historico_dossies'):
            raise HTTPException(404, "Nenhum relatório encontrado.")

        dossie = user['historico_dossies'][-1]
        raw = dossie.get('conteudo_bruto', {})
        json_data = raw.get('json_full', {}) if isinstance(raw.get('json_full'), dict) else {}

        pdf = ModernPDF()
        pdf.add_page()

        # --- CAPA DO RELATÓRIO ---
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, sanitizar_texto(f"ATLETA: {user.get('nome', 'N/A').upper()}"), ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(180, 180, 180)
        pdf.cell(0, 5, f"DATA DA ANALISE: {dossie.get('data', 'N/A')}", ln=True)
        pdf.cell(0, 5, f"OBJETIVO: {sanitizar_texto(json_data.get('avaliacao', {}).get('insight', 'Alta Performance')[:50])}...", ln=True)
        pdf.ln(10)

        # --- 1. AVALIAÇÃO (EXPANDIDA) ---
        pdf.draw_section_title("1. ANALISE CORPORAL COMPLETA", icon="O")
        av = json_data.get('avaliacao', {})
        seg = av.get('segmentacao', {})
        dob = av.get('dobras', {})
        
        pdf.draw_card_text("Segmentacao Muscular:", 
                           f"- Tronco: {seg.get('tronco','')}\n- Superior: {seg.get('superior','')}\n- Inferior: {seg.get('inferior','')}")
        pdf.ln(2)
        
        # Nova Linha para Postura e Simetria
        pdf.set_fill_color(35, 35, 40)
        pdf.set_text_color(0, 255, 200) # Ciano
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(90, 8, "Analise Postural", 0, 0, 'L', True)
        pdf.cell(5, 8, "", 0, 0) # Espaço
        pdf.cell(90, 8, "Simetria", 0, 1, 'L', True)
        
        pdf.set_text_color(230, 230, 230)
        pdf.set_font("Helvetica", "", 9)
        postura = sanitizar_texto(av.get('analise_postural', 'N/A')[:200])
        simetria = sanitizar_texto(av.get('simetria', 'N/A')[:200])
        
        # Bloco Postura
        x_atual = pdf.get_x()
        y_atual = pdf.get_y()
        pdf.multi_cell(90, 5, postura, fill=True)
        y_fim_1 = pdf.get_y()
        
        # Bloco Simetria
        pdf.set_xy(x_atual + 95, y_atual)
        pdf.multi_cell(90, 5, simetria, fill=True)
        y_fim_2 = pdf.get_y()
        
        pdf.set_y(max(y_fim_1, y_fim_2) + 3)

        pdf.draw_card_text("Estimativa de Dobras:", 
                           f"- Abd: {dob.get('abdominal','')}\n- Supra: {dob.get('suprailiaca','')}\n- Peit: {dob.get('peitoral','')}")
        
        pdf.ln(2)
        pdf.set_text_color(0, 255, 200)
        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(0, 6, sanitizar_texto(f">> INSIGHT TECNICO: {av.get('insight', '')}"))

        # --- 2. DIETA (DETALHADA EM CARDS) ---
        pdf.add_page()
        pdf.draw_section_title("2. PROTOCOLO NUTRICIONAL", icon="U")
        
        dieta = json_data.get('dieta', [])
        if isinstance(dieta, list):
            for dia in dieta:
                pdf.ln(3)
                # Cabeçalho do Dia
                pdf.set_fill_color(50, 50, 60)
                pdf.set_text_color(16, 185, 129) # Verde
                pdf.set_font("Helvetica", "B", 10)
                
                header_text = sanitizar_texto(f"{dia.get('dia', '').upper()} | FOCO: {dia.get('foco_nutricional', '').upper()}")
                pdf.cell(0, 8, header_text, 0, 1, 'L', True)
                
                # Lista de Refeições
                refeicoes = dia.get('refeicoes', [])
                pdf.set_fill_color(35, 35, 40)
                pdf.set_text_color(230, 230, 230)
                
                if isinstance(refeicoes, list):
                    for ref in refeicoes:
                        hora = sanitizar_texto(ref.get('horario', ''))
                        nome = sanitizar_texto(ref.get('nome', ''))
                        alim = sanitizar_texto(ref.get('alimentos', ''))
                        
                        pdf.set_font("Helvetica", "B", 9)
                        pdf.cell(30, 6, f"{hora} - {nome}:", 0, 0, 'L', True)
                        pdf.set_font("Helvetica", "", 9)
                        pdf.multi_cell(0, 6, alim, fill=True)
                
                # Footer de Macros
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(150, 150, 150)
                pdf.cell(0, 6, sanitizar_texto(f"Macros: {dia.get('macros_totais', '')}"), 0, 1, 'R', True)
                
                # Linha divisória
                pdf.set_draw_color(30, 30, 30)
                pdf.line(10, pdf.get_y()+2, 200, pdf.get_y()+2)
                pdf.ln(2)

        pdf.ln(5)
        pdf.draw_card_text("Estrategia Geral:", json_data.get('dieta_insight', ''))

        # --- 3. TREINO (DETALHADO E BONITO) ---
        pdf.add_page()
        pdf.draw_section_title("3. PLANILHA DE TREINO", icon="X")
        
        treino = json_data.get('treino', [])
        if isinstance(treino, list):
            for item in treino:
                pdf.ln(3)
                # Cabeçalho do Dia
                pdf.set_fill_color(50, 50, 60)
                pdf.set_text_color(0, 255, 200) # Destaque Ciano
                pdf.set_font("Helvetica", "B", 10)
                foco_dia = sanitizar_texto(f"{item.get('dia','').upper()} - FOCO: {item.get('foco','').upper()}")
                pdf.cell(0, 8, foco_dia, 0, 1, 'L', True)

                # Fundo do corpo do card de treino
                pdf.set_fill_color(35, 35, 40)
                pdf.set_text_color(230, 230, 230)
                pdf.set_font("Helvetica", "", 9)

                # Lista de Exercícios
                exercicios = item.get('exercicios', [])
                if exercicios:
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.cell(0, 6, "Exercicios:", 0, 1, 'L', True)
                    pdf.set_font("Helvetica", "", 9)
                    for ex in exercicios:
                        nome_ex = sanitizar_texto(ex.get('nome', ''))
                        series = sanitizar_texto(ex.get('series_reps', ''))
                        linha = f"  > {nome_ex} [{series}]"
                        pdf.cell(0, 5, linha, 0, 1, 'L', True)
                
                # Alternativo e Justificativa
                alt = sanitizar_texto(item.get('treino_alternativo', 'N/A'))
                just = sanitizar_texto(item.get('justificativa', 'N/A'))
                
                pdf.ln(1)
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(150, 150, 150)
                pdf.multi_cell(0, 5, f"Alternativo: {alt}", fill=True)
                pdf.multi_cell(0, 5, f"Motivo: {just}", fill=True)
                
                # Linha separadora entre dias
                pdf.set_draw_color(30, 30, 30)
                pdf.line(10, pdf.get_y()+2, 200, pdf.get_y()+2)
                pdf.ln(2)

        pdf.ln(5)
        pdf.draw_card_text("Analise Biomecanica Geral:", json_data.get('treino_insight', ''))

        # --- 4. SUPLEMENTAÇÃO (CARDS VISUAIS) ---
        pdf.add_page()
        pdf.draw_section_title("4. ARSENAL DE SUPLEMENTOS", icon="+")
        
        suple = json_data.get('suplementacao', [])
        if isinstance(suple, list):
            # Layout de 2 colunas para economizar espaço
            col_width = 90
            spacing = 5
            
            for i in range(0, len(suple), 2):
                # Pega item par e impar (coluna 1 e 2)
                item1 = suple[i]
                item2 = suple[i+1] if i+1 < len(suple) else None
                
                # Altura fixa do card
                card_h = 35
                
                # --- COLUNA 1 ---
                x_start = pdf.get_x()
                y_start = pdf.get_y()
                
                # Card Background
                pdf.set_fill_color(35, 35, 40)
                pdf.rect(x_start, y_start, col_width, card_h, 'F')
                
                # Conteúdo
                pdf.set_xy(x_start + 2, y_start + 2)
                pdf.set_font("Helvetica", "B", 11)
                pdf.set_text_color(255, 165, 0) # Laranja
                pdf.cell(col_width-4, 6, sanitizar_texto(item1.get('nome', '')), 0, 1)
                
                pdf.set_x(x_start + 2)
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(255, 255, 255)
                pdf.cell(col_width-4, 5, sanitizar_texto(f"Dose: {item1.get('dose', '')} | {item1.get('horario', '')}"), 0, 1)
                
                pdf.set_x(x_start + 2)
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(200, 200, 200)
                pdf.multi_cell(col_width-4, 4, sanitizar_texto(item1.get('motivo', '')))
                
                # --- COLUNA 2 (Se existir) ---
                if item2:
                    pdf.set_xy(x_start + col_width + spacing, y_start)
                    
                    # Card Background
                    pdf.set_fill_color(35, 35, 40)
                    pdf.rect(pdf.get_x(), pdf.get_y(), col_width, card_h, 'F')
                    
                    # Conteúdo
                    current_x = pdf.get_x()
                    pdf.set_xy(current_x + 2, y_start + 2)
                    pdf.set_font("Helvetica", "B", 11)
                    pdf.set_text_color(255, 165, 0) 
                    pdf.cell(col_width-4, 6, sanitizar_texto(item2.get('nome', '')), 0, 1)
                    
                    pdf.set_x(current_x + 2)
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.set_text_color(255, 255, 255)
                    pdf.cell(col_width-4, 5, sanitizar_texto(f"Dose: {item2.get('dose', '')} | {item2.get('horario', '')}"), 0, 1)
                    
                    pdf.set_x(current_x + 2)
                    pdf.set_font("Helvetica", "", 8)
                    pdf.set_text_color(200, 200, 200)
                    pdf.multi_cell(col_width-4, 4, sanitizar_texto(item2.get('motivo', '')))

                # Move cursor para próxima linha de cards
                pdf.set_xy(x_start, y_start + card_h + spacing)

        pdf.ln(5)
        pdf.draw_card_text("Insight Ortomolecular:", json_data.get('suplementacao_insight', ''))

        # Output
        pdf_buffer = io.BytesIO()
        pdf_output = pdf.output(dest='S')
        
        if isinstance(pdf_output, str):
            pdf_buffer.write(pdf_output.encode('latin-1'))
        else:
            pdf_buffer.write(pdf_output)
            
        pdf_buffer.seek(0)
        
        headers = {'Content-Disposition': f'attachment; filename="TechnoBolt_Protocolo.pdf"'}
        return StreamingResponse(pdf_buffer, media_type="application/pdf", headers=headers)

    except Exception as e:
        print(f"ERRO CRÍTICO PDF: {e}")
        raise HTTPException(500, f"Erro ao gerar PDF: {str(e)}")

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
