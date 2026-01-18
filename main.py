# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from pydantic import BaseModel
from pymongo import MongoClient
import google.generativeai as genai
from PIL import Image, ImageOps
import io
import os
import urllib.parse
from datetime import datetime
from fpdf import FPDF
import base64

# --- CONFIGURAÇÃO E MODELOS DE IA ---
app = FastAPI(title="TechnoBolt Gym Hub API", version="50.0")

# Motores para Análise Profunda (Relatórios)
MOTORES_ANALISE = ["models/gemini-2.0-flash", "models/gemini-2.5-flash"]

# Motores para Rede Social e Validação Rápida (Alta Cota)
MOTORES_SOCIAL = ["models/gemini-3-flash-preview", "models/gemini-flash-latest"]

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

# --- FUNÇÕES UTILITÁRIAS ---

def rodar_ia(prompt, imagem_bytes, tipo="analise"):
    """
    Sistema de rotação inteligente.
    tipo='analise': usa modelos robustos.
    tipo='social': usa modelos rápidos/alta cota.
    """
    motores = MOTORES_SOCIAL if tipo == "social" else MOTORES_ANALISE
    chaves = [os.environ.get(f"GEMINI_CHAVE_{i}") for i in range(1, 8) if os.environ.get(f"GEMINI_CHAVE_{i}")]
    
    img_blob = {"mime_type": "image/jpeg", "data": imagem_bytes} if imagem_bytes else None
    
    for chave in chaves:
        genai.configure(api_key=chave)
        for modelo in motores:
            try:
                model = genai.GenerativeModel(modelo)
                inputs = [prompt, img_blob] if img_blob else [prompt]
                response = model.generate_content(inputs)
                if response and response.text:
                    return response.text
            except Exception as e:
                continue
    return None

def otimizar_imagem(file_bytes, qualidade=70):
    """Comprime imagem para não estourar o banco de dados"""
    img = Image.open(io.BytesIO(file_bytes))
    img = img.convert("RGB")
    img.thumbnail((800, 800)) # Redimensiona para HD Mobile
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=qualidade)
    return output.getvalue()

def gerar_pdf_bytes(nome, dados):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"TECHNOBOLT GYM - RELATÓRIO: {nome.upper()}", ln=True, align='C')
    pdf.set_font("Arial", "", 12)
    
    # Sanitiza e adiciona conteúdo
    texto_completo = f"""
    DATA: {datetime.now().strftime("%d/%m/%Y")}
    
    1. ANÁLISE CORPORAL:
    {dados.get('r1', 'N/A')}
    
    2. DIETA & NUTROGENÔMICA:
    {dados.get('r2', 'N/A')}
    
    3. SUPLEMENTAÇÃO (NEXO METABÓLICO):
    {dados.get('r3', 'N/A')}
    
    4. TREINO (BIOMECÂNICA):
    {dados.get('r4', 'N/A')}
    """
    # Tratamento básico de encoding para PDF
    texto_safe = texto_completo.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, texto_safe)
    return pdf.output(dest='S').encode('latin-1')

# --- ENDPOINTS: GESTÃO DE ACESSO ---

@app.post("/auth/registro")
def registrar(dados: dict):
    # dados: nome, usuario, senha, genero, peso, altura
    if db.usuarios.find_one({"usuario": dados['usuario']}):
        raise HTTPException(400, "Usuário já existe")
    
    novo = {
        **dados,
        "status": "ativo", # Cadastro direto como ativo para MVP
        "is_admin": False,
        "pontos_social": 0,
        "avaliacoes_restantes": 1, # Crédito inicial
        "historico_dossies": []
    }
    db.usuarios.insert_one(novo)
    return {"sucesso": True}

@app.post("/auth/login")
def login(dados: dict):
    user = db.usuarios.find_one({"usuario": dados['usuario'], "senha": dados['senha']})
    if not user: raise HTTPException(401, "Credenciais inválidas")
    return {
        "sucesso": True,
        "usuario": user['usuario'],
        "nome": user['nome'],
        "is_admin": user.get('is_admin', False),
        "peso": user.get('peso'),
        "altura": user.get('altura'),
        "pontos": user.get('pontos_social', 0)
    }

@app.post("/admin/tornar-admin")
def toggle_admin(dados: dict):
    # Quem está pedindo deve ser admin
    requester = db.usuarios.find_one({"usuario": dados['requester'], "is_admin": True})
    if not requester: raise HTTPException(403, "Sem permissão")
    
    db.usuarios.update_one({"usuario": dados['target']}, {"$set": {"is_admin": True}})
    return {"sucesso": True}

# --- ENDPOINTS: ANÁLISE CORPORAL (IA) ---

@app.post("/analise/executar")
async def executar_analise(
    usuario: str = Form(...),
    nome_completo: str = Form(...),
    peso: float = Form(...),
    altura: int = Form(...),
    objetivo: str = Form(...),
    foto: UploadFile = File(...)
):
    # 1. Atualizar dados cadastrais antes da análise
    db.usuarios.update_one({"usuario": usuario}, {"$set": {"nome": nome_completo, "peso": peso, "altura": altura}})
    
    # 2. Ler e processar imagem
    content = await foto.read()
    imc = peso / ((altura/100)**2)
    
    # 3. Prompt MESTRE (Exatamente como solicitado)
    prompt = f"""VOCÊ É UM CONSELHO TÉCNICO DE ESPECIALISTAS DE ELITE DA TECHNOBOLT GYM.
    ATLETA: {nome_completo} | IMC: {imc:.2f} | META: {objetivo}.

    RESTRITO: SEM SAUDAÇÕES. RESPOSTA DIRETA EM LISTAS.
    [RETORNE UM JSON NO FORMATO]: {{"r1": "...", "r2": "...", "r3": "...", "r4": "..."}}

    1. [AVALIACAO] (Antropometria ISAK 4): Segmentação e Dobras.
    2. [NUTRICAO] (Nutrogenômica): Dieta extensa.
    3. [SUPLEMENTACAO] (Ortomolecular): Nexo Metabólico.
    4. [TREINO] (Biomecânica): 7 dias, alto volume, alternativas técnicas.
    """
    
    raw_res = rodar_ia(prompt, content, tipo="analise")
    
    # Parser simples do retorno (idealmente a IA retorna JSON puro, mas aqui tratamos string)
    # Assumindo que a IA obedeceu o formato JSON ou tags. Para robustez, usaremos regex no Frontend ou aqui.
    # Vamos salvar o RAW e o Frontend trata a exibição nas abas.
    
    dossie = {
        "data": datetime.now().strftime("%d/%m/%Y"),
        "peso_reg": peso,
        "conteudo_bruto": raw_res # O App vai dividir isso nas abas
    }
    
    db.usuarios.update_one({"usuario": usuario}, {"$push": {"historico_dossies": dossie}})
    
    return {"sucesso": True, "resultado": dossie}

# --- ENDPOINTS: TECHNO GYMERS (REDE SOCIAL & GAMIFICATION) ---

@app.post("/social/postar")
async def criar_post(
    usuario: str = Form(...),
    texto: str = Form(...),
    imagem: UploadFile = File(None)
):
    img_b64 = None
    if imagem:
        bytes_img = await imagem.read()
        otimizada = otimizar_imagem(bytes_img)
        img_b64 = base64.b64encode(otimizada).decode('utf-8')
    
    post = {
        "autor": usuario,
        "texto": texto,
        "imagem": img_b64,
        "likes": [],
        "comentarios": [],
        "data": datetime.now().isoformat(),
        "tipo": "feed" # feed ou desafio_completado
    }
    db.posts.insert_one(post)
    return {"sucesso": True}

@app.post("/social/desafio/criar")
def criar_desafio(dados: dict):
    # dados: titulo, descricao, prazo, criador
    desafio = {
        **dados,
        "participantes": [dados['criador']],
        "status": "ativo",
        "ranking": {} # {usuario: pontos}
    }
    db.desafios.insert_one(desafio)
    return {"sucesso": True}

@app.post("/social/desafio/validar-ia")
async def validar_desafio(
    usuario: str = Form(...),
    id_desafio: str = Form(...),
    foto_prova: UploadFile = File(...)
):
    """
    O JUIZ DE IA: Analisa a foto e valida se o desafio foi cumprido.
    """
    from bson.objectid import ObjectId
    desafio = db.desafios.find_one({"_id": ObjectId(id_desafio)})
    if not desafio: raise HTTPException(404, "Desafio não existe")
    
    content = await foto_prova.read()
    
    # Prompt do Juiz
    prompt_juiz = f"""
    VOCÊ É UM JUIZ DE COMPETIÇÃO FITNESS IMPARCIAL.
    O DESAFIO É: "{desafio['titulo']} - {desafio.get('descricao')}".
    
    Analise a imagem fornecida. Ela comprova que o usuário cumpriu este desafio específico?
    Se for uma foto genérica, escura ou que não prova nada, REPROVE.
    
    Retorne APENAS um JSON:
    {{"aprovado": true/false, "motivo": "Explicação curta", "pontos": 10}}
    """
    
    res_ia = rodar_ia(prompt_juiz, content, tipo="social")
    
    # (Aqui entra um parser de JSON do texto da IA, vamos simular o objeto para brevidade)
    # Supondo que res_ia seja a string JSON
    import json
    try:
        resultado = json.loads(res_ia.replace("```json", "").replace("```", ""))
    except:
        # Fallback se a IA não retornar JSON limpo
        resultado = {"aprovado": "true" in res_ia.lower(), "motivo": "Análise automática", "pontos": 10}

    if resultado['aprovado']:
        # Atualiza Ranking
        db.desafios.update_one(
            {"_id": ObjectId(id_desafio)},
            {"$inc": {f"ranking.{usuario}": resultado['pontos']}}
        )
        # Dá pontos globais ao usuário
        db.usuarios.update_one({"usuario": usuario}, {"$inc": {"pontos_social": resultado['pontos']}})
        
        # Posta a prova no feed automaticamente
        otimizada = otimizar_imagem(content)
        img_b64 = base64.b64encode(otimizada).decode('utf-8')
        db.posts.insert_one({
            "autor": usuario,
            "texto": f"✅ Cumpriu o desafio: {desafio['titulo']}! Motivo: {resultado['motivo']}",
            "imagem": img_b64,
            "likes": [],
            "comentarios": [],
            "data": datetime.now().isoformat(),
            "tipo": "prova_desafio"
        })
        
    return resultado

@app.get("/social/feed")
def get_feed():
    # Retorna os últimos 50 posts
    posts = list(db.posts.find().sort("data", -1).limit(50))
    for p in posts: p['_id'] = str(p['_id'])
    return {"feed": posts}

@app.post("/social/chat/enviar")
def enviar_mensagem(dados: dict):
    # dados: remetente, destinatario, mensagem
    msg = {
        **dados,
        "timestamp": datetime.now().isoformat(),
        "lida": False
    }
    db.chat.insert_one(msg)
    return {"sucesso": True}

@app.get("/social/chat/{usuario}")
def ler_chat(usuario: str):
    # Pega mensagens onde o usuário é remetente ou destinatário
    msgs = list(db.chat.find({
        "$or": [{"remetente": usuario}, {"destinatario": usuario}]
    }).sort("timestamp", 1))
    for m in msgs: m['_id'] = str(m['_id'])
    return {"mensagens": msgs}
