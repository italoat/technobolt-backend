from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pymongo import MongoClient
from bson.objectid import ObjectId
import google.generativeai as genai
from PIL import Image, ImageOps
import io
import os
import re
import urllib.parse
from datetime import datetime
import base64
import random
import pillow_heif

# --- INICIALIZAÇÃO DE SUPORTE HEIC ---
pillow_heif.register_heif_opener()

app = FastAPI(title="TechnoBolt Gym Hub API", version="67.0-Elite")

# --- MOTORES DE IA (SEUS MODELOS ESPECÍFICOS) ---
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

# --- FUNÇÕES UTILITÁRIAS ---

def rodar_ia(prompt, imagem_bytes=None):
    # Recupera todas as chaves disponíveis no ambiente
    chaves = [os.environ.get(f"GEMINI_CHAVE_{i}") for i in range(1, 8) if os.environ.get(f"GEMINI_CHAVE_{i}")]
    img_blob = {"mime_type": "image/jpeg", "data": imagem_bytes} if imagem_bytes else None
    
    # Embaralha as chaves para distribuir a carga
    random.shuffle(chaves)
    
    # Tenta cada chave disponível
    for chave in chaves:
        genai.configure(api_key=chave)
        # Tenta cada modelo na ordem de preferência (3 -> 2.5 -> 2.0 -> Latest)
        for modelo in MOTORES_TECHNOBOLT:
            try:
                model = genai.GenerativeModel(modelo)
                inputs = [prompt, img_blob] if img_blob else [prompt]
                response = model.generate_content(inputs)
                if response and response.text:
                    return response.text
            except Exception:
                continue # Se der erro no modelo, tenta o próximo da lista
    return None

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

def extrair_tags(texto_ia, tag_inicio, tag_fim=None):
    t_i = tag_inicio.replace('[', '\\[').replace(']', '\\]')
    if tag_fim:
        t_f = tag_fim.replace('[', '\\[').replace(']', '\\]')
        padrao = f"{t_i}\\s*(.*?)\\s*(?={t_f}|$)"
    else:
        padrao = f"{t_i}\\s*(.*)"
    
    match = re.search(padrao, texto_ia, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else "Conteúdo não gerado corretamente pela IA."

# --- ENDPOINTS: AUTH & PERFIL ---

@app.post("/auth/login")
def login(dados: dict):
    user = db.usuarios.find_one({"usuario": dados['usuario'], "senha": dados['senha']})
    if not user: raise HTTPException(401, "Credenciais inválidas")
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

# --- ENDPOINTS: ANÁLISE ---

@app.post("/analise/executar")
async def executar_analise(
    usuario: str = Form(...),
    nome_completo: str = Form(...),
    peso: float = Form(...),
    altura: int = Form(...),
    objetivo: str = Form(...),
    genero: str = Form("Masculino"),
    foto: UploadFile = File(...)
):
    user_data = db.usuarios.find_one({"usuario": usuario})
    r_a = user_data.get('restricoes_alim', 'Nenhuma')
    r_m = user_data.get('medicamentos', 'Nenhum')
    r_f = user_data.get('restricoes_fis', 'Nenhuma')
    info = user_data.get('info_add', '')

    # Atualiza dados básicos
    db.usuarios.update_one({"usuario": usuario}, {"$set": {"nome": nome_completo, "peso": peso, "altura": altura, "genero": genero}})
    
    content = await foto.read()
    img_otimizada = otimizar_imagem(content, quality=85, size=(800, 800))
    imc = peso / ((altura/100)**2)
    
    prompt_mestre = f"""VOCÊ É UM CONSELHO TÉCNICO DE ESPECIALISTAS DE ELITE DA TECHNOBOLT GYM.
    ATLETA: {nome_completo} | GÊNERO: {genero} | IMC: {imc:.2f}.
    META: {objetivo}. 
    RESTRIÇÕES: {r_a}, {r_m}, {r_f}.
    OBSERVAÇÕES: {info}.

    RESTRITO: SEM SAUDAÇÕES OU TÍTULOS. RESPOSTA DIRETA EM LISTAS.
    
    [AVALIACAO]
    Aja como Especialista em Cineantropometria.
    1. SEGMENTAÇÃO CORPORAL (PONTOS DE ATENÇÃO):
    - Tronco e Cabeça.
    - Membros Superiores.
    - Membros Inferiores.
    2. ESTIMATIVA DE DOBRAS CUTÂNEAS.
    AO FINAL: 🚀 TECHNOBOLT INSIGHT: 3 recomendações técnicas.

    [NUTRICAO]
    Especialista em Nutrogenômica. Plano dietético extenso.
    AO FINAL: 🚀 TECHNOBOLT INSIGHT: 3 recomendações.

    [SUPLEMENTACAO]
    Especialista Ortomolecular. 3-10 itens via Nexo Metabólico.
    AO FINAL: 🚀 TECHNOBOLT INSIGHT: 3 recomendações.

    [TREINO]
    Especialista em Neuromecânica. O TREINO DEVE RESOLVER AS FALHAS DA FOTO.
    7 DIAS EM LISTA DETALHADA.
    AO FINAL: 🚀 TECHNOBOLT INSIGHT: 3 recomendações.
    """
    
    resultado_texto = rodar_ia(prompt_mestre, img_otimizada)
    
    if not resultado_texto:
        raise HTTPException(503, "IA Indisponível.")

    r1 = extrair_tags(resultado_texto, "[AVALIACAO]", "[NUTRICAO]")
    r2 = extrair_tags(resultado_texto, "[NUTRICAO]", "[SUPLEMENTACAO]")
    r3 = extrair_tags(resultado_texto, "[SUPLEMENTACAO]", "[TREINO]")
    r4 = extrair_tags(resultado_texto, "[TREINO]")

    dossie = {
        "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "peso_reg": peso,
        "conteudo_bruto": {"r1": r1, "r2": r2, "r3": r3, "r4": r4, "full_text": resultado_texto}
    }
    
    if not user_data.get('is_admin', False):
        db.usuarios.update_one({"usuario": usuario}, {"$push": {"historico_dossies": dossie}, "$inc": {"avaliacoes_restantes": -1}})
    else:
        db.usuarios.update_one({"usuario": usuario}, {"$push": {"historico_dossies": dossie}})
    
    return {"sucesso": True, "resultado": dossie}

@app.get("/historico/{usuario}")
def buscar_historico(usuario: str):
    user = db.usuarios.find_one({"usuario": usuario})
    if not user: return {"sucesso": True, "historico": []}
    return {"sucesso": True, "historico": user.get('historico_dossies', [])}

# --- ENDPOINTS: SOCIAL E DESAFIOS ---

@app.get("/social/feed")
def get_feed():
    posts = list(db.posts.find().sort("data", -1).limit(50))
    for p in posts: p['_id'] = str(p['_id'])
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

@app.post("/social/desafio/criar")
def criar_desafio(dados: dict):
    prompt_validacao = f"Analise se este desafio é relacionado a saúde/fitness: '{dados['titulo']} - {dados.get('descricao')}'. Responda APENAS 'SIM' ou 'NAO'."
    res = rodar_ia(prompt_validacao)
    
    if not res or "SIM" not in res.upper():
        return {"sucesso": False, "mensagem": "A IA detectou que este desafio não é focado em saúde."}

    novo_desafio = {**dados, "criador": dados['usuario'], "participantes": [dados['usuario']], "ranking": {dados['usuario']: 0}, "status": "ativo"}
    db.desafios.insert_one(novo_desafio)
    return {"sucesso": True}

@app.get("/social/desafios")
def listar_desafios():
    desafios = list(db.desafios.find().sort("_id", -1))
    for d in desafios: d['_id'] = str(d['_id'])
    return {"sucesso": True, "desafios": desafios}

@app.post("/social/desafio/validar-ia")
async def validar_desafio(usuario: str = Form(...), id_desafio: str = Form(...), foto_prova: UploadFile = File(...)):
    content = await foto_prova.read()
    img = otimizar_imagem(content)
    
    prompt = "Você é um Juiz de Provas de Fitness. Analise a imagem. O usuário completou um exercício físico ou desafio de saúde? Responda APENAS 'SIM' ou 'NAO'. Se NAO, explique em 5 palavras."
    res = rodar_ia(prompt, img)
    
    aprovado = res and "SIM" in res.upper()
    pontos = 10 if aprovado else 0
    motivo = res if not aprovado else "Prova aceita."

    if aprovado:
        # Postar no feed automaticamente
        img_b64 = base64.b64encode(img).decode('utf-8')
        db.posts.insert_one({
            "autor": usuario,
            "legenda": f"🏆 Cumpriu o desafio! (+{pontos}pts)",
            "imagem": img_b64,
            "data": datetime.now().isoformat(),
            "tipo": "prova_desafio",
            "likes": [], "comentarios": []
        })

    return {"sucesso": True, "aprovado": aprovado, "pontos": pontos, "motivo": motivo}

# --- ENDPOINTS: ADMIN & SETUP ---

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
