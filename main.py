from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import google.generativeai as genai
from PIL import Image, ImageOps
import io
import os
import urllib.parse
from datetime import datetime
import pillow_heif

# --- CONFIGURAÇÃO INICIAL ---
app = FastAPI(title="TechnoBolt API", version="1.0.0")
pillow_heif.register_heif_opener()

# --- CONEXÃO MONGODB (Mantendo a mesma base) ---
def get_database():
    try:
        user = os.environ.get("MONGO_USER", "technobolt")
        password_raw = os.environ.get("MONGO_PASS", "tech@132")
        host = os.environ.get("MONGO_HOST", "cluster0.zbjsvk6.mongodb.net")
        password = urllib.parse.quote_plus(password_raw)
        uri = f"mongodb+srv://{user}:{password}@{host}/?appName=Cluster0"
        client = MongoClient(uri, serverSelectionTimeoutMS=5000, tlsAllowInvalidCertificates=True)
        return client['technoboltgym']
    except Exception as e:
        print(f"Erro Mongo: {e}")
        return None

db = get_database()

# --- MOTOR DE IA (Mantendo lógica de rotação) ---
def realizar_scan_phd(prompt_mestre, img_byte_arr):
    img_blob = {"mime_type": "image/jpeg", "data": img_byte_arr}
    
    # Recupera chaves do ambiente
    chaves = [os.environ.get(f"GEMINI_CHAVE_{i}") for i in range(1, 8)]
    chaves = [k for k in chaves if k]
    
    motores = ["models/gemini-2.0-flash", "models/gemini-1.5-flash"] # Atualizei para modelos mais estáveis de API
    
    for idx, key in enumerate(chaves):
        try:
            genai.configure(api_key=key)
            for m in motores:
                try:
                    model = genai.GenerativeModel(m)
                    response = model.generate_content([prompt_mestre, img_blob])
                    if response and response.text:
                        return response.text
                except: continue
        except: continue
    return None

# --- MODELOS DE DADOS (Para validar entrada) ---
class LoginRequest(BaseModel):
    usuario: str
    senha: str

class CadastroRequest(BaseModel):
    nome: str
    usuario: str
    senha: str
    genero: str

# --- ENDPOINTS (As "Portas" da sua API) ---

@app.get("/")
def home():
    return {"status": "TechnoBolt Brain Online", "version": "42.0-API"}

@app.post("/login")
def login(dados: LoginRequest):
    if db is None: raise HTTPException(status_code=500, detail="Erro banco de dados")
    
    user = db.usuarios.find_one({"usuario": dados.usuario.lower().strip()})
    
    if user and user['senha'] == dados.senha:
        if user['status'] != 'ativo':
            return {"sucesso": False, "mensagem": "Usuário pendente ou inativo."}
            
        return {
            "sucesso": True,
            "usuario": user['usuario'],
            "nome": user['nome'],
            "is_admin": user.get('is_admin', False),
            "creditos": user.get('avaliacoes_restantes', 0),
            "historico_count": len(user.get('historico_dossies', [])),
            "genero": user.get('genero', 'Masculino')
        }
    return {"sucesso": False, "mensagem": "Credenciais inválidas."}

@app.post("/cadastro")
def cadastro(dados: CadastroRequest):
    if db is None: raise HTTPException(status_code=500, detail="Erro banco de dados")
    
    if db.usuarios.find_one({"usuario": dados.usuario.lower().strip()}):
        raise HTTPException(status_code=400, detail="Usuário já existe")
    
    novo_user = {
        "usuario": dados.usuario.lower().strip(),
        "senha": dados.senha,
        "nome": dados.nome,
        "genero": dados.genero,
        "status": "pendente",
        "avaliacoes_restantes": 0,
        "historico_dossies": [],
        "data_renovacao": datetime.now().strftime("%d/%m/%Y")
    }
    db.usuarios.insert_one(novo_user)
    return {"sucesso": True, "mensagem": "Cadastro solicitado com sucesso!"}

@app.post("/analisar")
async def analisar(
    file: UploadFile = File(...),
    usuario: str = Form(...),
    peso: float = Form(...),
    altura: int = Form(...),
    objetivo: str = Form(...),
    restricoes_a: str = Form("Nenhuma"),
    restricoes_m: str = Form("Nenhum"),
    restricoes_f: str = Form("Nenhuma"),
    genero: str = Form(...)
):
    # 1. Verificar Créditos
    user_doc = db.usuarios.find_one({"usuario": usuario})
    if not user_doc: raise HTTPException(404, "Usuário não encontrado")
    if user_doc.get('avaliacoes_restantes', 0) <= 0 and not user_doc.get('is_admin', False):
        raise HTTPException(402, "Créditos insuficientes")

    # 2. Processar Imagem
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((800, 800)) # Otimização para API
    
    # Converter para bytes novamente para enviar pro Gemini
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=85)
    img_bytes = img_byte_arr.getvalue()

    # 3. Calcular IMC e Montar Prompt
    imc = peso / ((altura/100)**2)
    
    prompt = f"""VOCÊ É UM CONSELHO TÉCNICO DE ELITE TECHNOBOLT.
    ATLETA: {user_doc.get('nome')} | GÊNERO: {genero} | IMC: {imc:.2f} | META: {objetivo}.
    RESTRIÇÕES: {restricoes_a}, {restricoes_m}, {restricoes_f}.
    
    [RETORNE APENAS JSON PURO]
    {{"avaliacao": "texto...", "nutricao": "texto...", "suplementacao": "texto...", "treino": "texto..."}}
    
    DETALHES DO PEDIDO:
    1. AVALIACAO: Antropometria ISAK 4 visual (Segmentação e Dobras).
    2. NUTRICAO: Nutrogenômica e Flexibilidade Metabólica.
    3. SUPLEMENTACAO: Nexo Metabólico (mTOR).
    4. TREINO: Lista detalhada 7 dias, alto volume, com alternativa técnica.
    """

    # 4. Chamar IA
    resultado_texto = realizar_scan_phd(prompt, img_bytes)
    
    if not resultado_texto:
        raise HTTPException(503, "IA Indisponível no momento")

    # 5. Salvar no Banco
    # (Aqui faríamos um tratamento de string para JSON melhorado, 
    # mas para manter simples, salvaremos o texto bruto e o App trata a exibição)
    
    novo_historico = {
        "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "peso_reg": peso,
        "conteudo_bruto": resultado_texto # O App vai receber isso e formatar
    }
    
    if not user_doc.get('is_admin', False):
        db.usuarios.update_one({"usuario": usuario}, {"$push": {"historico_dossies": novo_historico}, "$inc": {"avaliacoes_restantes": -1}})
    else:
        db.usuarios.update_one({"usuario": usuario}, {"$push": {"historico_dossies": novo_historico}})

    return {"sucesso": True, "data": novo_historico}
