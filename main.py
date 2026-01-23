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

app = FastAPI(title="TechnoBolt Gym Hub API", version="71.0-Elite-JSON")

# --- MOTORES DE IA ---
MOTORES_TECHNOBOLT = [
    "models/gemini-2.0-flash",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro"
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
                # Configuração para forçar JSON (quando suportado pelo modelo) ou texto limpo
                generation_config = {"response_mime_type": "application/json"} if "json" in prompt.lower() else None
                
                inputs = [prompt, img_blob] if img_blob else [prompt]
                
                if generation_config:
                    response = model.generate_content(inputs, generation_config=generation_config)
                else:
                    response = model.generate_content(inputs)

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
            "avaliacao": {"insight": "Erro na IA"},
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

# --- PDF GENERATOR (ATUALIZADO PARA JSON) ---

def sanitizar_texto(texto):
    """Remove emojis e caracteres incompatíveis com Latin-1 do PDF"""
    if not texto: return ""
    if not isinstance(texto, str): texto = str(texto)
    texto = texto.replace("🚀", ">>").replace("✅", "[OK]").replace("⚠️", "[!]")
    texto = texto.replace("💊", "").replace("🥗", "").replace("🏋️", "").replace("📊", "")
    texto = texto.replace("**", "").replace("###", "").replace("##", "")
    return texto.encode('latin-1', 'replace').decode('latin-1')

def converter_json_para_texto(dados, tipo):
    """Converte a estrutura JSON em texto corrido para o PDF"""
    texto = ""
    if tipo == "dieta":
        for item in dados:
            texto += f"\n[{item.get('dia', 'Dia')}]\n"
            texto += f"Refeicoes: {item.get('refeicoes', '')}\n"
            texto += f"Macros: {item.get('macros', '')}\n"
            texto += "-" * 40 + "\n"
    elif tipo == "treino":
        for item in dados:
            texto += f"\n[{item.get('dia', 'Dia')}] - {item.get('titulo', '')}\n"
            texto += f"Detalhes: {item.get('detalhe', '')}\n"
            texto += "-" * 40 + "\n"
    elif tipo == "suplementos":
        for item in dados:
            texto += f"Item: {item.get('titulo', '')}\n"
            texto += f"Uso: {item.get('detalhe', '')}\n\n"
    elif tipo == "avaliacao":
         texto += f"Tronco: {dados.get('segmentacao', {}).get('tronco', '')}\n"
         texto += f"Membros Sup: {dados.get('segmentacao', {}).get('superior', '')}\n"
         texto += f"Membros Inf: {dados.get('segmentacao', {}).get('inferior', '')}\n"
    
    return texto

class TechnoBoltPDF(FPDF):
    def header(self):
        self.set_fill_color(13, 13, 13)
        self.rect(0, 0, 210, 40, 'F')
        self.set_xy(10, 10)
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(59, 130, 246)
        self.cell(0, 10, "TECHNOBOLT GYM HUB", ln=True, align='L')
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(200, 200, 200)
        self.cell(0, 5, "RELATORIO DE ALTA PERFORMANCE | PHP PROTOCOL", ln=True, align='L')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(59, 130, 246)
        self.cell(0, 10, sanitizar_texto(label), 0, 1, 'L')
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def chapter_body(self, body):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(50, 50, 50)
        texto_limpo = sanitizar_texto(body)
        self.multi_cell(0, 7, texto_limpo)
        self.ln()

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

# --- ENDPOINTS: ANÁLISE (REFATORADO PARA JSON) ---

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

    db.usuarios.update_one({"usuario": usuario}, {"$set": {"nome": nome_completo, "peso": peso, "altura": altura, "genero": genero}})
    
    content = await foto.read()
    img_otimizada = otimizar_imagem(content, quality=85, size=(800, 800))
    imc = peso / ((altura/100)**2)
    
    # [NOVO PROMPT ESTRUTURADO PARA JSON]
    prompt_mestre = f"""
    ATLETA: {nome_completo} ({genero}), IMC {imc:.2f}. META: {objetivo}.
    RESTRIÇÕES: {r_a}, {r_m}, {r_f}. OBS: {info}.

    Aja como um conselho de especialistas da TechnoBolt.
    Analise a foto do corpo e os dados.
    
    ⚠️ IMPORTANTE: VOCÊ DEVE RETORNAR APENAS UM OBJETO JSON VÁLIDO.
    NÃO USE MARKDOWN. A ESTRUTURA DEVE SER EXATAMENTE ESTA:

    {{
      "avaliacao": {{
        "segmentacao": {{
          "tronco": "Texto curto sobre tronco/cabeça",
          "superior": "Texto curto sobre braços/ombros",
          "inferior": "Texto curto sobre pernas"
        }},
        "dobras": {{
          "abdominal": "Texto estimativa",
          "suprailiaca": "Texto estimativa",
          "peitoral": "Texto estimativa"
        }},
        "insight": "3 recomendações técnicas de avaliação"
      }},
      "dieta": [
        {{ "dia": "Segunda", "refeicoes": "Resumo das 3 opções", "macros": "Resumo macros" }},
        {{ "dia": "Terça", "refeicoes": "...", "macros": "..." }},
        {{ "dia": "Quarta", "refeicoes": "...", "macros": "..." }},
        {{ "dia": "Quinta", "refeicoes": "...", "macros": "..." }},
        {{ "dia": "Sexta", "refeicoes": "...", "macros": "..." }},
        {{ "dia": "Sábado", "refeicoes": "...", "macros": "..." }},
        {{ "dia": "Domingo", "refeicoes": "...", "macros": "..." }}
      ],
      "dieta_insight": "Insight nutricional geral",
      "suplementacao": [
        {{ "titulo": "Nome Suplemento", "detalhe": "Motivo, dosagem e como tomar" }},
        {{ "titulo": "Nome Suplemento 2", "detalhe": "..." }},
        {{ "titulo": "Nome Suplemento 3", "detalhe": "..." }}
      ],
      "suplementacao_insight": "Insight ortomolecular",
      "treino": [
        {{ "dia": "Segunda", "titulo": "Grupo Muscular (ex: Peito)", "detalhe": "Lista de exercicios resumida" }},
        {{ "dia": "Terça", "titulo": "...", "detalhe": "..." }},
        {{ "dia": "Quarta", "titulo": "...", "detalhe": "..." }},
        {{ "dia": "Quinta", "titulo": "...", "detalhe": "..." }},
        {{ "dia": "Sexta", "titulo": "...", "detalhe": "..." }},
        {{ "dia": "Sábado", "titulo": "...", "detalhe": "..." }},
        {{ "dia": "Domingo", "titulo": "...", "detalhe": "..." }}
      ],
      "treino_insight": "Insight biomecânico"
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
            # Mapeamos para as chaves que o Flutter espera (r1, r2...), mas agora enviando JSON ou Texto formatado
            # Para o Flutter novo, vamos enviar o JSON completo na chave 'json_full'
            "json_full": conteudo_json,
            # Mantemos compatibilidade parcial caso precise
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

@app.get("/historico/{usuario}")
def buscar_historico(usuario: str):
    user = db.usuarios.find_one({"usuario": usuario})
    if not user: return {"sucesso": True, "historico": []}
    
    # Processamento para garantir que o front receba dados compatíveis
    historico = user.get('historico_dossies', [])
    
    # Adaptação retroativa: Se for JSON novo, formata para o front antigo se necessário
    # (Mas como você atualizou o front, ele vai ler o json_full se existir)
    return {"sucesso": True, "historico": historico}

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

# --- ENDPOINTS: ADMIN & PDF ---

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

@app.get("/analise/baixar-pdf/{usuario}")
def baixar_pdf_completo(usuario: str):
    try:
        user = db.usuarios.find_one({"usuario": usuario})
        if not user or not user.get('historico_dossies'):
            raise HTTPException(404, "Nenhum relatório encontrado.")

        dossie = user['historico_dossies'][-1]
        raw = dossie.get('conteudo_bruto', {})

        # Verifica se temos o novo formato JSON ou o antigo Texto
        if 'json_full' in raw and isinstance(raw['json_full'], dict):
            json_data = raw['json_full']
            r1_txt = converter_json_para_texto(json_data.get('avaliacao'), "avaliacao") + "\n" + json_data.get('avaliacao', {}).get('insight', '')
            r2_txt = converter_json_para_texto(json_data.get('dieta'), "dieta") + "\n" + json_data.get('dieta_insight', '')
            r3_txt = converter_json_para_texto(json_data.get('suplementacao'), "suplementos") + "\n" + json_data.get('suplementacao_insight', '')
            r4_txt = converter_json_para_texto(json_data.get('treino'), "treino") + "\n" + json_data.get('treino_insight', '')
        else:
            # Fallback para relatórios antigos (texto puro)
            r1_txt = raw.get('r1', "Sem dados")
            r2_txt = raw.get('r2', "Sem dados")
            r3_txt = raw.get('r3', "Sem dados")
            r4_txt = raw.get('r4', "Sem dados")

        pdf = TechnoBoltPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, sanitizar_texto(f"ATLETA: {user.get('nome', 'N/A').upper()}"), ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 10, f"DATA DA ANALISE: {dossie.get('data', 'N/A')}", ln=True)
        pdf.ln(10)

        secoes = [
            ("1. AVALIACAO ANTROPOMETRICA", r1_txt),
            ("2. PROTOCOLO NUTRICIONAL", r2_txt),
            ("3. SUPLEMENTACAO AVANCADA", r3_txt),
            ("4. PLANILHA DE TREINO", r4_txt)
        ]

        for titulo, texto in secoes:
            pdf.chapter_title(titulo)
            pdf.chapter_body(str(texto))

        pdf_buffer = io.BytesIO()
        pdf_output = pdf.output(dest='S')
        
        if isinstance(pdf_output, str):
            pdf_buffer.write(pdf_output.encode('latin-1'))
        else:
            pdf_buffer.write(pdf_output)
            
        pdf_buffer.seek(0)
        
        headers = {'Content-Disposition': f'attachment; filename="TechnoBolt_Laudo.pdf"'}
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
