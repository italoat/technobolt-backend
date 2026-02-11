"""
TechnoBolt Gym Hub API - Enterprise Edition (Titanium-Restore & Fix)
Version: 112.0-Stabilized
Architecture: Hexagonal-ish with Chain-of-Thought AI Pipeline & Multi-Level Rotation
Copyright (c) 2026 TechnoBolt Solutions.
"""

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
import time
import functools
import traceback
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict, Union, Callable, TypeVar, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod

# --- FRAMEWORKS E UTILITÁRIOS EXTERNOS ---
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status, Body, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.security import APIKeyHeader

# --- VALIDAÇÃO DE DADOS ---
from pydantic import BaseModel, Field, BeforeValidator, ConfigDict, validator, field_validator, HttpUrl
from typing_extensions import Annotated

# --- BANCO DE DADOS ---
from pymongo import MongoClient, ASCENDING, DESCENDING, IndexModel
from bson.objectid import ObjectId
from pymongo.errors import (
    PyMongoError, 
    ServerSelectionTimeoutError, 
    NetworkTimeout, 
    DuplicateKeyError, 
    OperationFailure
)

# --- IA E PROCESSAMENTO DE IMAGEM ---
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter
import pillow_heif

# --- GERAÇÃO DE RELATÓRIOS ---
from fpdf import FPDF

# ==============================================================================
# SEÇÃO 1: CONFIGURAÇÃO DE LOGGING & MONITORAMENTO
# ==============================================================================

class EnterpriseLogger:
    """
    Configura um sistema de logging estruturado capaz de rastrear transações
    e erros críticos em ambiente de produção.
    """
    @staticmethod
    def setup() -> logging.Logger:
        logger = logging.getLogger("TechnoBoltAPI")
        logger.setLevel(logging.INFO)
        
        # Evita duplicação de handlers no reload do uvicorn
        if logger.hasHandlers():
            logger.handlers.clear()
            
        handler = logging.StreamHandler()
        # Formato rico com timestamp, nível, módulo e linha
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | [%(name)s] %(module)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

logger = EnterpriseLogger.setup()

# ==============================================================================
# SEÇÃO 2: INICIALIZAÇÃO DE SUPORTE A FORMATOS
# ==============================================================================

try:
    pillow_heif.register_heif_opener()
    logger.info("✅ Codec HEIC/HEIF registrado com sucesso (Suporte iOS ativado).")
except Exception as e:
    logger.warning(f"⚠️ Falha ao registrar codec HEIC. Uploads de iPhone podem falhar: {e}")

# ==============================================================================
# SEÇÃO 3: GERENCIAMENTO DE CONFIGURAÇÃO (ENVIRONMENT)
# ==============================================================================

class Settings:
    """
    Singleton para gerenciamento de configurações sensíveis e variáveis de ambiente.
    Realiza validação no startup para garantir integridade.
    """
    def __init__(self):
        logger.info("⚙️  Carregando configurações do sistema...")
        
        # Credenciais de Banco de Dados
        self.MONGO_USER = self._get_env("MONGO_USER", "technobolt")
        self.MONGO_PASS = self._get_env("MONGO_PASS", "tech@132")
        self.MONGO_HOST = self._get_env("MONGO_HOST", "cluster0.zbjsvk6.mongodb.net")
        self.DB_NAME = self._get_env("DB_NAME", "technoboltgym")
        
        # Metadados da API
        self.API_TITLE = "TechnoBolt Gym Hub API"
        self.API_VERSION = "112.0-Titanium-Fixed"
        
        # Carregamento do Pool de Chaves de IA
        self.GEMINI_KEYS = self._load_api_keys()
        
        # ======================================================================
        # DEFINIÇÃO DE MOTORES (RESTAURADOS CONFORME SOLICITAÇÃO)
        # ======================================================================
        
        # Brain Models: Raciocínio profundo e estratégia
        self.REASONING_MODELS = [
            "models/gemini-3-flash-preview", 
            "models/gemini-2.5-flash", 
            "models/gemini-2.0-flash"
        ]
        
        # Structuring Models: Velocidade e formatação JSON
        self.STRUCTURING_MODELS = [
            "models/gemini-flash-latest"
        ]
        
        logger.info(f"🧠 Motores de Raciocínio: {self.REASONING_MODELS}")
        logger.info(f"⚡ Motores de Estruturação: {self.STRUCTURING_MODELS}")

    def _get_env(self, key: str, default: Any = None) -> str:
        value = os.environ.get(key, default)
        if value is None:
            logger.warning(f"⚠️ Variável de ambiente crítica ausente: {key}")
        return value

    def _load_api_keys(self) -> List[str]:
        """Carrega até 20 chaves de API para balanceamento de carga."""
        keys = []
        for i in range(1, 21):
            key_val = os.environ.get(f"GEMINI_CHAVE_{i}")
            if key_val and len(key_val.strip()) > 10:
                keys.append(key_val.strip())
        
        if not keys:
            logger.critical("❌ ERRO CRÍTICO: Nenhuma GEMINI_CHAVE encontrada! O sistema de IA falhará.")
        else:
            logger.info(f"🔑 Pool de IA carregado: {len(keys)} chaves disponíveis.")
        
        return keys

settings = Settings()

# ==============================================================================
# 4. TRATAMENTO DE ERROS E EXCEÇÕES CUSTOMIZADAS
# ==============================================================================

class BaseAPIException(Exception):
    def __init__(self, message: str, status_code: int = 500, details: Any = None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)

class DatabaseConnectionError(BaseAPIException):
    def __init__(self, details: str):
        super().__init__("Serviço de banco de dados indisponível.", 503, details)

class AIReasoningError(BaseAPIException):
    def __init__(self, details: str):
        super().__init__("O Cérebro da IA falhou em gerar uma estratégia válida.", 502, details)

class AIStructuringError(BaseAPIException):
    def __init__(self, details: str):
        super().__init__("A IA falhou em estruturar os dados (Erro de JSON).", 502, details)

class ResourceNotFoundError(BaseAPIException):
    def __init__(self, resource: str):
        super().__init__(f"{resource} não encontrado.", 404)

# ==============================================================================
# 5. DECORATORS E MIDDLEWARE DE PERFORMANCE
# ==============================================================================

def measure_time(func):
    """Mede o tempo de execução de funções assíncronas (Rotas)."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"⏱️  [{func.__name__}] concluído em {elapsed:.2f}ms")
    return wrapper

def sync_measure_time(func):
    """Mede o tempo de execução de funções síncronas."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"⏱️  [{func.__name__}] concluído em {elapsed:.2f}ms")
    return wrapper

# ==============================================================================
# 6. CAMADA DE DADOS: GERENCIADOR MONGODB
# ==============================================================================

PyObjectId = Annotated[str, BeforeValidator(str)]

class MongoManager:
    """
    Gerenciador de Banco de Dados com Padrão Singleton.
    Garante uma única conexão global e gerencia reconexões automáticas.
    """
    _instance = None
    client: MongoClient = None
    db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        try:
            logger.info("🔌 Inicializando driver MongoDB...")
            password = urllib.parse.quote_plus(settings.MONGO_PASS)
            uri = f"mongodb+srv://{settings.MONGO_USER}:{password}@{settings.MONGO_HOST}/?appName=Cluster0"
            
            # Configurações de timeout agressivas para falhar rápido
            self.client = MongoClient(
                uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000,
                maxPoolSize=100,
                minPoolSize=10,
                retryWrites=True
            )
            
            # Health check inicial
            self.client.admin.command('ping')
            self.db = self.client[settings.DB_NAME]
            logger.info(f"✅ Conexão MongoDB estabelecida: {settings.DB_NAME}")
            
            self._ensure_indexes()
            
        except Exception as e:
            logger.critical(f"❌ Falha fatal ao conectar no MongoDB: {e}")

    def _ensure_indexes(self):
        """Cria índices para performance crítica."""
        try:
            if self.db is not None:
                self.db.usuarios.create_index("usuario", unique=True)
                self.db.posts.create_index([("data", DESCENDING)])
                self.db.checkins.create_index([("usuario", ASCENDING), ("data", DESCENDING)])
        except Exception as e:
            logger.warning(f"⚠️ Aviso de índice: {e}")

    def get_collection(self, collection_name: str):
        """Obtém uma coleção garantindo que a conexão está ativa."""
        if self.client is None or self.db is None:
            logger.warning("🔄 Tentando reconexão com o banco...")
            self._initialize()
            
        if self.db is None:
             raise DatabaseConnectionError("Serviço de banco de dados indisponível.")
             
        return self.db[collection_name]

# Instância Global
mongo_db = MongoManager()

# ==============================================================================
# 7. MODELOS DE DADOS (PYDANTIC SCHEMAS)
# ==============================================================================

class MongoBaseModel(BaseModel):
    """Classe base para todos os modelos que interagem com MongoDB."""
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    
    model_config = ConfigDict(
        populate_by_name=True, 
        arbitrary_types_allowed=True, 
        json_encoders={ObjectId: str, datetime: lambda v: v.isoformat()}
    )

class UserLogin(BaseModel):
    usuario: str = Field(..., min_length=3)
    senha: str = Field(..., min_length=3)

class UserRegister(BaseModel):
    usuario: str = Field(..., min_length=3)
    senha: str = Field(..., min_length=3)
    nome: str = Field(..., min_length=2)
    peso: float = Field(..., gt=0, lt=500)
    altura: float = Field(..., gt=0, lt=300)
    genero: str = Field(..., pattern="^(Masculino|Feminino|Outro)$")

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
    texto: str = Field(..., min_length=1, max_length=500)

class ChatMessageRequest(BaseModel):
    remetente: str
    destinatario: str
    texto: str

class AdminUserEdit(BaseModel):
    target_user: str
    status: Optional[str] = None
    creditos: Optional[int] = None

# ==============================================================================
# 8. REPOSITÓRIO DE EXERCÍCIOS (CACHE)
# ==============================================================================

class ExerciseRepository:
    """Gerencia o cache em memória dos exercícios para validação e prompts."""
    _db: Dict[str, str] = {}
    _keys_string: str = ""
    
    @classmethod
    def load(cls):
        try:
            path = "exercises.json"
            if not os.path.exists(path):
                logger.warning("⚠️ exercises.json não encontrado. Validação desabilitada.")
                return

            with open(path, "r", encoding="utf-8") as f:
                cls._db = json.load(f)
                
            all_keys = list(cls._db.keys())
            # Limita a lista para não estourar o contexto da IA (Top 600)
            cls._keys_string = ", ".join(all_keys[:600]) 
            logger.info(f"✅ ExerciseRepository carregado: {len(cls._db)} itens.")
        except Exception as e:
            logger.error(f"❌ Erro crítico no ExerciseRepository: {e}")

    @classmethod
    def get_keys_string(cls) -> str:
        return cls._keys_string

    @classmethod
    def get_db(cls) -> Dict[str, str]:
        return cls._db

ExerciseRepository.load()

# ==============================================================================
# 9. SERVIÇOS DE IA: KEY MANAGER
# ==============================================================================

class KeyRotationManager:
    """
    Gerencia o pool de chaves de API com lógica de Round-Robin e Cooldown.
    Evita erros 429 (Rate Limit) rotacionando chaves automaticamente.
    """
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.cooldowns: Dict[str, float] = {} 
        self.COOLDOWN_SECONDS = 60.0

    def get_available_keys(self) -> List[str]:
        """Retorna chaves que não estão em cooldown."""
        now = time.time()
        # Remove chaves que já passaram do tempo de espera
        self.cooldowns = {k: v for k, v in self.cooldowns.items() if v > now}
        
        available = [k for k in self.keys if k not in self.cooldowns]
        
        # Se todas estiverem bloqueadas, retorna a lista completa (Fail-Open strategy)
        if not available and self.keys:
            logger.warning("⚠️ Todas as chaves em cooldown. Forçando uso do pool completo.")
            return self.keys
            
        random.shuffle(available) # Balanceamento estocástico
        return available

    def report_rate_limit(self, key: str):
        """Bloqueia uma chave temporariamente."""
        logger.warning(f"⚠️ Rate Limit atingido na chave ...{key[-4:]}. Pausando por {self.COOLDOWN_SECONDS}s.")
        self.cooldowns[key] = time.time() + self.COOLDOWN_SECONDS

key_manager = KeyRotationManager(settings.GEMINI_KEYS)

# ==============================================================================
# 10. SERVIÇOS DE IA: CORE LOGIC & CHAIN OF THOUGHT
# ==============================================================================

class JSONRepairKit:
    """
    Ferramentas avançadas para reparo de strings JSON malformadas.
    Resolve o erro 'Expecting , delimiter' inserindo vírgulas faltantes.
    """
    
    @staticmethod
    def fix_json_string(text: str) -> str:
        try:
            text = text.strip()
            # Remove blocos markdown
            if "```" in text:
                text = re.sub(r'```json|```', '', text).strip()
            
            # Remove comentários estilo C/JS
            text = re.sub(r'//.*?\n|/\*.*?\*/', '', text, flags=re.S)
            
            # Remove vírgulas trailing (erro comum)
            text = re.sub(r',(\s*[}\]])', r'\1', text)
            
            # --- FIX NUCLEAR PARA ERRO DE VÍRGULA FALTANDO ---
            # Caso 1: Entre objetos/arrays (ex: } {)
            text = re.sub(r'([}\]])\s*([{\[])', r'\1,\2', text)
            
            # Caso 2: Entre valor e chave (ex: "valor" "chave")
            # Procura aspas, espaço, e novas aspas, exceto se tiver : no meio
            text = re.sub(r'("\s*)\s+"', r'\1,"', text)
            
            # Caso 3: Entre número e chave (ex: 123 "chave")
            text = re.sub(r'(\d+)\s+"', r'\1,"', text)

            # Balanceamento de chaves
            open_braces = text.count('{')
            close_braces = text.count('}')
            if open_braces > close_braces: text += '}' * (open_braces - close_braces)
                
            open_brackets = text.count('[')
            close_brackets = text.count(']')
            if open_brackets > close_brackets: text += ']' * (open_brackets - close_brackets)
                
            return text
        except: return text

    @staticmethod
    def safe_parse(text: str) -> Dict:
        """Pipeline de tentativa de parseamento robusto."""
        # 1. Tentativa Direta
        try: return json.loads(text)
        except: pass
        
        # 2. Extração de Bloco Regex
        try:
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match: return json.loads(match.group(1))
        except: pass
            
        # 3. Reparo Agressivo
        try:
            repaired = JSONRepairKit.fix_json_string(text)
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON Parse Falhou. Texto: {text[:100]}...")
            raise AIStructuringError("Falha crítica na formatação do JSON.")

class AIOrchestrator:
    """
    Orquestrador principal da IA. Implementa a arquitetura Chain of Thought
    com rodízio de chaves aninhado e correção automática de geração vazia.
    """
    
    @staticmethod
    def _call_gemini_with_retry(model_name: str, prompt: str, image_bytes: Optional[bytes] = None, 
                              json_mode: bool = False, temperature: float = 0.7) -> str:
        """
        NÚCLEO DO RODÍZIO: Tenta TODAS as chaves disponíveis para o modelo.
        """
        keys = key_manager.get_available_keys()
        if not keys:
            raise AIProcessingError("Sem chaves de API disponíveis no pool.")
            
        last_error = None
        
        # Itera sobre todas as chaves disponíveis
        for api_key in keys:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                
                config = genai.types.GenerationConfig(
                    response_mime_type="application/json" if json_mode else "text/plain",
                    max_output_tokens=8192,
                    temperature=temperature
                )
                
                inputs = [prompt]
                if image_bytes:
                    inputs.append({"mime_type": "image/jpeg", "data": image_bytes})
                
                # Chamada Síncrona
                response = model.generate_content(inputs, generation_config=config)
                
                if response and response.text:
                    logger.info(f"   ✅ Sucesso: {model_name} (Key final ...{api_key[-4:]})")
                    return response.text
                
            except Exception as e:
                err_str = str(e)
                # Verifica se é erro de cota
                if "429" in err_str or "Resource exhausted" in err_str:
                    key_manager.report_rate_limit(api_key)
                
                logger.warning(f"   ⚠️ Falha parcial: {model_name} (Key ...{api_key[-4:]}): {err_str[:100]}")
                last_error = e
                # Delay para evitar spam na API
                time.sleep(1.0) 
                continue # Tenta próxima chave
                
        # Se saiu do loop, falhou com todas as chaves
        logger.error(f"❌ Falha total ao invocar modelo {model_name} com todas as chaves disponíveis.")
        raise last_error if last_error else Exception(f"Falha total no modelo {model_name}")

    @staticmethod
    def execute_chain_of_thought(context_prompt: str, image_bytes: Optional[bytes]) -> Dict:
        """
        Pipeline Principal (Two-Pass Generation):
        1. FASE 1: Raciocínio (Brain) -> Gera texto livre e detalhado.
        2. FASE 2: Estruturação (Formatter) -> Converte para JSON estrito.
        """
        
        # --- FASE 1: RACIOCÍNIO (Brain) ---
        strategy_text = None
        
        # Tenta cada modelo de raciocínio na ordem de preferência
        for model in settings.REASONING_MODELS:
            try:
                logger.info(f"🧠 [Fase 1 - Brain] Iniciando Raciocínio com {model}...")
                
                prompt_p1 = context_prompt + "\n\nINSTRUÇÃO CRÍTICA: Gere uma estratégia textual DETALHADA. Não use JSON ainda. Foque na qualidade técnica. Seja VERBOSO."
                
                strategy_text = AIOrchestrator._call_gemini_with_retry(
                    model_name=model,
                    prompt=prompt_p1,
                    image_bytes=image_bytes,
                    json_mode=False,
                    temperature=0.7 
                )
                if strategy_text:
                    logger.info("🧠 [Fase 1] Estratégia gerada com sucesso.")
                    break 
            except Exception as e:
                logger.warning(f"⚠️ Modelo de raciocínio {model} esgotado. Tentando próximo...")
                continue
        
        if not strategy_text:
            # Fallback final: Tenta usar o modelo de estruturação para pensar
            try:
                logger.warning("⚠️ Todos modelos de raciocínio falharam. Usando fallback...")
                strategy_text = AIOrchestrator._call_gemini_with_retry(
                    model_name=settings.STRUCTURING_MODELS[0],
                    prompt=context_prompt,
                    image_bytes=image_bytes,
                    temperature=0.7
                )
            except Exception as e:
                raise AIProcessingError(f"Falha catastrófica na IA (Fase 1 - Brain): {e}")

        # --- FASE 2: ESTRUTURAÇÃO (Formatter) ---
        logger.info("⚡ [Phase 2] Iniciando Estruturação JSON...")
        
        exercise_list = ExerciseRepository.get_keys_string()
        
        prompt_p2 = f"""
        TASK: Convert this Strategy into VALID JSON.
        STRATEGY:
        {strategy_text}
        
        RULES:
        1. Output ONLY JSON.
        2. Validate Exercise Names against: [{exercise_list}].
        3. CRITICAL: 'dieta' must be an array of 7 objects.
        4. CRITICAL: 'treino' must be an array of 7 objects.
        5. CRITICAL: 'suplementacao' must be a non-empty array.
        
        REQUIRED SCHEMA (NO EMPTY LISTS):
        {{
          "avaliacao": {{ "segmentacao": {{...}}, "dobras": {{...}}, "analise_postural": "...", "simetria": "...", "insight": "..." }},
          "dieta": [ {{ "dia": "Segunda", "foco_nutricional": "...", "refeicoes": [ {{ "horario": "...", "nome": "...", "alimentos": "..." }} ], "macros_totais": "..." }}, ... ],
          "dieta_insight": "...",
          "suplementacao": [ {{ "nome": "...", "dose": "...", "horario": "...", "motivo": "..." }} ],
          "suplementacao_insight": "...",
          "treino": [ 
             {{ "dia": "Segunda", "foco": "...", "exercicios": [ {{ "nome": "...", "series_reps": "...", "execucao": "...", "justificativa_individual": "..." }} ], "treino_alternativo": "...", "justificativa": "..." }},
             ...
          ],
          "treino_insight": "..."
        }}
        """
        
        try:
            # Tenta estruturar usando o modelo rápido com temperatura 0.0 (Correção do erro de JSON)
            json_text = AIOrchestrator._call_gemini_with_retry(
                model_name=settings.STRUCTURING_MODELS[0],
                prompt=prompt_p2,
                image_bytes=None, 
                json_mode=True,
                temperature=0.0 # ZERO CRIATIVIDADE AQUI PARA EVITAR ERROS DE SINTAXE
            )
            data = JSONRepairKit.parse_robust(json_text)
            
            # Validação Failsafe
            if not data.get('dieta'): data['dieta'] = []
            if not data.get('treino'): data['treino'] = []
            
            return data
            
        except Exception as e:
            logger.error(f"Fase 2 falhou. Tentando parsear texto original... {e}")
            return JSONRepairKit.safe_parse(strategy_text)

    @staticmethod
    def simple_generation(prompt: str, image_bytes: Optional[bytes] = None) -> str:
        """Geração rápida para tarefas simples (ex: comentários)."""
        try:
            return AIOrchestrator._call_gemini_with_retry(
                settings.STRUCTURING_MODELS[0], 
                prompt, 
                image_bytes, 
                json_mode=False
            )
        except: return "Estou analisando seu treino... continue focado!"

class ImageService:
    """Utilitários para processamento e otimização de imagens."""
    
    @staticmethod
    def optimize(file_bytes: bytes, quality: int = 75, max_size: tuple = (800, 800)) -> bytes:
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode != 'RGB':
                    img = img.convert("RGB")
                img.thumbnail(max_size)
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality, optimize=True)
                return output.getvalue()
        except Exception as e:
            logger.error(f"Erro na otimização de imagem: {e}")
            return file_bytes

class PDFReport(FPDF):
    """Gerador de relatórios PDF customizado."""
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.col_bg = (20, 20, 25)
        self.col_text = (230, 230, 230)
        self.col_accent = (0, 200, 255)

    def sanitize(self, txt: Any) -> str:
        if not txt: return ""
        s = str(txt).replace("’", "'").replace("–", "-")
        return s.encode('latin-1', 'replace').decode('latin-1')

    def header(self):
        self.set_fill_color(*self.col_bg)
        self.rect(0, 0, 210, 297, 'F')
        self.set_font("Arial", "B", 20)
        self.set_text_color(*self.col_accent)
        self.cell(0, 10, "TECHNOBOLT PROTOCOL", 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, label):
        self.set_font("Arial", "B", 14)
        self.set_text_color(*self.col_accent)
        self.cell(0, 10, self.sanitize(label.upper()), 0, 1, 'L')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def chapter_body(self, body):
        self.set_font("Arial", "", 10)
        self.set_text_color(*self.col_text)
        self.multi_cell(0, 6, self.sanitize(body))
        self.ln()
    
    def card(self, title, body):
        self.set_fill_color(*self.col_card)
        self.set_text_color(*self.col_azul)
        self.set_font("Arial", "B", 11)
        self.multi_cell(0, 6, self.sanitize(title), fill=True)
        self.set_text_color(*self.col_texto)
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 6, self.sanitize(body), fill=True)
        self.ln(2)

# ==============================================================================
# SEÇÃO 11: HELPERS DE NEGÓCIO
# ==============================================================================

def normalizar_texto(texto: str) -> str:
    if not texto: return ""
    return "".join(c for c in unicodedata.normalize('NFD', str(texto)) if unicodedata.category(c) != 'Mn').lower().strip()

def validar_exercicios_final(treino_data: list) -> list:
    """Validação final pós-IA. Tenta casar nomes de exercícios."""
    db = ExerciseRepository.get_db()
    if not treino_data or not db: return treino_data
    
    base_url = "[https://raw.githubusercontent.com/italoat/technobolt-backend/main/assets/exercises](https://raw.githubusercontent.com/italoat/technobolt-backend/main/assets/exercises)"
    db_map = {normalizar_texto(k): v for k, v in db.items()}
    db_titles = {normalizar_texto(k): k for k, v in db.items()}

    for dia in treino_data:
        if 'exercicios' not in dia: continue
        corrected_exs = []
        for ex in dia['exercicios']:
            raw_name = ex.get('nome', 'Exercício Geral')
            norm_name = normalizar_texto(raw_name)
            path = None
            final_name = raw_name
            
            if norm_name in db_map:
                path = db_map[norm_name]
                final_name = db_titles[norm_name]
            else:
                matches = difflib.get_close_matches(norm_name, db_map.keys(), n=1, cutoff=0.6)
                if matches:
                    path = db_map[matches[0]]
                    final_name = db_titles[matches[0]]
                else:
                    for k in db_map.keys():
                        if k in norm_name or norm_name in k:
                            path = db_map[k]
                            final_name = db_titles[k]
                            break
                    if not path and "polichinelo" in db_map:
                        path = db_map["polichinelo"]
                        final_name = f"{raw_name} (Adaptado)"

            ex['nome'] = str(final_name).title()
            if path: ex['imagens_demonstracao'] = [f"{base_url}/{path}/0.jpg", f"{base_url}/{path}/1.jpg"]
            else: ex['imagens_demonstracao'] = []
            corrected_exs.append(ex)
        dia['exercicios'] = corrected_exs
    return treino_data

def calcular_medalha(username: str) -> str:
    try:
        user = mongo_db.get_collection("usuarios").find_one({"usuario": username})
        return "🥇" if user and user.get('pontos', 0) > 1000 else ""
    except: return ""

# ==============================================================================
# SEÇÃO 12: APLICAÇÃO FASTAPI & ROTAS
# ==============================================================================

app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ROTAS DE AUTENTICAÇÃO ---

@app.post("/auth/login", tags=["Auth"])
@sync_measure_time
def login(dados: UserLogin):
    col = mongo_db.get_collection("usuarios")
    user = col.find_one({"usuario": dados.usuario, "senha": dados.senha})
    
    if not user: raise HTTPException(401, "Credenciais inválidas")
    if user.get("status") != "ativo" and not user.get("is_admin"): raise HTTPException(403, "Conta pendente.")
    
    return {
        "sucesso": True,
        "dados": {
            "usuario": user['usuario'],
            "nome": user.get('nome'),
            "is_admin": user.get('is_admin', False),
            "creditos": user.get('avaliacoes_restantes', 0),
            "pontos": user.get('pontos', 0),
            "foto_perfil": user.get('foto_perfil'),
            "peso": user.get('peso'),
            "altura": user.get('altura'),
            "genero": user.get('genero'),
            "restricoes_alim": user.get('restricoes_alim'),
            "restricoes_fis": user.get('restricoes_fis'),
            "medicamentos": user.get('medicamentos'),
            "info_add": user.get('info_add')
        }
    }

@app.post("/auth/registro", tags=["Auth"])
def registrar(dados: UserRegister):
    col = mongo_db.get_collection("usuarios")
    if col.find_one({"usuario": dados.usuario}): raise HTTPException(400, "Usuário já existe")
    col.insert_one({**dados.model_dump(), "status": "pendente", "avaliacoes_restantes": 0, "pontos": 0, "historico_dossies": [], "is_admin": False, "created_at": datetime.now()})
    return {"sucesso": True, "mensagem": "Registro realizado."}

@app.post("/perfil/atualizar", tags=["Perfil"])
def atualizar_perfil(dados: UserUpdate):
    col = mongo_db.get_collection("usuarios")
    data = {k: v for k, v in dados.model_dump(exclude={'usuario'}).items() if v is not None}
    col.update_one({"usuario": dados.usuario}, {"$set": data})
    return {"sucesso": True}

# --- ROTA CORE: ANÁLISE ---

@app.post("/analise/executar", tags=["Analise"])
@measure_time
async def executar_analise(
    usuario: str = Form(...), nome_completo: str = Form(...), peso: str = Form(...), 
    altura: str = Form(...), objetivo: str = Form(...), genero: str = Form("Masculino"),
    observacoes: str = Form(""), foto: UploadFile = File(...)
):
    logger.info(f"🚀 Iniciando análise completa para: {usuario}")
    
    try:
        peso_float = float(str(peso).replace(',', '.'))
        alt = float(str(altura).replace(',', '.').replace('cm', '').strip())
        altura_int = int(alt * 100) if alt < 3.0 else int(alt)
    except: peso_float = 70.0; altura_int = 175
    
    col = mongo_db.get_collection("usuarios")
    col.update_one({"usuario": usuario}, {"$set": {"nome": nome_completo, "peso": peso_float, "altura": altura_int, "genero": genero, "info_add": observacoes}})
    user_data = col.find_one({"usuario": usuario})
    if not user_data: raise HTTPException(404, "Erro de consistência de usuário.")

    raw_img = await foto.read()
    img_opt = ImageService.optimize(raw_img)
    
    prompt = f"""
    ACT AS AN ELITE SPORTS SCIENTIST. CREATE THE ULTIMATE PROTOCOL.
    CLIENT: {nome_completo} ({genero}), {peso_float}kg, {altura_int}cm.
    GOAL: {objetivo}. RESTRICTIONS: {user_data.get('restricoes_fis')}, {user_data.get('restricoes_alim')}.
    TASKS: 1. PHYSIQUE ANALYSIS. 2. DIET (7 DAYS). 3. TRAINING (7 DAYS). 4. SUPPLEMENTS.
    """
    
    try:
        result_json = AIOrchestrator.execute_chain_of_thought(prompt, img_opt)
    except Exception as e:
        logger.error(f"CoT Failure: {e}")
        raise HTTPException(503, "IA indisponível no momento. Tente novamente.")

    if 'treino' in result_json:
        result_json['treino'] = validar_exercicios_final(result_json['treino'])

    dossie = {
        "id": str(ObjectId()),
        "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "timestamp": datetime.now(),
        "peso_reg": peso_float,
        "conteudo_bruto": {
            "json_full": result_json,
            "r1": str(result_json.get('avaliacao', {}).get('insight', '')),
            "r2": str(result_json.get('dieta_insight', '')),
            "r3": str(result_json.get('suplementacao_insight', '')),
            "r4": str(result_json.get('treino_insight', ''))
        }
    }
    
    update = {"$push": {"historico_dossies": dossie}}
    if not user_data.get('is_admin'): update["$inc"] = {"avaliacoes_restantes": -1}
    col.update_one({"usuario": usuario}, update)
    
    return {"sucesso": True, "resultado": dossie}

@app.post("/analise/regenerar-secao", tags=["Analise"])
async def regenerar_secao(dados: dict = Body(...)):
    col = mongo_db.get_collection("usuarios")
    user = col.find_one({"usuario": dados.get("usuario")})
    
    if not user or (user.get('avaliacoes_restantes', 0) <= 0 and not user.get('is_admin')):
        return {"sucesso": False, "mensagem": "Saldo insuficiente."}
        
    secao = dados.get("secao")
    dia = dados.get("dia", "")
    prompt = f"Regenerate '{secao}' for {user.get('nome')}. Focus: {dia if dia else 'Full Week'}. Make it HARDCORE."
    
    try:
        new_content = AIOrchestrator.execute_chain_of_thought(prompt, None)
        last = user['historico_dossies'][-1]
        full = last['conteudo_bruto']['json_full']
        
        if secao in new_content:
            full[secao] = new_content[secao]
            if f"{secao}_insight" in new_content: full[f"{secao}_insight"] = new_content[f"{secao}_insight"]
        
        if secao == 'treino':
            full['treino'] = validar_exercicios_final(full['treino'])
            
        col.update_one(
            {"usuario": dados.get("usuario"), "historico_dossies.data": last['data']},
            {"$set": {"historico_dossies.$.conteudo_bruto.json_full": full}}
        )
        return {"sucesso": True, "resultado": last}
    except: return {"sucesso": False}

# --- ROTA LEGADA PARA HISTÓRICO ---

@app.get("/historico/{usuario}", tags=["Perfil"])
def buscar_historico(usuario: str):
    col = mongo_db.get_collection("usuarios")
    user = col.find_one({"usuario": usuario})
    if not user: return {"sucesso": True, "historico": []}
    
    return {
        "sucesso": True, 
        "historico": jsonable_encoder(user.get('historico_dossies', [])), 
        "creditos": user.get('avaliacoes_restantes', 0), 
        "perfil": {
            "peso": user.get('peso'), "altura": user.get('altura'), "genero": user.get('genero'),
            "restricoes_alim": user.get('restricoes_alim'), "restricoes_fis": user.get('restricoes_fis'),
            "medicamentos": user.get('medicamentos'), "info_add": user.get('info_add'),
            "creditos": user.get('avaliacoes_restantes', 0)
        }
    }

# --- ROTAS SOCIAIS ---

@app.get("/social/feed", tags=["Social"])
def get_feed():
    col = mongo_db.get_collection("posts")
    posts = list(col.find().sort("data", DESCENDING).limit(50))
    for p in posts: p['_id'] = str(p['_id'])
    return {"sucesso": True, "feed": posts}

@app.post("/social/postar", tags=["Social"])
async def postar(usuario: str = Form(...), legenda: str = Form(...), imagem: UploadFile = File(...)):
    img_bytes = await imagem.read()
    img_opt = ImageService.optimize(img_bytes)
    cmt = AIOrchestrator.simple_generation(f"Comentário gym bro para: {legenda}", img_opt)
    mongo_db.get_collection("posts").insert_one({"autor": usuario, "legenda": legenda, "imagem": base64.b64encode(img_opt).decode('utf-8'), "data": datetime.now().isoformat(), "likes": [], "comentarios": [{"autor": "TechnoBolt AI", "texto": cmt}] if cmt else []})
    return {"sucesso": True}

@app.post("/social/post/deletar", tags=["Social"])
def deletar_post_social(dados: SocialPostRequest):
    col = mongo_db.get_collection("posts")
    res = col.delete_one({"_id": ObjectId(dados.post_id), "autor": dados.usuario})
    return {"sucesso": res.deleted_count > 0}

@app.post("/social/curtir", tags=["Social"])
def curtir_post(dados: SocialPostRequest):
    col = mongo_db.get_collection("posts")
    oid = ObjectId(dados.post_id)
    post = col.find_one({"_id": oid})
    if not post: return {"sucesso": False}
    if dados.usuario in post.get("likes", []):
        col.update_one({"_id": oid}, {"$pull": {"likes": dados.usuario}})
    else:
        col.update_one({"_id": oid}, {"$addToSet": {"likes": dados.usuario}})
    return {"sucesso": True}

@app.post("/social/comentar", tags=["Social"])
def postar_comentario(dados: SocialCommentRequest):
    col = mongo_db.get_collection("posts")
    cmt = {"autor": dados.usuario, "texto": dados.texto, "data": datetime.now().isoformat()}
    col.update_one({"_id": ObjectId(dados.post_id)}, {"$push": {"comentarios": cmt}})
    return {"sucesso": True}

# --- GAMIFICAÇÃO & VISION AI ---

@app.get("/social/ranking", tags=["Social"])
def get_ranking():
    col = mongo_db.get_collection("usuarios")
    users = list(col.find({"is_admin": False}, {"nome": 1, "usuario": 1, "pontos": 1, "foto_perfil": 1, "_id": 0}).sort("pontos", DESCENDING).limit(50))
    return {"sucesso": True, "ranking": users}

@app.get("/social/checkins", tags=["Social"])
def get_checkins(usuario: str):
    col = mongo_db.get_collection("checkins")
    checkins = list(col.find({"usuario": usuario}))
    formatted = {}
    for c in checkins:
        try:
            d = datetime.fromisoformat(c['data']).day
            formatted[d] = c['tipo']
        except: pass
    return {"sucesso": True, "checkins": formatted}

@app.post("/social/validar-conquista", tags=["Social"])
async def validar_conquista(usuario: str = Form(...), tipo: str = Form(...), foto: UploadFile = File(...)):
    col = mongo_db.get_collection("checkins")
    now = datetime.now()
    today = datetime(now.year, now.month, now.day).isoformat()
    if col.find_one({"usuario": usuario, "data": {"$gte": today}}): return {"sucesso": False, "mensagem": "Checkin já realizado."}

    content = await foto.read()
    img_opt = ImageService.optimize(content)
    resp = AIOrchestrator.simple_generation(f"Valide se esta imagem comprova um treino de {tipo}. Responda APROVADO ou REPROVADO.", img_opt)
    
    if resp and "APROVADO" in resp.upper():
        col.insert_one({"usuario": usuario, "tipo": tipo, "data": now.isoformat(), "pontos": 50})
        mongo_db.get_collection("usuarios").update_one({"usuario": usuario}, {"$inc": {"pontos": 50}})
        return {"sucesso": True, "aprovado": True, "pontos": 50}
    return {"sucesso": True, "aprovado": False}

# --- CHAT & ADMIN ---

@app.get("/chat/mensagens", tags=["Chat"])
def get_msgs(user1: str, user2: str):
    col = mongo_db.get_collection("chat")
    q = {"$or": [{"remetente": user1, "destinatario": user2}, {"remetente": user2, "destinatario": user1}]}
    msgs = list(col.find(q).sort("timestamp", ASCENDING))
    for m in msgs: m['_id'] = str(m['_id'])
    return {"sucesso": True, "mensagens": msgs}

@app.post("/chat/enviar", tags=["Chat"])
def send_msg(dados: ChatMessageRequest):
    col = mongo_db.get_collection("chat")
    col.insert_one(dados.model_dump())
    return {"sucesso": True}

@app.get("/chat/usuarios", tags=["Chat"])
def list_chat_users(usuario_atual: str):
    col = mongo_db.get_collection("usuarios")
    users = list(col.find({"usuario": {"$ne": usuario_atual}}, {"usuario": 1, "nome": 1, "_id": 0}))
    return {"sucesso": True, "usuarios": users}

@app.get("/admin/listar", tags=["Admin"])
def admin_list():
    users = list(mongo_db.get_collection("usuarios").find())
    for u in users: u['_id'] = str(u['_id'])
    return {"sucesso": True, "usuarios": users}

@app.post("/admin/editar", tags=["Admin"])
def admin_edit(dados: AdminUserEdit):
    upd = {}
    if dados.status: upd["status"] = dados.status
    if dados.creditos is not None: upd["avaliacoes_restantes"] = dados.creditos
    mongo_db.get_collection("usuarios").update_one({"usuario": dados.target_user}, {"$set": upd})
    return {"sucesso": True}

@app.post("/admin/excluir", tags=["Admin"])
def admin_del(dados: AdminUserEdit):
    mongo_db.get_collection("usuarios").delete_one({"usuario": dados.target_user})
    return {"sucesso": True}

@app.get("/setup/criar-admin", tags=["Admin"])
def create_admin():
    col = mongo_db.get_collection("usuarios")
    if col.find_one({"usuario": "admin"}): return {"sucesso": False}
    col.insert_one({"usuario": "admin", "senha": "123", "nome": "Admin", "is_admin": True, "status": "ativo", "avaliacoes_restantes": 9999})
    return {"sucesso": True}

# --- PDF ---

@app.get("/analise/baixar-pdf/{usuario}", tags=["Export"])
def download_pdf(usuario: str):
    try:
        col = mongo_db.get_collection("usuarios")
        user = col.find_one({"usuario": usuario})
        if not user or not user.get('historico_dossies'): raise HTTPException(404)
        
        dossie = user['historico_dossies'][-1]
        data = dossie['conteudo_bruto']['json_full']
        
        pdf = PDFReport()
        pdf.add_page()
        pdf.chapter_title(f"RELATORIO: {user.get('nome', '').upper()}")
        
        if 'avaliacao' in data: pdf.card("Avaliação", data['avaliacao'].get('insight', ''))
        if 'dieta' in data:
            pdf.add_page()
            pdf.chapter_title("DIETA 7 DIAS")
            for d in data['dieta']:
                pdf.card(f"{d.get('dia')} - {d.get('foco_nutricional')}", d.get('macros_totais'))
                for r in d.get('refeicoes', []): pdf.chapter_body(f"{r.get('horario')}: {r.get('alimentos')}")
        if 'treino' in data:
            pdf.add_page()
            pdf.chapter_title("TREINO 7 DIAS")
            for t in data['treino']:
                pdf.card(f"{t.get('dia')} - {t.get('foco')}", t.get('justificativa', ''))
                for ex in t.get('exercicios', []): pdf.chapter_body(f"> {ex.get('nome')} [{ex.get('series_reps')}]")
        
        buf = io.BytesIO()
        out = pdf.output(dest='S')
        if isinstance(out, str): buf.write(out.encode('latin-1'))
        else: buf.write(out)
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/pdf", headers={'Content-Disposition': 'attachment; filename="report.pdf"'})
    except Exception as e:
        logger.error(f"PDF Err: {e}")
        raise HTTPException(500)
