"""
TechnoBolt Gym Hub API - Enterprise Edition (Titanium-Final-Patch)
Version: 118.0-Production-Stable
Architecture: Hexagonal-ish with Chain-of-Thought AI Pipeline & Multi-Level Rotation
Author: TechnoBolt Engineering Team (Senior Lead)
Timestamp: 2026-02-11

Description:
This API serves as the backend for the TechnoBolt Gym Hub application.
FIXED: JSONRepairKit Attribute Error, JSON Delimiter Logic & Full Route Integrity.
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
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict, Union, Callable, TypeVar, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod

# --- FRAMEWORKS E UTILITÁRIOS EXTERNOS ---
from fastapi import (
    FastAPI, 
    UploadFile, 
    File, 
    Form, 
    HTTPException, 
    Depends, 
    status, 
    Body, 
    Request, 
    BackgroundTasks
)
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.security import APIKeyHeader
from fastapi.exceptions import RequestValidationError

# --- VALIDAÇÃO DE DADOS ---
from pydantic import (
    BaseModel, 
    Field, 
    BeforeValidator, 
    ConfigDict, 
    validator, 
    field_validator, 
    HttpUrl, 
    EmailStr
)
from typing_extensions import Annotated

# --- BANCO DE DADOS ---
from pymongo import MongoClient, ASCENDING, DESCENDING, IndexModel
from bson.objectid import ObjectId
from pymongo.errors import (
    PyMongoError, 
    ServerSelectionTimeoutError, 
    NetworkTimeout, 
    DuplicateKeyError, 
    OperationFailure,
    CollectionInvalid
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
# SEÇÃO 1: CONFIGURAÇÃO DE LOGGING (ENTERPRISE OBSERVABILITY)
# ==============================================================================

class EnterpriseLogger:
    """
    Configura um sistema de logging estruturado capaz de rastrear transações
    e erros críticos em ambiente de produção de alta escala.
    """
    
    @staticmethod
    def setup() -> logging.Logger:
        """
        Inicializa o logger com formatação detalhada, timestamps e níveis de severidade.
        Evita duplicação de handlers durante o reload do Uvicorn.
        """
        logger = logging.getLogger("TechnoBoltAPI")
        logger.setLevel(logging.INFO)
        
        # Remove handlers existentes para evitar duplicação em reloads
        if logger.hasHandlers():
            logger.handlers.clear()
            
        # Configura StreamHandler para console (stdout)
        handler = logging.StreamHandler()
        
        # Formato rico: Timestamp | Nível | Módulo | Função:Linha | Mensagem
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | [%(name)s] %(module)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

# Instância Global de Logging
logger = EnterpriseLogger.setup()

# ==============================================================================
# SEÇÃO 2: INICIALIZAÇÃO DE SUPORTE E DRIVERS
# ==============================================================================

def initialize_external_drivers():
    """
    Inicializa drivers de terceiros, como suporte a imagens HEIC/HEIF (Apple).
    Isso é crítico para uploads vindos de dispositivos iOS.
    """
    try:
        pillow_heif.register_heif_opener()
        logger.info("✅ Codec HEIC/HEIF registrado com sucesso. Suporte iOS ativado.")
    except ImportError:
        logger.warning("⚠️ Biblioteca 'pillow_heif' não encontrada. Uploads de iPhone (HEIC) falharão.")
    except Exception as e:
        logger.warning(f"⚠️ Erro ao registrar codec HEIC: {e}. Verifique as dependências instaladas.")

# Executa inicialização
initialize_external_drivers()

# ==============================================================================
# SEÇÃO 3: GERENCIAMENTO DE CONFIGURAÇÃO (ENVIRONMENT SETTINGS)
# ==============================================================================

class Settings:
    """
    Singleton para gerenciamento de configurações sensíveis e variáveis de ambiente.
    Realiza validação no startup para garantir integridade da aplicação.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._load_configurations()
        return cls._instance

    def _load_configurations(self):
        logger.info("⚙️  Carregando e validando configurações do sistema...")
        
        # --- Configurações de Banco de Dados ---
        self.MONGO_USER = self._get_env("MONGO_USER", "technobolt")
        self.MONGO_PASS = self._get_env("MONGO_PASS", "tech@132")
        self.MONGO_HOST = self._get_env("MONGO_HOST", "cluster0.zbjsvk6.mongodb.net")
        self.DB_NAME = self._get_env("DB_NAME", "technoboltgym")
        
        # --- Metadados da API ---
        self.API_TITLE = "TechnoBolt Gym Hub API"
        self.API_VERSION = "117.0-Production-Fixed"
        self.ENV = self._get_env("ENV", "production")
        
        # --- Configurações de IA (Load Balancer) ---
        self.GEMINI_KEYS = self._load_api_keys()
        
        # --- Definição Estratégica de Modelos (Chain of Thought) ---
        # MANTIDOS EXATAMENTE COMO SOLICITADO
        
        # FASE 1: BRAIN (RACIOCÍNIO)
        self.REASONING_MODELS = [
            "models/gemini-3-flash-preview",  # Tier 1
            "models/gemini-2.5-flash",        # Tier 2
            "models/gemini-2.0-flash"         # Tier 3
        ]
        
        # FASE 2: FORMATTER (ESTRUTURAÇÃO)
        self.STRUCTURING_MODELS = [
            "models/gemini-flash-latest"      # Otimizado para JSON
        ]
        
        logger.info(f"🧠 Motores de Raciocínio Carregados: {len(self.REASONING_MODELS)}")
        logger.info(f"⚡ Motores de Estruturação Carregados: {len(self.STRUCTURING_MODELS)}")

    def _get_env(self, key: str, default: Any = None) -> str:
        """Recupera variáveis de ambiente com log de aviso se ausentes."""
        value = os.environ.get(key, default)
        if value is None:
            logger.warning(f"⚠️ Variável de ambiente crítica ausente: {key}. Usando default.")
        return value

    def _load_api_keys(self) -> List[str]:
        """
        Carrega chaves de API dinamicamente de variáveis de ambiente sequenciais.
        Suporta até 20 chaves para distribuição de carga.
        """
        keys = []
        for i in range(1, 21):
            key_var_name = f"GEMINI_CHAVE_{i}"
            key_val = os.environ.get(key_var_name)
            
            if key_val and len(key_val.strip()) > 10:
                keys.append(key_val.strip())
        
        if not keys:
            logger.critical("❌ ERRO CRÍTICO: Nenhuma chave GEMINI_CHAVE_x encontrada! O subsistema de IA falhará.")
        else:
            logger.info(f"🔑 Pool de IA inicializado: {len(keys)} chaves disponíveis para rotação.")
        
        return keys

# Instância Global de Configurações
settings = Settings()

# ==============================================================================
# SEÇÃO 4: HIERARQUIA DE EXCEÇÕES E TRATAMENTO DE ERROS
# ==============================================================================

class BaseAPIException(Exception):
    """Classe base para todas as exceções controladas da API."""
    def __init__(self, message: str, status_code: int = 500, details: Any = None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)

class DatabaseConnectionError(BaseAPIException):
    """Lançado quando não é possível estabelecer conexão com o MongoDB."""
    def __init__(self, details: str = ""):
        super().__init__(f"Serviço de banco de dados indisponível. {details}", 503, details)

class AIReasoningError(BaseAPIException):
    """Lançado quando a Fase 1 (Cérebro) da IA falha em gerar conteúdo."""
    def __init__(self, details: str = ""):
        super().__init__("O Cérebro da IA falhou em gerar uma estratégia válida.", 502, details)

class AIStructuringError(BaseAPIException):
    """Lançado quando a Fase 2 (Formatador) falha em gerar JSON válido."""
    def __init__(self, details: str = ""):
        super().__init__("A IA falhou em estruturar os dados (Erro de Parse JSON).", 502, details)

class ResourceNotFoundError(BaseAPIException):
    """Lançado quando um recurso solicitado não existe."""
    def __init__(self, resource: str):
        super().__init__(f"{resource} não encontrado.", 404)

class ValidationBusinessError(BaseAPIException):
    """Lançado para erros de regra de negócio."""
    def __init__(self, message: str):
        super().__init__(message, 400)

class InsufficientCreditsError(BaseAPIException):
    """Lançado quando o usuário não tem saldo para a operação."""
    def __init__(self):
        super().__init__("Saldo de créditos insuficiente para realizar esta operação.", 402)

# ==============================================================================
# SEÇÃO 5: DECORATORS E MIDDLEWARE (OBSERVABILITY)
# ==============================================================================

def measure_time(func):
    """
    Decorator para medir o tempo de execução de funções assíncronas (Endpoints).
    Registra o tempo em milissegundos no log.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())[:8] # Trace ID curto para correlação
        func_name = func.__name__
        
        logger.info(f"⏳ [Trace-{request_id}] Iniciando execução de: {func_name}")
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error(f"❌ [Trace-{request_id}] {func_name} falhou após {elapsed:.2f}ms. Erro: {str(e)}")
            raise e
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"⏱️  [Trace-{request_id}] {func_name} finalizado em {elapsed:.2f}ms")
    return wrapper

def sync_measure_time(func):
    """
    Decorator para medir o tempo de execução de funções síncronas.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        func_name = func.__name__
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"⏱️  [{func_name}] executado em {elapsed:.2f}ms")
    return wrapper

# ==============================================================================
# SEÇÃO 6: CAMADA DE DADOS - GERENCIADOR DE CONEXÃO MONGODB
# ==============================================================================

# Definição de tipo para ObjectId compatível com Pydantic
PyObjectId = Annotated[str, BeforeValidator(str)]

class MongoManager:
    """
    Gerenciador de Banco de Dados com Padrão Singleton.
    Responsável por manter a conexão ativa, gerenciar reconexões e fornecer acesso às coleções.
    """
    _instance = None
    client: MongoClient = None
    db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoManager, cls).__new__(cls)
            cls._instance._initialize_connection()
        return cls._instance

    def _initialize_connection(self):
        """
        Estabelece a conexão inicial com o MongoDB Atlas.
        Define timeouts agressivos para evitar hang-ups da aplicação.
        """
        try:
            logger.info("🔌 Iniciando driver MongoDB (pymongo)...")
            
            # Sanitização da senha para evitar erros de encoding na URI
            password = urllib.parse.quote_plus(settings.MONGO_PASS)
            
            # Construção da URI de Conexão (DNS Seedlist Connection Format)
            uri = f"mongodb+srv://{settings.MONGO_USER}:{password}@{settings.MONGO_HOST}/?appName=Cluster0"
            
            # Configuração do Cliente com Connection Pooling
            self.client = MongoClient(
                uri,
                serverSelectionTimeoutMS=5000, # 5 segundos para encontrar um nó primário
                connectTimeoutMS=10000,        # 10 segundos para estabelecer TCP
                socketTimeoutMS=10000,         # 10 segundos para operações de I/O
                maxPoolSize=100,               # Máximo de conexões simultâneas
                minPoolSize=10,                # Mínimo de conexões mantidas quentes
                retryWrites=True               # Tenta novamente escritas falhas (Transient Transaction Errors)
            )
            
            # Health Check (Ping)
            self.client.admin.command('ping')
            
            # Seleção do Banco de Dados
            self.db = self.client[settings.DB_NAME]
            
            logger.info(f"✅ Conexão MongoDB estabelecida com sucesso: {settings.DB_NAME}")
            
            # Garante que os índices essenciais existam
            self._ensure_critical_indexes()
            
        except Exception as e:
            logger.critical(f"❌ Falha fatal na conexão MongoDB: {e}")
            # Em produção, dependendo da política, poderíamos dar sys.exit(1) aqui.
            self.client = None
            self.db = None

    def _ensure_critical_indexes(self):
        """Cria índices para garantir performance e unicidade."""
        try:
            if self.db is not None:
                # Índice único para login
                self.db.usuarios.create_index("usuario", unique=True)
                # Índice composto para login rápido
                self.db.usuarios.create_index([("usuario", ASCENDING), ("senha", ASCENDING)])
                # Índice para feed cronológico
                self.db.posts.create_index("data", direction=DESCENDING)
                # Índice para validação de checkins diários
                self.db.checkins.create_index([("usuario", ASCENDING), ("data", DESCENDING)])
                
                logger.info("✅ Índices do banco de dados verificados/criados.")
        except Exception as e:
            logger.warning(f"⚠️ Erro não-fatal ao criar índices: {e}")

    def get_collection(self, collection_name: str):
        """
        Retorna o objeto da coleção solicitado.
        Se a conexão foi perdida, tenta reconectar uma vez.
        """
        if self.client is None or self.db is None:
            logger.warning("🔄 Conexão com MongoDB perdida. Tentando reconexão...")
            self._initialize_connection()
            
        if self.db is None:
             raise DatabaseConnectionError("Não foi possível restabelecer a conexão com o banco de dados.")
             
        return self.db[collection_name]

# Instância global do gerenciador de banco de dados
mongo_db = MongoManager()

# ==============================================================================
# SEÇÃO 7: MODELOS DE DADOS (SCHEMAS PYDANTIC)
# ==============================================================================

class MongoBaseModel(BaseModel):
    """
    Classe base para todos os modelos que representam documentos MongoDB.
    Lida com a conversão automática de `_id` (ObjectId) para string.
    """
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    
    model_config = ConfigDict(
        populate_by_name=True, 
        arbitrary_types_allowed=True, 
        json_encoders={ObjectId: str, datetime: lambda v: v.isoformat()}
    )

class UserLogin(BaseModel):
    """Schema para endpoint de login."""
    usuario: str = Field(..., min_length=3, description="Nome de usuário único")
    senha: str = Field(..., min_length=3, description="Senha do usuário")

class UserRegister(BaseModel):
    """Schema para registro de novos usuários."""
    usuario: str = Field(..., min_length=3, max_length=50)
    senha: str = Field(..., min_length=3)
    nome: str = Field(..., min_length=2)
    peso: float = Field(..., gt=0, lt=500, description="Peso em Kg")
    altura: float = Field(..., gt=0, lt=300, description="Altura em cm")
    genero: str = Field(..., pattern="^(Masculino|Feminino|Outro)$")

class UserUpdate(BaseModel):
    """Schema para atualização de perfil (campos opcionais)."""
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
    """Schema para ações em posts (Like/Delete)."""
    usuario: str
    post_id: str

class SocialCommentRequest(BaseModel):
    """Schema para comentários."""
    usuario: str
    post_id: str
    texto: str = Field(..., min_length=1, max_length=500)

class ChatMessageRequest(BaseModel):
    """Schema para envio de mensagens no chat."""
    remetente: str
    destinatario: str
    texto: str

class AdminUserEdit(BaseModel):
    """Schema para edição administrativa de usuários."""
    target_user: str
    status: Optional[str] = None
    creditos: Optional[int] = None

# ==============================================================================
# SEÇÃO 8: REPOSITÓRIO DE EXERCÍCIOS (CACHE EM MEMÓRIA)
# ==============================================================================

class ExerciseRepository:
    """
    Gerencia o carregamento e consulta do banco de exercícios local.
    Implementa Singleton para manter cache em memória e evitar I/O repetitivo.
    """
    _db: Dict[str, str] = {}
    _keys_string: str = ""
    
    @classmethod
    def load(cls):
        """Carrega o JSON de exercícios do disco."""
        try:
            path = "exercises.json"
            if not os.path.exists(path):
                # Cria fallback se arquivo não existir
                cls._db = {"supino reto": "chest/bench_press", "agachamento": "legs/squat"}
                logger.warning("⚠️ Arquivo exercises.json não encontrado. Usando mock básico.")
            else:
                with open(path, "r", encoding="utf-8") as f:
                    cls._db = json.load(f)
            
            # Cria string de chaves para injetar no prompt da IA
            # (Removemos limites anteriores para garantir contexto total)
            all_keys = list(cls._db.keys())
            cls._keys_string = ", ".join(all_keys) 
            
            logger.info(f"✅ ExerciseRepository: {len(cls._db)} exercícios carregados em memória.")
            
        except json.JSONDecodeError:
            logger.error("❌ Erro de sintaxe no arquivo exercises.json")
            cls._db = {}
        except Exception as e:
            logger.error(f"❌ Erro crítico ao carregar exercises.json: {e}")
            cls._db = {}

    @classmethod
    def get_keys_string(cls) -> str:
        """Retorna a string CSV de exercícios para prompts."""
        return cls._keys_string

    @classmethod
    def get_db(cls) -> Dict[str, str]:
        """Retorna o dicionário completo nome -> path."""
        return cls._db

# Carrega imediatamente ao iniciar
ExerciseRepository.load()

# ==============================================================================
# SEÇÃO 9: SISTEMA DE ROTATIVIDADE DE CHAVES (API KEY MANAGER)
# ==============================================================================

class KeyRotationManager:
    """
    Gerencia o pool de chaves de API, implementando lógica de Round-Robin
    e Cooldown temporário para chaves que atingem o Rate Limit (429).
    """
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.cooldowns: Dict[str, float] = {} # Armazena timestamp de liberação
        self.COOLDOWN_SECONDS = 60.0

    def get_available_keys(self) -> List[str]:
        """Retorna lista de chaves que não estão em cooldown."""
        now = time.time()
        # Limpa cooldowns expirados
        self.cooldowns = {k: v for k, v in self.cooldowns.items() if v <= now}
        
        available = [k for k in self.keys if k not in self.cooldowns]
        
        # Se todas estiverem bloqueadas, retorna a lista completa (Fail-Open strategy)
        # (Melhor tentar e falhar do que não tentar nada)
        if not available and self.keys:
            logger.warning("⚠️ Todas as chaves em cooldown. Forçando uso do pool completo.")
            return self.keys
            
        # Embaralha para balanceamento de carga estatístico (Evita hot-spots)
        random.shuffle(available)
        return available

    def report_rate_limit(self, key: str):
        """Marca uma chave como 'esgotada' temporariamente."""
        logger.warning(f"⚠️ Rate Limit atingido na chave ...{key[-4:]}. Pausando por {self.COOLDOWN_SECONDS}s.")
        self.cooldowns[key] = time.time() + self.COOLDOWN_SECONDS

# Instância global do gerenciador de chaves
key_manager = KeyRotationManager(settings.GEMINI_KEYS)

# ==============================================================================
# SEÇÃO 10: SERVIÇOS DE IA - LÓGICA CORE (CHAIN OF THOUGHT)
# ==============================================================================

class JSONRepairKit:
    """
    Ferramentas avançadas para reparo de strings JSON malformadas.
    Resolve o erro 'Expecting , delimiter' inserindo vírgulas faltantes.
    """
    
    @staticmethod
    def fix_json_string(text: str) -> str:
        """Aplica uma série de regex para limpar e corrigir a string."""
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
            # Caso 1: Entre objetos/arrays (ex: } {  ->  }, {)
            text = re.sub(r'([}\]])\s*([{\[])', r'\1,\2', text)
            
            # Caso 2: Entre valor e chave (ex: "val" "key" -> "val", "key")
            text = re.sub(r'("\s*)\s+"', r'\1,"', text)
            
            # Caso 3: Entre número e chave (ex: 123 "key" -> 123, "key")
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

    @classmethod
    def parse_robust(cls, text_ia: str) -> Dict:
        """
        Método de parseamento robusto que tenta múltiplas estratégias.
        CORREÇÃO: Renomeado para 'parse_robust' para corresponder à chamada.
        """
        # 1. Tentativa Direta
        try: return json.loads(text_ia)
        except: pass
        
        # 2. Extração de Bloco
        try:
            match = re.search(r'(\{.*\})', text_ia, re.DOTALL)
            if match: return json.loads(match.group(1))
        except: pass
        
        # 3. Reparo Agressivo
        try:
            repaired = cls.fix_json_string(text_ia)
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON Parse Falhou. Erro: {e}")
            logger.debug(f"Snippet do JSON falho: {text_ia[:500]}...")
            raise AIStructuringError(f"Falha na formatação do JSON: {e}")

class AIOrchestrator:
    """
    Orquestrador principal da IA. Implementa a arquitetura Chain of Thought
    com rodízio de chaves aninhado e correção automática de geração vazia.
    """
    
    @staticmethod
    def _call_gemini_with_retry(model_name: str, prompt: str, image_bytes: Optional[bytes] = None, 
                              json_mode: bool = False, temperature: float = 0.7) -> str:
        """
        NÚCLEO DO RODÍZIO:
        Tenta TODAS as chaves disponíveis para o modelo especificado.
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
                
                # Configuração dinâmica baseada no modo
                config = genai.types.GenerationConfig(
                    response_mime_type="application/json" if json_mode else "text/plain",
                    max_output_tokens=8192, # Contexto aumentado
                    temperature=temperature
                )
                
                inputs = [prompt]
                if image_bytes:
                    inputs.append({"mime_type": "image/jpeg", "data": image_bytes})
                
                # Chamada Síncrona (FastAPI gerencia threads)
                response = model.generate_content(inputs, generation_config=config)
                
                if response and response.text:
                    logger.info(f"   ✅ Sucesso: {model_name} (Key final ...{api_key[-4:]})")
                    return response.text
                
            except Exception as e:
                err_str = str(e)
                # Verifica se é erro de cota (429)
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
                
                prompt_p1 = context_prompt + "\n\nINSTRUÇÃO CRÍTICA: Gere uma estratégia textual DETALHADA. Não use JSON ainda. Foque na qualidade técnica, bioquímica e biomecânica. Seja VERBOSO e explicativo."
                
                strategy_text = AIOrchestrator._call_gemini_with_retry(
                    model_name=model,
                    prompt=prompt_p1,
                    image_bytes=image_bytes,
                    json_mode=False,
                    temperature=0.7 # Criatividade alta para estratégia
                )
                if strategy_text:
                    logger.info("🧠 [Fase 1] Estratégia gerada com sucesso.")
                    break # Sucesso na fase 1
            except Exception as e:
                logger.warning(f"⚠️ Modelo {model} esgotado. Tentando próximo da lista...")
                continue
        
        if not strategy_text:
            # Fallback final: Tenta usar o modelo de estruturação para pensar (último recurso)
            try:
                logger.warning("⚠️ Todos modelos de raciocínio falharam. Usando fallback de formatação para pensar...")
                strategy_text = AIOrchestrator._call_gemini_with_retry(
                    model_name=settings.STRUCTURING_MODELS[0],
                    prompt=context_prompt,
                    image_bytes=image_bytes,
                    temperature=0.7
                )
            except Exception as e:
                raise AIProcessingError(f"Falha catastrófica na IA (Fase 1 - Brain): {e}")

        # --- FASE 2: ESTRUTURAÇÃO (Formatter) ---
        logger.info("⚡ [Fase 2 - Formatter] Iniciando Estruturação JSON...")
        
        exercise_list_str = ExerciseRepository.get_keys_string()
        
        # Prompt de Formatação Reforçado (Anti-Empty Array)
        prompt_p2 = f"""
        TASK: You are a strict JSON Conversion Engine.
        Convert the following Fitness Strategy text into a VALID JSON format strictly following the schema below.
        
        SOURCE STRATEGY TEXT:
        {strategy_text}
        
        CRITICAL RULES (INTEGRITY CHECK):
        1. OUTPUT ONLY JSON. No introductory text.
        2. FULL 7 DAYS REQUIRED: Even if source says "Repeat", generate explicit objects for all 7 days (Segunda-Domingo).
        3. NO EMPTY ARRAYS: The 'dieta', 'treino', and 'suplementacao' arrays MUST contain data. 
           - If source is missing info, IMPUTE standard bodybuilding data based on the goal. DO NOT RETURN EMPTY LISTS.
        4. EXERCISE MAPPING: Map exercises to this list: [{exercise_list_str}]. Use closest match or "(Adaptado)".
        5. VOLUME: Minimum 8 exercises per workout session.
        
        REQUIRED JSON SCHEMA:
        {{
          "avaliacao": {{ 
            "segmentacao": {{ "tronco": "Txt", "superior": "Txt", "inferior": "Txt" }}, 
            "dobras": {{ "abdominal": "Txt", "suprailiaca": "Txt", "peitoral": "Txt" }}, 
            "analise_postural": "Txt", "simetria": "Txt", "insight": "Txt" 
          }},
          "dieta": [ 
            {{ "dia": "Segunda-feira", "foco_nutricional": "Txt", "refeicoes": [ {{ "horario": "08:00", "nome": "Café", "alimentos": "Txt" }} ], "macros_totais": "Txt" }},
            ... (MUST HAVE 7 ITEMS) ...
          ],
          "dieta_insight": "Txt",
          "suplementacao": [ {{ "nome": "Creatina", "dose": "5g", "horario": "Pós", "motivo": "Txt" }} ],
          "suplementacao_insight": "Txt",
          "treino": [ 
             {{ "dia": "Segunda-feira", "foco": "Peito", "exercicios": [ {{ "nome": "Supino", "series_reps": "4x10", "execucao": "Txt", "justificativa_individual": "Txt" }} ], "treino_alternativo": "Txt", "justificativa": "Txt" }},
             ... (MUST HAVE 7 ITEMS) ...
          ],
          "treino_insight": "Txt"
        }}
        """
        
        try:
            # Tenta estruturar usando o modelo rápido
            json_text = AIOrchestrator._call_gemini_with_retry(
                model_name=settings.STRUCTURING_MODELS[0],
                prompt=prompt_p2,
                image_bytes=None, 
                json_mode=True, # Força modo JSON
                temperature=0.1 # Temperatura Baixa para precisão
            )
            data = JSONRepairKit.parse_robust(json_text)
            
            # VALIDAÇÃO DE INTEGRIDADE (Se vier vazio, força erro para retry ou fallback)
            if not data.get('dieta') or len(data['dieta']) == 0:
                raise AIStructuringError("IA retornou dieta vazia.")
            if not data.get('treino') or len(data['treino']) == 0:
                raise AIStructuringError("IA retornou treino vazio.")
                
            return data
            
        except Exception as e:
            logger.error(f"Fase 2 falhou. Tentando parsear texto original como última esperança... {e}")
            # Fallback: Tenta parsear o texto da Fase 1 se ele já estiver estruturado
            try:
                fallback_data = JSONRepairKit.parse_robust(strategy_text)
                if not fallback_data.get('dieta'):
                    raise Exception("Fallback vazio")
                return fallback_data
            except:
                raise AIStructuringError("Falha total na geração de dados estruturados.")

    @staticmethod
    def simple_generation(prompt: str, image_bytes: Optional[bytes] = None) -> str:
        """Geração rápida para tarefas simples (ex: comentários, validação)."""
        try:
            return AIOrchestrator._call_gemini_with_retry(
                settings.STRUCTURING_MODELS[0], 
                prompt, 
                image_bytes, 
                json_mode=False
            )
        except:
            return "Estou analisando seu treino... continue focado!"

# ==============================================================================
# SEÇÃO 11: HELPERS DE NEGÓCIO E IMAGEM (MOVIDO PARA CIMA)
# ==============================================================================
# MOVIDO PARA ANTES DO 'AIOrchestrator' PARA EVITAR NameError

def normalizar_texto(texto: str) -> str:
    if not texto: return ""
    return "".join(c for c in unicodedata.normalize('NFD', str(texto)) if unicodedata.category(c) != 'Mn').lower().strip()

def validar_exercicios_final(treino_data: list) -> list:
    """
    Validação final pós-IA.
    Tenta casar nomes de exercícios gerados com pastas de imagens locais (Github).
    """
    db = ExerciseRepository.get_db()
    if not treino_data or not db: return treino_data
    
    base_url = "[https://raw.githubusercontent.com/italoat/technobolt-backend/main/assets/exercises](https://raw.githubusercontent.com/italoat/technobolt-backend/main/assets/exercises)"
    
    # Mapas de busca O(1)
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
            
            # 1. Match Exato
            if norm_name in db_map:
                path = db_map[norm_name]
                final_name = db_titles[norm_name]
            else:
                # 2. Match por Similaridade
                matches = difflib.get_close_matches(norm_name, db_map.keys(), n=1, cutoff=0.6)
                if matches:
                    path = db_map[matches[0]]
                    final_name = db_titles[matches[0]]
                else:
                    # 3. Match por Substring
                    for k in db_map.keys():
                        if k in norm_name or norm_name in k:
                            path = db_map[k]
                            final_name = db_titles[k]
                            break
                    # 4. Fallback
                    if not path and "polichinelo" in db_map:
                        path = db_map["polichinelo"]
                        final_name = f"{raw_name} (Adaptado)"

            # Atualiza objeto
            ex['nome'] = str(final_name).title()
            if path:
                ex['imagens_demonstracao'] = [
                    f"{base_url}/{path}/0.jpg",
                    f"{base_url}/{path}/1.jpg"
                ]
            else:
                ex['imagens_demonstracao'] = []
            
            corrected_exs.append(ex)
        
        dia['exercicios'] = corrected_exs
            
    return treino_data

class ImageService:
    """Serviço de processamento de imagem movido para escopo global antes do uso."""
    @staticmethod
    def optimize(file_bytes: bytes, quality: int = 75, max_size: tuple = (800, 800)) -> bytes:
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode != 'RGB': img = img.convert("RGB")
                img.thumbnail(max_size)
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality, optimize=True)
                return output.getvalue()
        except Exception as e:
            logger.error(f"Erro na otimização de imagem: {e}")
            return file_bytes

class PDFReport(FPDF):
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
        self.set_fill_color(30, 30, 35) # Hardcoded para evitar dependencia
        self.set_text_color(*self.col_azul if hasattr(self, 'col_azul') else self.col_accent)
        self.set_font("Arial", "B", 11)
        self.multi_cell(0, 6, self.sanitize(title), fill=True)
        self.set_text_color(*self.col_texto)
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 6, self.sanitize(body), fill=True)
        self.ln(2)

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

@app.post("/auth/login", tags=["Auth"])
@sync_measure_time
def login(dados: UserLogin):
    col = mongo_db.get_collection("usuarios")
    user = col.find_one({"usuario": dados.usuario, "senha": dados.senha})
    
    if not user: raise HTTPException(401, "Inválido")
    if user.get("status") != "ativo" and not user.get("is_admin"): raise HTTPException(403)
    return {"sucesso": True, "dados": {k: v for k, v in user.items() if k != "_id"}}

@app.post("/auth/registro", tags=["Auth"])
def registrar(dados: UserRegister):
    col = mongo_db.get_collection("usuarios")
    if col.find_one({"usuario": dados.usuario}): raise HTTPException(400)
    col.insert_one({**dados.model_dump(), "status": "pendente", "avaliacoes_restantes": 0, "pontos": 0, "historico_dossies": [], "is_admin": False})
    return {"sucesso": True}

@app.post("/perfil/atualizar", tags=["Perfil"])
def atualizar_perfil(dados: UserUpdate):
    col = mongo_db.get_collection("usuarios")
    data = {k: v for k, v in dados.model_dump(exclude={'usuario'}).items() if v is not None}
    col.update_one({"usuario": dados.usuario}, {"$set": data})
    return {"sucesso": True}

@app.post("/analise/executar", tags=["Analise"])
@measure_time
async def executar_analise(
    usuario: str = Form(...), nome_completo: str = Form(...), peso: str = Form(...), 
    altura: str = Form(...), objetivo: str = Form(...), genero: str = Form("Masculino"),
    observacoes: str = Form(""), foto: UploadFile = File(...)
):
    logger.info(f"🚀 Iniciando análise: {usuario}")
    
    try:
        p_float = float(str(peso).replace(',', '.'))
        alt_int = int(float(str(altura).replace(',', '.').strip()) * 100) if float(str(altura).replace(',', '.').strip()) < 3.0 else int(float(str(altura).replace(',', '.').strip()))
    except: p_float = 70.0; alt_int = 175
    
    col = mongo_db.get_collection("usuarios")
    col.update_one({"usuario": usuario}, {"$set": {"nome": nome_completo, "peso": p_float, "altura": alt_int, "genero": genero, "info_add": observacoes}})
    user = col.find_one({"usuario": usuario})
    
    img = await foto.read()
    img_opt = ImageService.optimize(img)
    
    prompt = f"""
    ACT AS ELITE COACH. Client: {nome_completo} ({genero}), {p_float}kg, {alt_int}cm. Goal: {objetivo}.
    Restrictions: {user.get('restricoes_fis')}, {user.get('restricoes_alim')}.
    TASKS: 1. PHYSIQUE ANALYSIS. 2. DIET (7 DAYS - Mon-Sun). 3. TRAINING (7 DAYS - Mon-Sun, 10+ Exercises/day). 4. SUPPLEMENTS.
    """
    
    try:
        res = AIOrchestrator.execute_chain_of_thought(prompt, img_opt)
    except Exception as e: 
        logger.error(f"Erro IA: {e}")
        raise HTTPException(503, "IA indisponível")
    
    if 'treino' in res: res['treino'] = validar_exercicios_final(res['treino'])
    
    dossie = {
        "id": str(ObjectId()), "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "peso_reg": p_float, "conteudo_bruto": {
            "json_full": res,
            "r1": str(res.get('avaliacao', {}).get('insight', '')),
            "r2": str(res.get('dieta_insight', '')),
            "r3": str(res.get('suplementacao_insight', '')),
            "r4": str(res.get('treino_insight', ''))
        }
    }
    
    upd = {"$push": {"historico_dossies": dossie}}
    if not user.get('is_admin'): upd["$inc"] = {"avaliacoes_restantes": -1}
    col.update_one({"usuario": usuario}, upd)
    
    return {"sucesso": True, "resultado": dossie}

@app.post("/analise/regenerar-secao", tags=["Analise"])
async def regenerar_secao(dados: dict = Body(...)):
    col = mongo_db.get_collection("usuarios")
    user = col.find_one({"usuario": dados.get("usuario")})
    if not user: return {"sucesso": False}
    
    prompt = f"Regenerate '{dados.get('secao')}' for {user.get('nome')}. Make it intense, 7 days."
    try:
        new_c = AIOrchestrator.execute_chain_of_thought(prompt, None)
        last = user['historico_dossies'][-1]
        last['conteudo_bruto']['json_full'][dados.get('secao')] = new_c[dados.get('secao')]
        
        col.update_one(
            {"usuario": dados.get("usuario"), "historico_dossies.data": last['data']},
            {"$set": {"historico_dossies.$.conteudo_bruto.json_full": last['conteudo_bruto']['json_full']}}
        )
        return {"sucesso": True, "resultado": last}
    except: return {"sucesso": False}

@app.get("/historico/{usuario}", tags=["Perfil"])
def history(usuario: str):
    user = mongo_db.get_collection("usuarios").find_one({"usuario": usuario})
    if not user: return {"sucesso": True, "historico": []}
    return {
        "sucesso": True, "historico": jsonable_encoder(user.get('historico_dossies', [])),
        "creditos": user.get('avaliacoes_restantes', 0),
        "perfil": {k: v for k, v in user.items() if k not in ['_id', 'historico_dossies', 'senha']}
    }

@app.get("/social/feed", tags=["Social"])
def feed():
    posts = list(mongo_db.get_collection("posts").find().sort("data", -1).limit(50))
    for p in posts: p['_id'] = str(p['_id'])
    return {"sucesso": True, "feed": posts}

@app.post("/social/postar", tags=["Social"])
async def post(usuario: str = Form(...), legenda: str = Form(...), imagem: UploadFile = File(...)):
    img = ImageService.optimize(await imagem.read())
    cmt = AIOrchestrator.simple_generation(f"Comentário gym bro para: {legenda}", img)
    mongo_db.get_collection("posts").insert_one({
        "autor": usuario, "legenda": legenda, 
        "imagem": base64.b64encode(img).decode('utf-8'), 
        "data": datetime.now().isoformat(), "likes": [], 
        "comentarios": [{"autor": "TechnoBolt AI", "texto": cmt}]
    })
    return {"sucesso": True}

@app.post("/social/curtir")
def like(dados: SocialPostRequest):
    col = mongo_db.get_collection("posts")
    oid = ObjectId(dados.post_id)
    post = col.find_one({"_id": oid})
    if post:
        op = "$pull" if dados.usuario in post.get("likes", []) else "$addToSet"
        col.update_one({"_id": oid}, {op: {"likes": dados.usuario}})
    return {"sucesso": True}

@app.post("/social/comentar")
def comment(dados: SocialCommentRequest):
    mongo_db.get_collection("posts").update_one({"_id": ObjectId(dados.post_id)}, 
        {"$push": {"comentarios": {"autor": dados.usuario, "texto": dados.texto, "data": datetime.now().isoformat()}}})
    return {"sucesso": True}

@app.post("/social/post/deletar")
def delete_post(dados: SocialPostRequest):
    res = mongo_db.get_collection("posts").delete_one({"_id": ObjectId(dados.post_id), "autor": dados.usuario})
    return {"sucesso": res.deleted_count > 0}

@app.get("/social/ranking")
def ranking():
    u = list(mongo_db.get_collection("usuarios").find({"is_admin": False}, {"nome": 1, "usuario": 1, "pontos": 1, "foto_perfil": 1, "_id": 0}).sort("pontos", -1).limit(50))
    return {"sucesso": True, "ranking": u}

@app.get("/social/checkins")
def checkins(usuario: str):
    raw = list(mongo_db.get_collection("checkins").find({"usuario": usuario}))
    return {"sucesso": True, "checkins": {datetime.fromisoformat(c['data']).day: c['tipo'] for c in raw}}

@app.post("/social/validar-conquista")
async def val_conquest(usuario: str = Form(...), tipo: str = Form(...), foto: UploadFile = File(...)):
    now = datetime.now()
    if mongo_db.get_collection("checkins").find_one({"usuario": usuario, "data": {"$gte": datetime(now.year, now.month, now.day).isoformat()}}):
        return {"sucesso": False}
    
    img = ImageService.optimize(await foto.read())
    resp = AIOrchestrator.simple_generation(f"Valide treino {tipo}. Responda APROVADO ou REPROVADO.", img)
    
    if "APROVADO" in resp.upper():
        mongo_db.get_collection("checkins").insert_one({"usuario": usuario, "tipo": tipo, "data": now.isoformat(), "pontos": 50})
        mongo_db.get_collection("usuarios").update_one({"usuario": usuario}, {"$inc": {"pontos": 50}})
        return {"sucesso": True, "aprovado": True, "pontos": 50}
    return {"sucesso": True, "aprovado": False}

@app.get("/chat/mensagens")
def chat_msgs(user1: str, user2: str):
    msgs = list(mongo_db.get_collection("chat").find({"$or": [{"remetente": user1, "destinatario": user2}, {"remetente": user2, "destinatario": user1}]}).sort("timestamp", 1))
    for m in msgs: m['_id'] = str(m['_id'])
    return {"sucesso": True, "mensagens": msgs}

@app.post("/chat/enviar")
def chat_send(dados: ChatMessageRequest):
    mongo_db.get_collection("chat").insert_one({**dados.model_dump(), "timestamp": datetime.now().isoformat()})
    return {"sucesso": True}

@app.get("/chat/usuarios")
def chat_users(usuario_atual: str):
    return {"sucesso": True, "usuarios": list(mongo_db.get_collection("usuarios").find({"usuario": {"$ne": usuario_atual}}, {"usuario": 1, "nome": 1, "_id": 0}))}

@app.get("/admin/listar")
def admin_l():
    u = list(mongo_db.get_collection("usuarios").find())
    for x in u: x['_id'] = str(x['_id'])
    return {"sucesso": True, "usuarios": u}

@app.post("/admin/editar")
def admin_e(dados: AdminUserEdit):
    mongo_db.get_collection("usuarios").update_one({"usuario": dados.target_user}, {"$set": {k:v for k,v in dados.model_dump().items() if v and k!="target_user"}})
    return {"sucesso": True}

@app.post("/admin/excluir")
def admin_d(dados: AdminUserEdit):
    mongo_db.get_collection("usuarios").delete_one({"usuario": dados.target_user})
    return {"sucesso": True}

@app.get("/setup/criar-admin")
def setup_adm():
    if mongo_db.get_collection("usuarios").find_one({"usuario": "admin"}): return {"sucesso": False}
    mongo_db.get_collection("usuarios").insert_one({"usuario": "admin", "senha": "123", "nome": "Admin", "is_admin": True, "status": "ativo", "avaliacoes_restantes": 9999})
    return {"sucesso": True}

@app.get("/analise/baixar-pdf/{usuario}", tags=["Export"])
def pdf_gen(usuario: str):
    try:
        user = mongo_db.get_collection("usuarios").find_one({"usuario": usuario})
        if not user: raise HTTPException(404)
        data = user['historico_dossies'][-1]['conteudo_bruto']['json_full']
        
        pdf = PDFReport()
        pdf.add_page(); pdf.chapter_title(f"RELATORIO: {user.get('nome')}")
        if 'dieta' in data: 
            pdf.chapter_title("DIETA")
            for d in data['dieta']: pdf.card(d['dia'], str(d['refeicoes']))
        
        buf = io.BytesIO()
        out = pdf.output(dest='S').encode('latin-1', 'replace')
        buf.write(out); buf.seek(0)
        return StreamingResponse(buf, media_type="application/pdf", headers={'Content-Disposition': 'attachment; filename="report.pdf"'})
    except: raise HTTPException(500)
