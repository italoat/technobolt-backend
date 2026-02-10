"""
TechnoBolt Gym Hub API - Enterprise Edition
Version: 107.0-Titanium-FullScope
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

# --- FRAMEWORKS EXTERNOS ---
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

# --- IA E IMAGEM ---
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter
import pillow_heif

# --- GERAÇÃO DE RELATÓRIOS ---
from fpdf import FPDF

# ==============================================================================
# SEÇÃO 1: CONFIGURAÇÃO DE LOGGING AVANÇADO
# ==============================================================================

class EnterpriseLogger:
    """Configuração de Logging estruturado para monitoramento em produção."""
    
    @staticmethod
    def setup():
        logger = logging.getLogger("TechnoBoltAPI")
        logger.setLevel(logging.INFO)
        
        # Remove handlers existentes para evitar duplicação em reloads
        if logger.hasHandlers():
            logger.handlers.clear()
            
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

logger = EnterpriseLogger.setup()

# ==============================================================================
# SEÇÃO 2: INICIALIZAÇÃO DE SUPORTE
# ==============================================================================

# Suporte nativo para imagens HEIC (iOS) para evitar erros de upload de iPhone
try:
    pillow_heif.register_heif_opener()
    logger.info("✅ Suporte a HEIC/HEIF inicializado com sucesso.")
except Exception as e:
    logger.warning(f"⚠️ Falha ao inicializar suporte HEIC: {e}")

# ==============================================================================
# SEÇÃO 3: GERENCIAMENTO DE CONFIGURAÇÃO E AMBIENTE
# ==============================================================================

class Settings:
    """
    Gerenciador de Configurações Singleton.
    Valida e carrega variáveis de ambiente críticas na inicialização.
    """
    def __init__(self):
        logger.info("⚙️ Carregando configurações do ambiente...")
        
        # Banco de Dados
        self.MONGO_USER = self._get_env("MONGO_USER", "technobolt")
        self.MONGO_PASS = self._get_env("MONGO_PASS", "tech@132")
        self.MONGO_HOST = self._get_env("MONGO_HOST", "cluster0.zbjsvk6.mongodb.net")
        self.DB_NAME = self._get_env("DB_NAME", "technoboltgym")
        
        # Metadados da API
        self.API_TITLE = "TechnoBolt Gym Hub API"
        self.API_VERSION = "107.0-Titanium-FullScope"
        self.ENV = self._get_env("ENV", "production")
        
        # Carregamento dinâmico de chaves de API (Load Balancer)
        self.GEMINI_KEYS = self._load_api_keys()
        
        # Definição de Motores (Estratégia Hierárquica)
        # Brain (Raciocínio): Prioriza inteligência e contexto
        self.REASONING_MODELS = [
            "models/gemini-3-flash-preview", 
            "models/gemini-2.5-flash", 
            "models/gemini-2.0-flash"
        ]
        # Formatter (Estruturação): Prioriza velocidade e aderência a JSON
        self.STRUCTURING_MODELS = [
            "models/gemini-flash-latest"
        ]

    def _get_env(self, key: str, default: Any = None) -> str:
        value = os.environ.get(key, default)
        if value is None:
            logger.warning(f"⚠️ Variável de ambiente {key} não definida.")
        return value

    def _load_api_keys(self) -> List[str]:
        keys = []
        # Varre até 20 slots de chaves para garantir redundância e escalabilidade
        for i in range(1, 21):
            key_val = os.environ.get(f"GEMINI_CHAVE_{i}")
            if key_val and len(key_val.strip()) > 10:
                keys.append(key_val.strip())
        
        if not keys:
            logger.critical("❌ ERRO CRÍTICO: Nenhuma chave de API (GEMINI_CHAVE_x) encontrada!")
        else:
            logger.info(f"🔑 {len(keys)} chaves de API do Gemini carregadas no pool.")
        
        return keys

# Instância global de configurações
settings = Settings()

# ==============================================================================
# SEÇÃO 4: EXCEÇÕES CUSTOMIZADAS
# ==============================================================================

class BaseAPIException(Exception):
    """Classe base para erros da API com status code associado."""
    def __init__(self, message: str, status_code: int = 500, details: Any = None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)

class DatabaseConnectionError(BaseAPIException):
    def __init__(self, details: str):
        super().__init__("Erro de conexão com o banco de dados.", 503, details)

class AIProcessingError(BaseAPIException):
    def __init__(self, details: str):
        super().__init__("Erro no processamento de IA.", 503, details)

class AIContextLengthError(BaseAPIException):
    def __init__(self):
        super().__init__("Contexto da IA excedido.", 413)

# ==============================================================================
# SEÇÃO 5: DECORATORS E MIDDLEWARE
# ==============================================================================

def measure_time(func):
    """Decorator para medir tempo de execução de funções assíncronas."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"⏱️  {func.__name__} executado em {elapsed:.2f}ms")
    return wrapper

def sync_measure_time(func):
    """Decorator para medir tempo de execução de funções síncronas."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"⏱️  {func.__name__} executado em {elapsed:.2f}ms")
    return wrapper

# ==============================================================================
# SEÇÃO 6: CAMADA DE PERSISTÊNCIA (MONGODB)
# ==============================================================================

PyObjectId = Annotated[str, BeforeValidator(str)]

class MongoManager:
    """
    Gerenciador de Conexão MongoDB com padrão Singleton e Reconexão Automática.
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
        """Inicializa a conexão com parâmetros de tuning para produção."""
        try:
            logger.info("🔌 Inicializando driver MongoDB...")
            password = urllib.parse.quote_plus(settings.MONGO_PASS)
            uri = f"mongodb+srv://{settings.MONGO_USER}:{password}@{settings.MONGO_HOST}/?appName=Cluster0"
            
            self.client = MongoClient(
                uri,
                serverSelectionTimeoutMS=5000, # 5s timeout para seleção de servidor
                connectTimeoutMS=10000,        # 10s timeout de conexão
                socketTimeoutMS=10000,         # 10s timeout de socket
                maxPoolSize=100,               # Pool de conexões para alta concorrência
                minPoolSize=10,
                retryWrites=True
            )
            
            self.client.admin.command('ping')
            self.db = self.client[settings.DB_NAME]
            logger.info(f"✅ Conectado ao MongoDB: {settings.DB_NAME}")
            
        except Exception as e:
            logger.critical(f"❌ Falha fatal na conexão MongoDB: {e}")

    def get_collection(self, collection_name: str):
        """Retorna uma coleção, forçando reconexão se necessário."""
        if self.client is None or self.db is None:
            logger.warning("🔄 Tentando reconexão com MongoDB...")
            self._initialize()
            
        if self.db is None:
             raise DatabaseConnectionError("Banco de dados indisponível.")
             
        return self.db[collection_name]

# Instância global do banco
mongo_db = MongoManager()

# ==============================================================================
# SEÇÃO 7: MODELOS DE DADOS (SCHEMAS PYDANTIC)
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
# SEÇÃO 8: REPOSITÓRIO DE EXERCÍCIOS (CACHE)
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
        try:
            path = "exercises.json"
            if not os.path.exists(path):
                logger.warning("⚠️ Arquivo exercises.json não encontrado. Validação de exercícios será ignorada.")
                return

            with open(path, "r", encoding="utf-8") as f:
                cls._db = json.load(f)
                
            # Prepara string para prompt da IA
            # Limitamos a 600 exercícios para não estourar contexto
            all_keys = list(cls._db.keys())
            cls._keys_string = ", ".join(all_keys[:600])
            
            logger.info(f"✅ ExerciseRepository: {len(cls._db)} exercícios carregados em memória.")
            
        except json.JSONDecodeError:
            logger.error("❌ Erro de sintaxe no arquivo exercises.json")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar exercises.json: {e}")

    @classmethod
    def get_keys_string(cls) -> str:
        return cls._keys_string

    @classmethod
    def get_db(cls) -> Dict[str, str]:
        return cls._db

# Carrega no startup
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
        self.cooldowns = {k: v for k, v in self.cooldowns.items() if v > now}
        
        available = [k for k in self.keys if k not in self.cooldowns]
        
        # Se não houver chaves, tenta usar todas (desespero)
        if not available and self.keys:
            logger.warning("⚠️ Todas as chaves em cooldown. Forçando uso do pool completo.")
            return self.keys
            
        # Embaralha para balanceamento de carga estatístico
        random.shuffle(available)
        return available

    def report_rate_limit(self, key: str):
        """Marca uma chave como 'esgotada' temporariamente."""
        logger.warning(f"⚠️ Rate Limit atingido na chave ...{key[-4:]}. Pausando por {self.COOLDOWN_SECONDS}s.")
        self.cooldowns[key] = time.time() + self.COOLDOWN_SECONDS

key_manager = KeyRotationManager(settings.GEMINI_KEYS)

# ==============================================================================
# SEÇÃO 10: SERVIÇOS DE IA - LÓGICA CORE (CHAIN OF THOUGHT)
# ==============================================================================

class JSONRepairKit:
    """
    Ferramentas avançadas para reparo de strings JSON malformadas.
    Resolve problemas de sintaxe comuns em LLMs.
    """
    @staticmethod
    def extract_json_block(text: str) -> str:
        """Tenta encontrar o bloco JSON principal usando Regex."""
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            return match.group(1)
        return text

    @staticmethod
    def fix_syntax(text: str) -> str:
        """Corrige erros sintáticos comuns."""
        text = text.strip()
        
        # Remove Markdown
        if "```" in text:
            text = re.sub(r'```json|```', '', text).strip()
            
        # Remove comentários estilo C/JS
        text = re.sub(r'//.*?\n|/\*.*?\*/', '', text, flags=re.S)
        
        # Remove vírgulas trailing (Ex: {"a": 1,})
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Balanceamento de chaves (JSON Truncado)
        open_braces = text.count('{')
        close_braces = text.count('}')
        if open_braces > close_braces:
            text += '}' * (open_braces - close_braces)
            
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        if open_brackets > close_brackets:
            text += ']' * (open_brackets - close_brackets)
            
        return text

    @classmethod
    def parse_robust(cls, text_ia: str) -> Dict:
        """Pipeline de tentativa de parseamento."""
        # 1. Tentativa Direta
        try:
            return json.loads(text_ia)
        except: pass
        
        # 2. Extração de Bloco
        extracted = cls.extract_json_block(text_ia)
        try:
            return json.loads(extracted)
        except: pass
        
        # 3. Reparo Agressivo
        fixed = cls.fix_syntax(extracted)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON Irreparável. Erro: {e}")
            logger.debug(f"Snippet: {text_ia[:200]}...")
            raise AIProcessingError("Falha na estruturação do JSON pela IA.")

class AIOrchestrator:
    """
    Orquestrador principal da IA. Implementa a arquitetura Chain of Thought
    com rodízio de chaves aninhado (Nested Loop).
    """
    
    @staticmethod
    def _call_gemini_with_retry(model_name: str, prompt: str, image_bytes: Optional[bytes] = None, 
                              json_mode: bool = False, temperature: float = 0.7) -> str:
        """
        NÚCLEO DO RODÍZIO:
        Tenta TODAS as chaves disponíveis para o modelo especificado.
        Só levanta erro se todas as chaves falharem.
        """
        keys = key_manager.get_available_keys()
        if not keys:
            raise AIProcessingError("Sem chaves de API disponíveis.")
            
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
                    logger.info(f"   ✅ Sucesso: {model_name} (Key ...{api_key[-4:]})")
                    return response.text
                
            except Exception as e:
                err_str = str(e)
                # Verifica se é erro de cota
                if "429" in err_str or "Resource exhausted" in err_str:
                    key_manager.report_rate_limit(api_key)
                
                logger.warning(f"   ⚠️ Falha: {model_name} (Key ...{api_key[-4:]}): {err_str[:80]}")
                last_error = e
                continue # Tenta próxima chave
                
        # Se saiu do loop, falhou com todas as chaves para este modelo
        raise last_error if last_error else Exception(f"Falha total no modelo {model_name}")

    @staticmethod
    def execute_chain_of_thought(context_prompt: str, image_bytes: Optional[bytes]) -> Dict:
        """
        Pipeline Principal:
        1. Tenta modelos de RACIOCÍNIO (Fase 1).
        2. Envia resultado para modelos de ESTRUTURAÇÃO (Fase 2).
        """
        
        # --- FASE 1: RACIOCÍNIO (Brain) ---
        strategy_text = None
        
        # Tenta cada modelo de raciocínio na ordem de preferência
        # Cada modelo tentará todas as chaves disponíveis antes de passar para o próximo
        for model in settings.REASONING_MODELS:
            try:
                logger.info(f"🧠 [Fase 1] Iniciando Raciocínio com {model}...")
                
                prompt_p1 = context_prompt + "\n\nINSTRUÇÃO CRÍTICA: Gere uma estratégia textual DETALHADA. Não use JSON ainda. Foque na qualidade técnica, bioquímica e biomecânica. Seja detalhista."
                
                strategy_text = AIOrchestrator._call_gemini_with_retry(
                    model_name=model,
                    prompt=prompt_p1,
                    image_bytes=image_bytes,
                    json_mode=False,
                    temperature=0.7 # Criatividade alta para estratégia
                )
                if strategy_text:
                    break # Sucesso na fase 1
            except Exception as e:
                logger.warning(f"⚠️ Modelo {model} esgotado. Tentando próximo da lista...")
                continue
        
        if not strategy_text:
            # Fallback final: Tenta usar o modelo de estruturação para pensar (melhor que nada)
            try:
                logger.warning("⚠️ Todos modelos de raciocínio falharam. Usando fallback...")
                strategy_text = AIOrchestrator._call_gemini_with_retry(
                    model_name=settings.STRUCTURING_MODELS[0],
                    prompt=context_prompt,
                    image_bytes=image_bytes,
                    temperature=0.7
                )
            except Exception as e:
                raise AIReasoningError(f"Falha total na IA: {e}")

        # --- FASE 2: ESTRUTURAÇÃO (Formatter) ---
        exercise_list = ExerciseRepository.get_keys_string()
        
        prompt_p2 = f"""
        TASK: You are a strict Data Parsing Assistant.
        Convert the following Fitness Strategy into a VALID JSON format strictly following the schema below.
        
        SOURCE STRATEGY:
        {strategy_text}
        
        RULES:
        1. OUTPUT ONLY JSON. No text before/after.
        2. VALIDATE EXERCISE NAMES: You must map the exercises in the strategy to this database list: [{exercise_list}].
           - If the strategy mentions an exercise NOT in the list, verify if it is similar to one in the list and use the list name.
           - If no match found, use the closest logical match or keep the name but mark as "(Adaptado)".
        3. ENSURE COMPLETENESS:
           - Diet: 7 Days (Segunda to Domingo).
           - Workout: 7 Days (Segunda to Domingo).
           - Volume: Ensure at least 10 exercises per workout day.
        
        REQUIRED JSON SCHEMA:
        {{
          "avaliacao": {{ 
            "segmentacao": {{ "tronco": "...", "superior": "...", "inferior": "..." }}, 
            "dobras": {{ "abdominal": "...", "suprailiaca": "...", "peitoral": "..." }}, 
            "analise_postural": "...", 
            "simetria": "...", 
            "insight": "..." 
          }},
          "dieta": [ 
            {{ "dia": "Segunda-feira", "foco_nutricional": "...", "refeicoes": [ {{ "horario": "...", "nome": "...", "alimentos": "..." }} ], "macros_totais": "..." }},
            ... (Repeat for all 7 days) ...
          ],
          "dieta_insight": "...",
          "suplementacao": [ {{ "nome": "...", "dose": "...", "horario": "...", "motivo": "..." }} ],
          "suplementacao_insight": "...",
          "treino": [ 
             {{ "dia": "Segunda-feira", "foco": "...", "exercicios": [ {{ "nome": "...", "series_reps": "...", "execucao": "...", "justificativa_individual": "..." }} ], "treino_alternativo": "...", "justificativa": "..." }},
             ... (Repeat for all 7 days) ...
          ],
          "treino_insight": "..."
        }}
        """
        
        # Tenta modelos de formatação (geralmente o flash)
        for model in settings.STRUCTURING_MODELS:
            try:
                logger.info(f"⚡ [Fase 2] Estruturando com {model}...")
                json_text = AIOrchestrator._call_gemini_with_retry(
                    model_name=model,
                    prompt=prompt_p2,
                    image_bytes=None,
                    json_mode=True, # Força modo JSON
                    temperature=0.1 # Precisão máxima
                )
                return JSONRepairKit.parse_robust(json_text)
            except Exception as e:
                logger.warning(f"⚠️ Erro de formatação com {model}: {e}")
                continue
                
        # Se a formatação via IA falhar, tenta parsear o texto original da fase 1 se ele parecer JSON
        try:
            return JSONRepairKit.parse_robust(strategy_text)
        except:
            raise AIStructuringError("Não foi possível gerar um JSON válido após todas as tentativas.")

    @staticmethod
    def simple_generation(prompt: str, image_bytes: Optional[bytes] = None) -> str:
        """Geração rápida para tarefas simples (ex: comentários)."""
        try:
            # Usa o modelo mais rápido da lista de estruturação
            return AIOrchestrator._call_gemini_with_retry(
                settings.STRUCTURING_MODELS[0], 
                prompt, 
                image_bytes, 
                json_mode=False
            )
        except:
            return "Estou analisando seu treino... continue focado!"

class ImageService:
    """Utilitários para processamento e otimização de imagens."""
    
    @staticmethod
    def optimize(file_bytes: bytes, quality: int = 75, max_size: tuple = (800, 800)) -> bytes:
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                # Corrige orientação EXIF
                img = ImageOps.exif_transpose(img)
                # Converte para RGB
                if img.mode != 'RGB':
                    img = img.convert("RGB")
                # Resize inteligente
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
        # Garante compatibilidade Latin-1 para FPDF
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
# SEÇÃO 12: HELPERS DE NEGÓCIO
# ==============================================================================

def normalizar_texto(texto: str) -> str:
    if not texto: return ""
    return "".join(c for c in unicodedata.normalize('NFD', str(texto)) if unicodedata.category(c) != 'Mn').lower().strip()

def validar_exercicios_final(treino_data: list) -> list:
    """
    Validação final pós-IA.
    Tenta casar nomes de exercícios gerados com pastas de imagens locais.
    """
    if not treino_data or not EXERCISE_DB: return treino_data
    
    base_url = "[https://raw.githubusercontent.com/italoat/technobolt-backend/main/assets/exercises](https://raw.githubusercontent.com/italoat/technobolt-backend/main/assets/exercises)"
    
    # Mapas de busca O(1)
    db_map = {normalizar_texto(k): v for k, v in EXERCISE_DB.items()}
    db_titles = {normalizar_texto(k): k for k, v in EXERCISE_DB.items()}

    for dia in treino_data:
        if 'exercicios' not in dia: continue
        
        corrected_exs = []
        for ex in dia['exercicios']:
            raw_name = ex.get('nome', 'Exercício')
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

def calcular_medalha(username: str) -> str:
    # Lógica simplificada de gamification
    try:
        user = mongo_db.get_collection("usuarios").find_one({"usuario": username})
        return "🥇" if user and user.get('pontos', 0) > 1000 else ""
    except: return ""

# ==============================================================================
# SEÇÃO 13: APLICAÇÃO FASTAPI & ROTAS
# ==============================================================================

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Backend Enterprise da TechnoBolt. Arquitetura Chain-of-Thought (CoT)."
)

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
    
    if not user:
        raise HTTPException(401, "Credenciais inválidas")
    
    if user.get("status") != "ativo" and not user.get("is_admin"):
        raise HTTPException(403, "Conta pendente de aprovação.")
    
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
    if col.find_one({"usuario": dados.usuario}):
        raise HTTPException(400, "Usuário já existe")
    
    new_user = dados.model_dump()
    new_user.update({
        "status": "pendente",
        "avaliacoes_restantes": 0,
        "pontos": 0,
        "historico_dossies": [],
        "is_admin": False,
        "created_at": datetime.now()
    })
    col.insert_one(new_user)
    return {"sucesso": True, "mensagem": "Registro realizado."}

@app.post("/perfil/atualizar", tags=["Perfil"])
def atualizar_perfil(dados: UserUpdate):
    col = mongo_db.get_collection("usuarios")
    data = {k: v for k, v in dados.model_dump(exclude={'usuario'}).items() if v is not None}
    
    res = col.update_one({"usuario": dados.usuario}, {"$set": data})
    if res.matched_count == 0:
        raise HTTPException(404, "Usuário não encontrado")
    return {"sucesso": True}

# --- ROTA CORE: ANÁLISE ---

@app.post("/analise/executar", tags=["Analise"])
@measure_time
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
    logger.info(f"🚀 Iniciando análise completa para: {usuario}")
    
    # 1. Parse seguro de floats
    try:
        peso_float = float(str(peso).replace(',', '.'))
        alt_str = str(altura).replace(',', '.').replace('cm', '').strip()
        altura_int = int(float(alt_str) * 100) if float(alt_str) < 3.0 else int(float(alt_str))
    except:
        peso_float = 70.0; altura_int = 175
    
    # 2. Salva dados cadastrais
    col = mongo_db.get_collection("usuarios")
    col.update_one({"usuario": usuario}, {"$set": {
        "nome": nome_completo, "peso": peso_float, "altura": altura_int, 
        "genero": genero, "info_add": observacoes
    }})
    user_data = col.find_one({"usuario": usuario})
    if not user_data: raise HTTPException(404)

    # 3. Imagem
    raw_img = await foto.read()
    img_opt = ImageService.optimize(raw_img)
    
    # 4. Prompt Engineering (Fase 1 - Raciocínio)
    # Focamos em pedir raciocínio profundo
    prompt_brain = f"""
    ACT AS AN ELITE SPORTS SCIENTIST. CREATE THE ULTIMATE PROTOCOL.
    
    CLIENT: {nome_completo} ({genero}), {peso_float}kg, {altura_int}cm.
    GOAL: {objetivo}.
    RESTRICTIONS: {user_data.get('restricoes_fis')}, {user_data.get('restricoes_alim')}.
    
    TASKS:
    1. ANALYZE PHYSIQUE from image (fat distribution, insertions).
    2. DIET (7 DAYS): Detailed menu for Monday-Sunday. Exact macros.
    3. TRAINING (7 DAYS): Monday-Sunday split.
       - REQUIREMENT: Minimum 10 exercises per session. High Volume.
       - SELECTION: Use standard gym equipment.
       - BIOMECHANICS: Justify every exercise choice.
    4. SUPPLEMENTS: Evidence-based recommendations.
    """
    
    # 5. Pipeline CoT
    try:
        result_json = AIOrchestrator.execute_chain_of_thought(prompt_brain, img_opt)
    except Exception as e:
        logger.error(f"CoT Failure: {e}")
        raise HTTPException(503, "IA indisponível no momento. Tente novamente.")

    # 6. Validação de Exercícios
    if 'treino' in result_json:
        result_json['treino'] = validar_exercicios_final(result_json['treino'])

    # 7. Salvar e Cobrar
    dossie = {
        "id": str(ObjectId()),
        "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "timestamp": datetime.now(),
        "peso_reg": peso_float,
        "conteudo_bruto": {
            "json_full": result_json,
            # Compatibilidade legada
            "r1": str(result_json.get('avaliacao', {}).get('insight', '')),
            "r2": str(result_json.get('dieta_insight', '')),
            "r3": str(result_json.get('suplementacao_insight', '')),
            "r4": str(result_json.get('treino_insight', ''))
        }
    }
    
    update = {"$push": {"historico_dossies": dossie}}
    if not user_data.get('is_admin'): 
        update["$inc"] = {"avaliacoes_restantes": -1}
        
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
    
    prompt = f"Regenerate '{secao}' for {user.get('nome')}. Focus: {dia if dia else 'Full Week'}. Make it HARDCORE and DETAILED. Minimum 10 exercises/meals."
    
    try:
        # Usa CoT sem imagem
        new_content = AIOrchestrator.execute_chain_of_thought(prompt, None)
        
        # Merge simples no último dossiê
        last_dossie = user['historico_dossies'][-1]
        json_full = last_dossie['conteudo_bruto']['json_full']
        
        if secao in new_content:
            json_full[secao] = new_content[secao]
            if f"{secao}_insight" in new_content:
                json_full[f"{secao}_insight"] = new_content[f"{secao}_insight"]
        
        if secao == 'treino':
            json_full['treino'] = validar_exercicios_final(json_full['treino'])
            
        col.update_one(
            {"usuario": dados.get("usuario"), "historico_dossies.data": last_dossie['data']},
            {"$set": {"historico_dossies.$.conteudo_bruto.json_full": json_full}}
        )
        return {"sucesso": True, "resultado": last_dossie}
    except:
        return {"sucesso": False}

# --- ROTA LEGADA PARA HISTÓRICO (CORREÇÃO DE ERRO NO FLUTTER) ---

@app.get("/historico/{usuario}", tags=["Perfil"])
def buscar_historico(usuario: str):
    """
    Retorna o histórico completo e perfil atualizado.
    Rota crucial para evitar 'Exception not Found' no carregamento do app.
    """
    col = mongo_db.get_collection("usuarios")
    user = col.find_one({"usuario": usuario})
    if not user: 
        # Retorna estrutura vazia válida em vez de 404 para não crashar o app
        return {"sucesso": True, "historico": []}
    
    return {
        "sucesso": True, 
        "historico": jsonable_encoder(user.get('historico_dossies', [])), 
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

# --- ROTAS SOCIAIS ---

@app.get("/social/feed", tags=["Social"])
def get_feed():
    col = mongo_db.get_collection("posts")
    posts = list(col.find().sort("data", DESCENDING).limit(50))
    for p in posts: 
        p['_id'] = str(p['_id'])
        p['medalha'] = calcular_medalha(p.get('autor'))
    return {"sucesso": True, "feed": posts}

@app.post("/social/postar", tags=["Social"])
async def postar(
    usuario: str = Form(...), 
    legenda: str = Form(...), 
    imagem: UploadFile = File(...)
):
    img_bytes = await imagem.read()
    img_opt = ImageService.optimize(img_bytes, size=(600, 600))
    
    # Comentário rápido
    cmt = AIOrchestrator.simple_generation(f"Comentário curto e motivador (gym bro) para: {legenda}", img_opt)
    
    col = mongo_db.get_collection("posts")
    col.insert_one({
        "autor": usuario, "legenda": legenda, 
        "imagem": base64.b64encode(img_opt).decode('utf-8'),
        "data": datetime.now().isoformat(), "likes": [],
        "comentarios": [{"autor": "TechnoBolt AI", "texto": cmt}] if cmt else []
    })
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
    cmt = {
        "autor": dados.usuario,
        "texto": dados.texto,
        "data": datetime.now().isoformat()
    }
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
    now = datetime.now()
    start = datetime(now.year, now.month, 1).isoformat()
    checkins = list(col.find({"usuario": usuario, "data": {"$gte": start}}))
    
    formatted = {}
    for c in checkins:
        try:
            d = datetime.fromisoformat(c['data']).day
            formatted[d] = c['tipo']
        except: pass
    return {"sucesso": True, "checkins": formatted}

@app.post("/social/validar-conquista", tags=["Social"])
async def validar_conquista(
    usuario: str = Form(...),
    tipo: str = Form(...), 
    foto: UploadFile = File(...)
):
    """Valida checkin via IA de Visão."""
    col = mongo_db.get_collection("checkins")
    now = datetime.now()
    today_start = datetime(now.year, now.month, now.day).isoformat()
    
    if col.find_one({"usuario": usuario, "data": {"$gte": today_start}}):
        return {"sucesso": False, "mensagem": "Checkin já realizado hoje."}

    content = await foto.read()
    img_opt = ImageService.optimize(content)
    
    # Usa modelo rápido para visão
    resp = AIOrchestrator.simple_generation(f"Valide se esta imagem comprova um treino de {tipo}. Responda APROVADO ou REPROVADO.", img_opt)
    
    if resp and "APROVADO" in resp.upper():
        pts = 50
        col.insert_one({"usuario": usuario, "tipo": tipo, "data": now.isoformat(), "pontos": pts})
        mongo_db.get_collection("usuarios").update_one({"usuario": usuario}, {"$inc": {"pontos": pts}})
        return {"sucesso": True, "aprovado": True, "pontos": pts}
    else:
        return {"sucesso": True, "aprovado": False, "mensagem": "Não foi possível validar o treino."}

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

@app.get("/setup/criar-admin", tags=["Admin"])
def create_admin():
    col = mongo_db.get_collection("usuarios")
    if col.find_one({"usuario": "admin"}): return {"sucesso": False, "mensagem": "Admin existe."}
    col.insert_one({"usuario": "admin", "senha": "123", "nome": "Super Admin", "is_admin": True, "status": "ativo", "avaliacoes_restantes": 9999})
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
        
        if 'avaliacao' in data:
            pdf.card("Avaliação", data['avaliacao'].get('insight', ''))
            
        if 'dieta' in data:
            pdf.add_page()
            pdf.chapter_title("DIETA 7 DIAS")
            for d in data['dieta']:
                pdf.card(f"{d.get('dia')} - {d.get('foco_nutricional')}", d.get('macros_totais'))
                for r in d.get('refeicoes', []):
                    pdf.chapter_body(f"{r.get('horario')}: {r.get('alimentos')}")
        
        if 'treino' in data:
            pdf.add_page()
            pdf.chapter_title("TREINO HARDCORE 7 DIAS")
            for t in data['treino']:
                pdf.card(f"{t.get('dia')} - {t.get('foco')}", t.get('justificativa', ''))
                for ex in t.get('exercicios', []):
                    pdf.chapter_body(f"> {ex.get('nome')} [{ex.get('series_reps')}]")

        buf = io.BytesIO()
        out = pdf.output(dest='S')
        if isinstance(out, str): buf.write(out.encode('latin-1'))
        else: buf.write(out)
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="application/pdf", headers={'Content-Disposition': 'attachment; filename="TechnoBolt.pdf"'})
    except Exception as e:
        logger.error(f"PDF Err: {e}")
        raise HTTPException(500)
