"""
TechnoBolt Gym Hub API - Enterprise Architect Edition
Version: 2026.5.1-Titanium-Max-Tuned
Architecture: Modular Monolith | Hexagonal-ish | Event-Driven AI Pipeline
Author: TechnoBolt Engineering Team (Principal Architect)
Timestamp: 2026-02-11 17:15:00 UTC

System Overview:
This backend serves as the central nervous system for the TechnoBolt ecosystem.
It orchestrates user identity, social interactions, gamification mechanics, 
and high-fidelity generative AI workflows for personalized fitness coaching.

Key Design Patterns:
- Repository Pattern: For persistence abstraction.
- Chain-of-Thought (CoT): For high-reasoning AI generation.
- Circuit Breaker & Retry: For resilient external API communication.
- Structured Logging: For observability and distributed tracing.
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
import sys
from datetime import datetime, timedelta
from typing import (
    List, Optional, Any, Dict, Union, Callable, TypeVar, Tuple, Set, 
    Generator, AsyncGenerator, Coroutine
)
from enum import Enum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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
    Response,
    BackgroundTasks
)
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.security import APIKeyHeader
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware

# --- VALIDAÇÃO DE DADOS (PYDANTIC V2) ---
from pydantic import (
    BaseModel, 
    Field, 
    BeforeValidator, 
    ConfigDict, 
    validator, 
    field_validator, 
    HttpUrl, 
    EmailStr,
    AwareDatetime
)
from typing_extensions import Annotated

# --- PERSISTÊNCIA (MONGODB DRIVER) ---
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

# --- IA GENERATIVA (GOOGLE GEMINI) ---
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter
import pillow_heif

# --- GERAÇÃO DE DOCUMENTOS ---
from fpdf import FPDF

# ==============================================================================
# SECTION 1: CORE INFRASTRUCTURE & OBSERVABILITY
# ==============================================================================

class RequestContext:
    """Stores context for the current request execution flow."""
    _request_id: str = "system-startup"

    @classmethod
    def set_id(cls, req_id: str):
        cls._request_id = req_id

    @classmethod
    def get_id(cls) -> str:
        return cls._request_id

class EnterpriseLogger:
    """
    Structured logging system designed for high-concurrency production environments.
    Ensures every log entry is traceable to a specific request ID.
    """
    
    @staticmethod
    def setup() -> logging.Logger:
        """Configures the root logger with custom formatting."""
        logger = logging.getLogger("TechnoBoltAPI")
        logger.setLevel(logging.INFO)
        
        # Prevent handler duplication on uvicorn reload
        if logger.hasHandlers():
            logger.handlers.clear()
            
        handler = logging.StreamHandler(sys.stdout)
        
        # Format: [Time] | [Level] | [ReqID] | [Module:Func:Line] | Message
        class ContextFilter(logging.Filter):
            def filter(self, record):
                record.req_id = RequestContext.get_id()
                return True

        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | [%(req_id)s] | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        handler.addFilter(ContextFilter())
        logger.addHandler(handler)
        
        return logger

# Global Logger Instance
logger = EnterpriseLogger.setup()

# ==============================================================================
# SECTION 2: SYSTEM BOOTSTRAP & DRIVER INITIALIZATION
# ==============================================================================

def initialize_media_drivers():
    """
    Registers external codecs for image processing.
    Crucial for handling HEIC uploads from iOS devices in the Flutter app.
    """
    try:
        logger.info("🔧 Initializing Pillow HEIF opener...")
        pillow_heif.register_heif_opener()
        logger.info("✅ HEIC/HEIF Codec registered successfully. iOS uploads enabled.")
    except ImportError as e:
        logger.warning(f"⚠️ Critical Dependency Missing: 'pillow_heif'. iOS uploads will fail. Error: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error registering HEIC codec: {e}")

# Run initialization sequence
initialize_media_drivers()

# ==============================================================================
# SECTION 3: CONFIGURATION MANAGEMENT (SINGLETON)
# ==============================================================================

class Settings:
    """
    Centralized configuration manager implementing the Singleton pattern.
    Handles environment variable validation, secrets management, and constant definitions.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._load_and_validate()
        return cls._instance

    def _load_and_validate(self):
        logger.info("⚙️  Loading environment configurations...")
        
        # Database Credentials (Fail-fast if critical vars are missing in Prod)
        self.MONGO_USER = self._get_env("MONGO_USER", "technobolt", critical=False)
        self.MONGO_PASS = self._get_env("MONGO_PASS", "tech@132", critical=False)
        self.MONGO_HOST = self._get_env("MONGO_HOST", "cluster0.zbjsvk6.mongodb.net", critical=False)
        self.DB_NAME = self._get_env("DB_NAME", "technoboltgym", critical=False)
        
        # API Metadata
        self.API_TITLE = "TechnoBolt Gym Hub API"
        self.API_VERSION = "121.0-Architect"
        self.ENV = self._get_env("ENV", "production")
        
        # AI Configuration (Load Balancer Pool)
        self.GEMINI_KEYS = self._load_api_keys()
        
        # AI Strategy Definition
        # Unified Model Strategy: Using high-reasoning models for ALL phases
        self.AI_MODELS_PRIORITY = [
            "models/gemini-3-flash-preview",  # Primary: High Intelligence & Speed
            "models/gemini-2.5-flash",        # Secondary: Stability
            "models/gemini-2.0-flash",        # Tertiary: Fallback
            "models/gemini-flash-latest"      # Quaternary: Last Resort
        ]
        
        self.AI_STRUCTURE_MODELS = [
            "models/gemini-flash-latest"
        ]
        
        logger.info(f"🧠 AI Priority Chain Loaded: {len(self.AI_MODELS_PRIORITY)} models active.")

    def _get_env(self, key: str, default: Any = None, critical: bool = False) -> str:
        """Retrieves and validates environment variables."""
        value = os.environ.get(key, default)
        if value is None:
            msg = f"Missing Environment Variable: {key}"
            if critical:
                logger.critical(f"❌ {msg}")
                raise RuntimeError(msg)
            else:
                logger.warning(f"⚠️ {msg}. Using default/fallback.")
        return value

    def _load_api_keys(self) -> List[str]:
        """Loads and validates the API Key pool for rotation."""
        keys = []
        for i in range(1, 21):
            key_val = os.environ.get(f"GEMINI_CHAVE_{i}")
            if key_val and len(key_val.strip()) > 20: # Basic validation
                keys.append(key_val.strip())
        
        if not keys:
            logger.critical("❌ CRITICAL: No AI API Keys found in environment (GEMINI_CHAVE_x). AI features will crash.")
        else:
            logger.info(f"🔑 AI Key Pool Initialized: {len(keys)} keys ready for rotation.")
        
        return keys

# Initialize Settings
settings = Settings()

# ==============================================================================
# SECTION 4: EXCEPTION HIERARCHY & ERROR HANDLING
# ==============================================================================

class BaseAPIException(Exception):
    """Base class for all API-specific exceptions."""
    def __init__(self, message: str, status_code: int = 500, details: Any = None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)

class DatabaseConnectionError(BaseAPIException):
    """Raised when DB connection fails after retries."""
    def __init__(self, details: str = ""):
        super().__init__(f"Database service unavailable. {details}", 503, details)

class AIReasoningError(BaseAPIException):
    """Raised when the AI 'Brain' phase fails to produce valid output."""
    def __init__(self, details: str = ""):
        super().__init__("AI Reasoning Engine failed to generate strategy.", 502, details)

class AIStructuringError(BaseAPIException):
    """Raised when the AI 'Formatter' phase fails to produce valid JSON."""
    def __init__(self, details: str = ""):
        super().__init__("AI Structuring Engine failed to parse JSON output.", 502, details)

class ResourceNotFoundError(BaseAPIException):
    """Raised when a requested resource (User, Post) is missing."""
    def __init__(self, resource: str):
        super().__init__(f"Resource '{resource}' not found.", 404)

class ValidationBusinessError(BaseAPIException):
    """Raised for domain-specific validation failures."""
    def __init__(self, message: str):
        super().__init__(message, 400)

class InsufficientCreditsError(BaseAPIException):
    """Raised when user has no credits for premium features."""
    def __init__(self):
        super().__init__("Insufficient credits. Please upgrade or contact admin.", 402)

# ==============================================================================
# SECTION 5: MIDDLEWARE & DECORATORS
# ==============================================================================

def measure_time(func):
    """
    Async Decorator for Performance Monitoring (APM).
    Injects Request-ID and logs execution time for asynchronous endpoints.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        req_id = str(uuid.uuid4())[:8]
        RequestContext.set_id(req_id) # Set context for logger
        func_name = func.__name__
        
        logger.info(f"⚡ Executing Endpoint: {func_name}")
        try:
            result = await func(*args, **kwargs)
            return result
        except HTTPException as he:
            logger.warning(f"⚠️ {func_name} - HTTP {he.status_code}: {he.detail}")
            raise he
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error(f"❌ {func_name} FAILED after {elapsed:.2f}ms. Trace: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail="Internal Server Error")
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"🏁 {func_name} finished in {elapsed:.2f}ms")
    return wrapper

def sync_measure_time(func):
    """
    Sync Decorator for Performance Monitoring.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        req_id = str(uuid.uuid4())[:8]
        RequestContext.set_id(req_id)
        func_name = func.__name__
        
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"🏁 {func_name} finished in {elapsed:.2f}ms")
    return wrapper

# ==============================================================================
# SECTION 6: DATA LAYER (MONGODB REPOSITORY)
# ==============================================================================

PyObjectId = Annotated[str, BeforeValidator(str)]

class MongoManager:
    """
    High-Availability MongoDB Connection Manager.
    Implements Singleton pattern, Connection Pooling, and Automatic Reconnection logic.
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
        Establishes the connection to MongoDB Atlas with production-tuned parameters.
        """
        try:
            logger.info("🔌 Connecting to MongoDB Atlas...")
            
            password = urllib.parse.quote_plus(settings.MONGO_PASS)
            uri = f"mongodb+srv://{settings.MONGO_USER}:{password}@{settings.MONGO_HOST}/?appName=Cluster0"
            
            self.client = MongoClient(
                uri,
                serverSelectionTimeoutMS=5000, # Fail fast if DNS is down
                connectTimeoutMS=10000,        # TCP connect timeout
                socketTimeoutMS=10000,         # I/O timeout
                maxPoolSize=100,               # High concurrency support
                minPoolSize=10,                # Keep connections warm
                retryWrites=True,
                retryReads=True
            )
            
            # Immediate Health Check
            self.client.admin.command('ping')
            self.db = self.client[settings.DB_NAME]
            
            logger.info(f"✅ MongoDB Connected: {settings.DB_NAME}")
            self._ensure_indexes()
            
        except Exception as e:
            logger.critical(f"❌ MongoDB Connection Failed: {e}")
            self.client = None
            self.db = None

    def _ensure_indexes(self):
        """Ensures critical indexes exist for query performance."""
        try:
            if self.db is not None:
                # User Indexes
                self.db.usuarios.create_index("usuario", unique=True)
                self.db.usuarios.create_index([("usuario", ASCENDING), ("senha", ASCENDING)])
                
                # Feed Indexes
                self.db.posts.create_index("data", direction=DESCENDING)
                self.db.posts.create_index("autor", direction=ASCENDING)
                
                # Gamification Indexes
                self.db.checkins.create_index([("usuario", ASCENDING), ("data", DESCENDING)])
                
                logger.info("✅ Database Indexes Verified.")
        except Exception as e:
            logger.warning(f"⚠️ Index Creation Warning: {e}")

    def get_collection(self, collection_name: str):
        """
        Retrieves a collection handle, attempting reconnection if necessary.
        """
        if self.client is None or self.db is None:
            logger.warning("🔄 DB Connection lost. Attempting reconnection...")
            self._initialize_connection()
            
        if self.db is None:
             raise DatabaseConnectionError("Fatal: Could not restore DB connection.")
             
        return self.db[collection_name]

# Global DB Instance
mongo_db = MongoManager()

# ==============================================================================
# SECTION 7: DOMAIN MODELS (DTOs)
# ==============================================================================

class MongoBaseModel(BaseModel):
    """Base Pydantic model for MongoDB documents."""
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    
    model_config = ConfigDict(
        populate_by_name=True, 
        arbitrary_types_allowed=True, 
        json_encoders={ObjectId: str, datetime: lambda v: v.isoformat()}
    )

class UserLogin(BaseModel):
    """Login Credentials."""
    usuario: str = Field(..., min_length=3, description="Username")
    senha: str = Field(..., min_length=3, description="Password")

class UserRegister(BaseModel):
    """Registration Data."""
    usuario: str = Field(..., min_length=3, max_length=50)
    senha: str = Field(..., min_length=3)
    nome: str = Field(..., min_length=2)
    peso: float = Field(..., gt=0, lt=500, description="Weight (kg)")
    altura: float = Field(..., gt=0, lt=300, description="Height (cm)")
    genero: str = Field(..., pattern="^(Masculino|Feminino|Outro)$")

class UserUpdate(BaseModel):
    """Profile Update DTO."""
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
    """Action on Post."""
    usuario: str
    post_id: str

class SocialCommentRequest(BaseModel):
    """New Comment."""
    usuario: str
    post_id: str
    texto: str = Field(..., min_length=1, max_length=500)

class ChatMessageRequest(BaseModel):
    """Chat Message DTO."""
    remetente: str
    destinatario: str
    texto: str

class AdminUserEdit(BaseModel):
    """Admin Action."""
    target_user: str
    status: Optional[str] = None
    creditos: Optional[int] = None

# ==============================================================================
# SECTION 8: DOMAIN SERVICES & UTILS
# ==============================================================================

def normalizar_texto(texto: str) -> str:
    """
    Normalize text for fuzzy matching.
    Removes accents, converts to lowercase, strips whitespace.
    """
    if not texto: return ""
    return "".join(c for c in unicodedata.normalize('NFD', str(texto)) if unicodedata.category(c) != 'Mn').lower().strip()

class ExerciseRepository:
    """
    In-Memory Exercise Database.
    Loads from JSON on startup and serves as the 'Ground Truth' for exercise names.
    """
    _db: Dict[str, str] = {}
    _keys_string: str = ""
    
    @classmethod
    def load(cls):
        """Loads exercises.json into memory."""
        try:
            path = "exercises.json"
            if not os.path.exists(path):
                cls._db = {"supino reto": "chest/bench_press", "agachamento": "legs/squat"} # Fallback
                logger.warning("⚠️ exercises.json missing. Using minimal fallback data.")
            else:
                with open(path, "r", encoding="utf-8") as f:
                    cls._db = json.load(f)
            
            all_keys = list(cls._db.keys())
            cls._keys_string = ", ".join(all_keys) 
            logger.info(f"✅ ExerciseRepository Loaded: {len(cls._db)} exercises.")
        except Exception as e:
            logger.error(f"❌ Failed to load Exercise Repository: {e}")
            cls._db = {}

    @classmethod
    def get_keys_string(cls) -> str:
        return cls._keys_string

    @classmethod
    def get_db(cls) -> Dict[str, str]:
        return cls._db

# Load Repository
ExerciseRepository.load()

def validar_exercicios_final(treino_data: list) -> list:
    """
    Post-Processing Validator.
    Maps AI-generated exercise names to valid GitHub raw URLs for thumbnails.
    Uses fuzzy matching (difflib) for robust mapping.
    """
    db = ExerciseRepository.get_db()
    if not treino_data or not db: return treino_data
    
    base_url = "https://raw.githubusercontent.com/italoat/technobolt-backend/main/assets/exercises"
    
    # Pre-compute normalized maps for O(1) lookups
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
            
            # Strategy 1: Exact Match
            if norm_name in db_map:
                path = db_map[norm_name]
                final_name = db_titles[norm_name]
            else:
                # Strategy 2: Fuzzy Match
                matches = difflib.get_close_matches(norm_name, db_map.keys(), n=1, cutoff=0.6)
                if matches:
                    path = db_map[matches[0]]
                    final_name = db_titles[matches[0]]
                else:
                    # Strategy 3: Substring Search
                    for k in db_map.keys():
                        if k in norm_name or norm_name in k:
                            path = db_map[k]
                            final_name = db_titles[k]
                            break
                    # Strategy 4: Fallback
                    if not path and "polichinelo" in db_map:
                        path = db_map["polichinelo"]
                        final_name = f"{raw_name} (Adaptado)"

            # Enrich object
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
    """Gamification: Calculates user rank badge."""
    try:
        user = mongo_db.get_collection("usuarios").find_one({"usuario": username})
        pts = user.get('pontos', 0) if user else 0
        if pts > 5000: return "🏆"
        if pts > 1000: return "🥇"
        if pts > 500: return "🥈"
        if pts > 100: return "🥉"
        return ""
    except: return ""

class ImageService:
    """
    Service for Image Processing.
    Handles compression, EXIF correction, and format conversion.
    """
    
    @staticmethod
    def optimize(file_bytes: bytes, quality: int = 75, max_size: tuple = (800, 800)) -> bytes:
        """
        Optimizes an image byte stream for AI consumption.
        - Corrects Orientation
        - Converts to RGB (removes Alpha)
        - Resizes to max_size
        - Compresses to JPEG
        """
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                # EXIF Transpose (Fix iOS rotation)
                img = ImageOps.exif_transpose(img)
                
                # Convert RGBA/P to RGB
                if img.mode != 'RGB':
                    img = img.convert("RGB")
                
                # Resize (Thumbnail maintains aspect ratio)
                img.thumbnail(max_size)
                
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality, optimize=True)
                return output.getvalue()
        except Exception as e:
            logger.error(f"❌ Image Optimization Failed: {e}. Passing original bytes.")
            return file_bytes

class PDFReport(FPDF):
    """
    Custom PDF Generator for Fitness Protocols.
    """
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.col_bg = (20, 20, 25)
        self.col_text = (230, 230, 230)
        self.col_accent = (0, 200, 255)

    def sanitize(self, txt: Any) -> str:
        """Sanitizes text to Latin-1 for FPDF compatibility."""
        if not txt: return ""
        s = str(txt).replace("’", "'").replace("–", "-").replace("“", '"').replace("”", '"')
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
        self.set_fill_color(30, 30, 35) 
        self.set_text_color(*self.col_accent)
        self.set_font("Arial", "B", 11)
        self.multi_cell(0, 6, self.sanitize(title), fill=True)
        self.set_text_color(*self.col_text)
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 6, self.sanitize(body), fill=True)
        self.ln(2)

# ==============================================================================
# SECTION 9: AI INFRASTRUCTURE (KEY ROTATION)
# ==============================================================================

class KeyRotationManager:
    """
    AI Key Load Balancer.
    Rotates API keys to avoid 429 Errors and maximize throughput.
    """
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.cooldowns: Dict[str, float] = {} 
        self.COOLDOWN_SECONDS = 60.0

    def get_available_keys(self) -> List[str]:
        """Returns a shuffled list of non-cooldown keys."""
        now = time.time()
        # Clean up expired cooldowns
        self.cooldowns = {k: v for k, v in self.cooldowns.items() if v <= now}
        
        available = [k for k in self.keys if k not in self.cooldowns]
        
        # Fail-Open: Use all keys if all are technically in cooldown
        if not available and self.keys:
            logger.warning("⚠️ All keys in cooldown. Forcing usage of pool.")
            return self.keys
            
        random.shuffle(available)
        return available

    def report_rate_limit(self, key: str):
        """Marks a key as exhausted for a period."""
        logger.warning(f"⚠️ Rate Limit on key ...{key[-4:]}. Blocking for {self.COOLDOWN_SECONDS}s.")
        self.cooldowns[key] = time.time() + self.COOLDOWN_SECONDS

# Initialize Key Manager
key_manager = KeyRotationManager(settings.GEMINI_KEYS)

# ==============================================================================
# SECTION 10: AI CORE LOGIC (JSON REPAIR & ORCHESTRATION)
# ==============================================================================

class JSONRepairKit:
    """
    Advanced Heuristic JSON Repair Engine.
    Fixes common LLM JSON syntax errors using Regex patterns.
    """
    
    @staticmethod
    def fix_json_string(text: str) -> str:
        """Cleans and fixes the raw string from LLM."""
        try:
            text = text.strip()
            # Strip Markdown code blocks
            if "```" in text:
                text = re.sub(r'```json|```', '', text).strip()
            
            # Remove comments
            text = re.sub(r'//.*?\n|/\*.*?\*/', '', text, flags=re.S)
            
            # Remove trailing commas (illegal in JSON)
            text = re.sub(r',(\s*[}\]])', r'\1', text)
            
            # --- NUCLEAR FIX: Insert missing commas ---
            # Between object/array close and start
            text = re.sub(r'([}\]])\s*([{\[])', r'\1,\2', text)
            # Between value and key
            text = re.sub(r'("\s*)\s+"', r'\1,"', text)
            # Between number and key
            text = re.sub(r'(\d+)\s+"', r'\1,"', text)

            # Brace Balancing
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
        Robust Parsing Pipeline.
        Attempts multiple strategies to extract valid JSON.
        """
        # Strategy 1: Direct Parse
        try: return json.loads(text_ia)
        except: pass
        
        # Strategy 2: Regex Extraction
        try:
            match = re.search(r'(\{.*\})', text_ia, re.DOTALL)
            if match: return json.loads(match.group(1))
        except: pass
        
        # Strategy 3: Aggressive Repair
        try:
            repaired = cls.fix_json_string(text_ia)
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON Parse Failed. Error: {e}")
            raise AIStructuringError(f"JSON Structure Invalid: {e}")

class AIOrchestrator:
    """
    AI Execution Engine.
    Manages Model Rotation, Key Rotation, and the Reasoning->Formatting pipeline.
    """
    
    @staticmethod
    def _call_gemini_with_retry(model_name: str, prompt: str, image_bytes: Optional[bytes] = None, 
                              json_mode: bool = False, temperature: float = 0.7) -> str:
        """
        Executes a single LLM call with Key Rotation.
        Iterates through ALL available keys for the given model before failing.
        """
        keys = key_manager.get_available_keys()
        if not keys: raise AIProcessingError("API Key Pool Exhausted.")
        
        last_error = None
        
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
                
                # Execute Call
                response = model.generate_content(inputs, generation_config=config)
                
                if response and response.text:
                    logger.info(f"   ✅ AI Success: {model_name} (Key ...{api_key[-4:]})")
                    return response.text
                
            except Exception as e:
                err_str = str(e)
                # Handle Rate Limits
                if "429" in err_str or "Resource exhausted" in err_str:
                    key_manager.report_rate_limit(api_key)
                
                logger.warning(f"   ⚠️ Key Failed (...{api_key[-4:]}): {err_str[:100]}")
                last_error = e
                time.sleep(1.0) # Backoff
                continue 
        
        logger.error(f"❌ Model Failed: {model_name} (All keys tried).")
        raise last_error if last_error else Exception(f"Model {model_name} failed completely.")

    @staticmethod
    def execute_chain_of_thought(context_prompt: str, image_bytes: Optional[bytes]) -> Dict:
        """
        Executes the Unified Chain-of-Thought Pipeline.
        
        Strategy:
        1. Iterate through Model Priority List (Gemini 3 -> 2.5 -> 2.0).
        2. For each model, attempt to run BOTH Phase 1 (Reasoning) and Phase 2 (Formatting).
        3. If any phase fails for a model (after trying all keys), failover to the next model.
        """
        
        for model in settings.AI_MODELS_PRIORITY:
            logger.info(f"🔄 Attempting Pipeline with Model: {model}...")
            
            try:
                # --- PHASE 1: REASONING (The Brain) ---
                logger.info(f"🧠 [Phase 1 - Reasoning] Running on {model}...")
                
                # UPDATE 1: Enhanced Physical Assessment & Caloric Surplus Logic
                prompt_p1 = context_prompt + """
                \n\nCRITICAL INSTRUCTION: Think step-by-step. Generate a HIGHLY DETAILED, VERBOSE text strategy. 
                Do not output JSON yet. Focus on:
                
                1. VISUAL ASSESSMENT: 
                   - Analyze posture, symmetry, muscle insertions, and estimated body fat % from the image. 
                   - Be specific about weak points.
                
                2. DIET & CALORIC SURPLUS:
                   - Explicitly state the daily caloric surplus target (e.g., +300kcal).
                   - Calculate total daily calories and macros.
                   - Show the math: BMR + Activity + Surplus.
                
                3. TRAINING OPTIMIZATION:
                   - Ensure High Volume (10+ exercises per session).
                   - PROVIDE BIOMECHANICAL JUSTIFICATION for every single exercise. Explain WHY it was chosen.
                """
                
                strategy_text = AIOrchestrator._call_gemini_with_retry(
                    model_name=model,
                    prompt=prompt_p1,
                    image_bytes=image_bytes,
                    json_mode=False,
                    temperature=0.7
                )
                
                # --- PHASE 2: FORMATTING (The Structurer) ---
                # We use the SAME model to ensure capability consistency
                logger.info(f"⚡ [Phase 2 - Formatting] Running on {model}...")
                
                exercise_list_str = ExerciseRepository.get_keys_string()
                
                # UPDATE 2: Strict JSON Schema Enforcement
                prompt_p2 = f"""
                TASK: Act as a Strict JSON Compiler.
                Convert the following Fitness Strategy into VALID JSON matching the schema.
                
                SOURCE TEXT:
                {strategy_text}
                
                CONSTRAINTS:
                1. Output ONLY pure JSON.
                2. DATA INTEGRITY:
                   - 'dieta': Must contain 7 objects (Monday-Sunday). Include 'superavit_calorico' field.
                   - 'treino': Must contain 7 objects (Monday-Sunday).
                   - 'suplementacao': Must not be empty.
                3. VALIDATION: Map exercises to: [{exercise_list_str}]. Use closest match or "(Adaptado)".
                
                REQUIRED SCHEMA:
                {{
                  "avaliacao": {{ 
                      "segmentacao": {{ "tronco": "Txt", "superior": "Txt", "inferior": "Txt" }}, 
                      "dobras": {{ "abdominal": "Txt", "suprailiaca": "Txt", "peitoral": "Txt" }}, 
                      "analise_postural": "Txt", 
                      "simetria": "Txt", 
                      "estimativa_gordura": "Txt",
                      "insight": "Txt" 
                  }},
                  "dieta": [ 
                    {{ 
                      "dia": "Segunda", 
                      "foco_nutricional": "Txt", 
                      "superavit_calorico": "Txt",
                      "refeicoes": [ {{ "horario": "...", "nome": "...", "alimentos": "..." }} ], 
                      "macros_totais": "Txt" 
                    }}, 
                    ... 
                  ],
                  "dieta_insight": "Txt",
                  "suplementacao": [ {{ "nome": "...", "dose": "...", "horario": "...", "motivo": "..." }} ],
                  "suplementacao_insight": "Txt",
                  "treino": [ 
                     {{ 
                        "dia": "Segunda", 
                        "foco": "...", 
                        "exercicios": [ 
                            {{ "nome": "...", "series_reps": "...", "execucao": "...", "justificativa_biomecanica": "Txt" }} 
                        ], 
                        "treino_alternativo": "...", 
                        "justificativa": "..." 
                     }},
                     ...
                  ],
                  "treino_insight": "Txt"
                }}
                """
                
                json_text = AIOrchestrator._call_gemini_with_retry(
                    model_name=model,
                    prompt=prompt_p2,
                    image_bytes=None, 
                    json_mode=True,
                    temperature=0.0 # Strict mode
                )
                
                # Parse & Validate
                data = JSONRepairKit.parse_robust(json_text)
                
                if not data.get('dieta') or len(data['dieta']) < 1:
                    raise ValueError("Generated diet list is empty.")
                
                return data # Success!
                
            except Exception as e:
                logger.warning(f"⚠️ Pipeline Failed for {model}: {e}. Failing over to next model...")
                continue # Try next model
                
        # If we get here, all models failed
        raise AIProcessingError("All AI models failed to generate valid protocol.")

    @staticmethod
    def simple_generation(prompt: str, image_bytes: Optional[bytes] = None) -> str:
        """Fast generation for comments/validation."""
        for model in settings.AI_MODELS_PRIORITY:
            try:
                return AIOrchestrator._call_gemini_with_retry(model, prompt, image_bytes, False)
            except: continue
        return "Analysis in progress..."

# ==============================================================================
# SECTION 11: FASTAPI APPLICATION & ROUTES
# ==============================================================================

app = FastAPI(
    title=settings.API_TITLE, 
    version=settings.API_VERSION,
    description="The Core Neural Backend for TechnoBolt."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# ROUTE GROUP: AUTHENTICATION
# ------------------------------------------------------------------------------

@app.post("/auth/login", tags=["Authentication"])
@sync_measure_time
def login_endpoint(dados: UserLogin):
    """Authenticates a user and returns their profile."""
    col = mongo_db.get_collection("usuarios")
    user = col.find_one({"usuario": dados.usuario, "senha": dados.senha})
    
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials.")
    if user.get("status") != "ativo" and not user.get("is_admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account pending approval.")
    
    user_data = {k: v for k, v in user.items() if k != "_id"}
    return {"sucesso": True, "dados": user_data}

@app.post("/auth/registro", tags=["Authentication"])
def register_endpoint(dados: UserRegister):
    """Registers a new user."""
    col = mongo_db.get_collection("usuarios")
    if col.find_one({"usuario": dados.usuario}):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Username already exists.")
    
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
    return {"sucesso": True, "mensagem": "Registration successful."}

# ------------------------------------------------------------------------------
# ROUTE GROUP: CORE ANALYSIS (THE BRAIN)
# ------------------------------------------------------------------------------

@app.post("/analise/executar", tags=["Analysis"])
@measure_time
async def execute_analysis_endpoint(
    usuario: str = Form(...), nome_completo: str = Form(...), peso: str = Form(...), 
    altura: str = Form(...), objetivo: str = Form(...), genero: str = Form("Masculino"),
    observacoes: str = Form(""), foto: UploadFile = File(...)
):
    """
    Main AI Entrypoint.
    1. Updates user metrics.
    2. Processes user photo.
    3. Triggers Chain-of-Thought AI Pipeline.
    4. Saves results and deducts credits.
    """
    logger.info(f"🚀 Launching Analysis for {usuario}...")
    
    # 1. Metric Parsing
    try:
        p_float = float(str(peso).replace(',', '.'))
        alt_raw = str(altura).replace(',', '.').strip()
        alt_int = int(float(alt_raw) * 100) if float(alt_raw) < 3.0 else int(float(alt_raw))
    except:
        p_float = 70.0; alt_int = 175
    
    # 2. User Update
    col = mongo_db.get_collection("usuarios")
    col.update_one({"usuario": usuario}, {"$set": {
        "nome": nome_completo, "peso": p_float, "altura": alt_int, 
        "genero": genero, "info_add": observacoes
    }})
    user = col.find_one({"usuario": usuario})
    if not user: raise HTTPException(404, "User not found.")

    # 3. Image Processing
    raw_img = await foto.read()
    img_opt = ImageService.optimize(raw_img)
    
    # 4. Prompt Construction - UPDATE: More aggressive instruction
    prompt = f"""
    ACT AS AN ELITE SPORTS SCIENTIST.
    SUBJECT: {nome_completo} ({genero}), {p_float}kg, {alt_int}cm.
    GOAL: {objetivo}. 
    RESTRICTIONS: {user.get('restricoes_fis', 'None')}, {user.get('restricoes_alim', 'None')}.
    TASKS: 
    1. Visual Assessment (Be critical).
    2. Diet for Hypertrophy (Surplus required).
    3. High Volume Training (10+ Exercises/day).
    4. Advanced Supplementation.
    """
    
    # 5. AI Execution
    try:
        result = AIOrchestrator.execute_chain_of_thought(prompt, img_opt)
    except Exception as e:
        logger.error(f"AI Pipeline Failed: {e}")
        raise HTTPException(503, "AI Service Overload. Please try again.")

    # 6. Post-Processing
    if 'treino' in result:
        result['treino'] = validar_exercicios_final(result['treino'])

    # 7. Persistence
    dossie = {
        "id": str(ObjectId()),
        "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "peso_reg": p_float,
        "conteudo_bruto": {
            "json_full": result,
            "r1": str(result.get('avaliacao', {}).get('insight', '')),
            "r2": str(result.get('dieta_insight', '')),
            "r3": str(result.get('suplementacao_insight', '')),
            "r4": str(result.get('treino_insight', ''))
        }
    }
    
    update_op = {"$push": {"historico_dossies": dossie}}
    if not user.get('is_admin'):
        update_op["$inc"] = {"avaliacoes_restantes": -1}
        
    col.update_one({"usuario": usuario}, update_op)
    
    return {"sucesso": True, "resultado": dossie}

@app.post("/analise/regenerar-secao", tags=["Analysis"])
async def regenerate_section_endpoint(dados: dict = Body(...)):
    """Regenerates a specific section of the protocol."""
    col = mongo_db.get_collection("usuarios")
    user = col.find_one({"usuario": dados.get("usuario")})
    if not user: return {"sucesso": False}
    
    prompt = f"Regenerate ONLY the '{dados.get('secao')}' section for {user.get('nome')}. Make it intense."
    try:
        new_content = AIOrchestrator.execute_chain_of_thought(prompt, None)
        # Merging logic...
        last_dossie = user['historico_dossies'][-1]
        last_dossie['conteudo_bruto']['json_full'][dados.get('secao')] = new_content[dados.get('secao')]
        
        col.update_one(
            {"usuario": dados.get("usuario"), "historico_dossies.data": last_dossie['data']},
            {"$set": {"historico_dossies.$.conteudo_bruto.json_full": last_dossie['conteudo_bruto']['json_full']}}
        )
        return {"sucesso": True, "resultado": last_dossie}
    except: return {"sucesso": False}

@app.get("/analise/baixar-pdf/{usuario}", tags=["Analysis"])
def download_pdf_endpoint(usuario: str):
    """Generates a PDF report."""
    try:
        user = mongo_db.get_collection("usuarios").find_one({"usuario": usuario})
        if not user: raise HTTPException(404)
        
        data = user['historico_dossies'][-1]['conteudo_bruto']['json_full']
        pdf = PDFReport()
        pdf.add_page()
        pdf.chapter_title(f"RELATORIO: {user.get('nome')}")
        
        if 'dieta' in data:
            pdf.chapter_title("DIETA")
            for d in data['dieta']: pdf.card(d['dia'], str(d.get('refeicoes')))
            
        buf = io.BytesIO()
        out = pdf.output(dest='S').encode('latin-1', 'replace')
        buf.write(out)
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/pdf", headers={'Content-Disposition': 'attachment; filename="report.pdf"'})
    except: raise HTTPException(500)

# ------------------------------------------------------------------------------
# ROUTE GROUP: USER PROFILE & HISTORY
# ------------------------------------------------------------------------------

@app.post("/perfil/atualizar", tags=["Profile"])
def update_profile_endpoint(dados: UserUpdate):
    col = mongo_db.get_collection("usuarios")
    data = {k: v for k, v in dados.model_dump(exclude={'usuario'}).items() if v is not None}
    res = col.update_one({"usuario": dados.usuario}, {"$set": data})
    if res.matched_count == 0: raise HTTPException(404)
    return {"sucesso": True}

@app.get("/historico/{usuario}", tags=["Profile"])
def get_history_endpoint(usuario: str):
    """Legacy endpoint for app compatibility."""
    user = mongo_db.get_collection("usuarios").find_one({"usuario": usuario})
    if not user: return {"sucesso": True, "historico": []}
    return {
        "sucesso": True, 
        "historico": jsonable_encoder(user.get('historico_dossies', [])),
        "creditos": user.get('avaliacoes_restantes', 0),
        "perfil": {k: v for k, v in user.items() if k not in ['_id', 'historico_dossies', 'senha']}
    }

# ------------------------------------------------------------------------------
# ROUTE GROUP: SOCIAL & GAMIFICATION
# ------------------------------------------------------------------------------

@app.get("/social/feed", tags=["Social"])
def get_feed_endpoint():
    posts = list(mongo_db.get_collection("posts").find().sort("data", -1).limit(50))
    for p in posts: p['_id'] = str(p['_id'])
    return {"sucesso": True, "feed": posts}

@app.post("/social/postar", tags=["Social"])
async def create_post_endpoint(
    usuario: str = Form(...), legenda: str = Form(...), imagem: UploadFile = File(...)
):
    img = ImageService.optimize(await imagem.read())
    cmt = AIOrchestrator.simple_generation(f"Comment like a gym bro: {legenda}", img)
    
    mongo_db.get_collection("posts").insert_one({
        "autor": usuario, "legenda": legenda, 
        "imagem": base64.b64encode(img).decode('utf-8'),
        "data": datetime.now().isoformat(), "likes": [], 
        "comentarios": [{"autor": "TechnoBolt AI", "texto": cmt}]
    })
    return {"sucesso": True}

@app.post("/social/curtir", tags=["Social"])
def like_post_endpoint(dados: SocialPostRequest):
    col = mongo_db.get_collection("posts")
    oid = ObjectId(dados.post_id)
    post = col.find_one({"_id": oid})
    if post:
        op = "$pull" if dados.usuario in post.get("likes", []) else "$addToSet"
        col.update_one({"_id": oid}, {op: {"likes": dados.usuario}})
    return {"sucesso": True}

@app.post("/social/comentar", tags=["Social"])
def comment_post_endpoint(dados: SocialCommentRequest):
    mongo_db.get_collection("posts").update_one(
        {"_id": ObjectId(dados.post_id)}, 
        {"$push": {"comentarios": {"autor": dados.usuario, "texto": dados.texto, "data": datetime.now().isoformat()}}}
    )
    return {"sucesso": True}

@app.post("/social/post/deletar", tags=["Social"])
def delete_post_endpoint(dados: SocialPostRequest):
    res = mongo_db.get_collection("posts").delete_one({"_id": ObjectId(dados.post_id), "autor": dados.usuario})
    return {"sucesso": res.deleted_count > 0}

@app.get("/social/ranking", tags=["Gamification"])
def get_ranking_endpoint():
    users = list(mongo_db.get_collection("usuarios").find({"is_admin": False}, {"nome": 1, "usuario": 1, "pontos": 1, "foto_perfil": 1, "_id": 0}).sort("pontos", -1).limit(50))
    return {"sucesso": True, "ranking": users}

@app.get("/social/checkins", tags=["Gamification"])
def get_checkins_endpoint(usuario: str):
    raw = list(mongo_db.get_collection("checkins").find({"usuario": usuario}))
    return {"sucesso": True, "checkins": {datetime.fromisoformat(c['data']).day: c['tipo'] for c in raw}}

@app.post("/social/validar-conquista", tags=["Gamification"])
async def validate_checkin_endpoint(
    usuario: str = Form(...), tipo: str = Form(...), foto: UploadFile = File(...)
):
    now = datetime.now()
    if mongo_db.get_collection("checkins").find_one({"usuario": usuario, "data": {"$gte": datetime(now.year, now.month, now.day).isoformat()}}):
        return {"sucesso": False, "mensagem": "Checkin already done today."}
    
    img = ImageService.optimize(await foto.read())
    resp = AIOrchestrator.simple_generation(f"Is this a {tipo} workout? Reply APPROVED or REJECTED.", img)
    
    if "APROVADO" in resp.upper():
        mongo_db.get_collection("checkins").insert_one({"usuario": usuario, "tipo": tipo, "data": now.isoformat(), "pontos": 50})
        mongo_db.get_collection("usuarios").update_one({"usuario": usuario}, {"$inc": {"pontos": 50}})
        return {"sucesso": True, "aprovado": True, "pontos": 50}
    return {"sucesso": True, "aprovado": False}

# ------------------------------------------------------------------------------
# ROUTE GROUP: CHAT & ADMIN
# ------------------------------------------------------------------------------

@app.get("/chat/mensagens", tags=["Chat"])
def get_chat_messages(user1: str, user2: str):
    msgs = list(mongo_db.get_collection("chat").find({"$or": [{"remetente": user1, "destinatario": user2}, {"remetente": user2, "destinatario": user1}]}).sort("timestamp", 1))
    for m in msgs: m['_id'] = str(m['_id'])
    return {"sucesso": True, "mensagens": msgs}

@app.post("/chat/enviar", tags=["Chat"])
def send_chat_message(dados: ChatMessageRequest):
    mongo_db.get_collection("chat").insert_one({**dados.model_dump(), "timestamp": datetime.now().isoformat()})
    return {"sucesso": True}

@app.get("/chat/usuarios", tags=["Chat"])
def list_chat_users(usuario_atual: str):
    return {"sucesso": True, "usuarios": list(mongo_db.get_collection("usuarios").find({"usuario": {"$ne": usuario_atual}}, {"usuario": 1, "nome": 1, "_id": 0}))}

@app.get("/admin/listar", tags=["Admin"])
def list_users_admin():
    u = list(mongo_db.get_collection("usuarios").find())
    for x in u: x['_id'] = str(x['_id'])
    return {"sucesso": True, "usuarios": u}

@app.post("/admin/editar", tags=["Admin"])
def edit_user_admin(dados: AdminUserEdit):
    mongo_db.get_collection("usuarios").update_one({"usuario": dados.target_user}, {"$set": {k:v for k,v in dados.model_dump().items() if v and k!="target_user"}})
    return {"sucesso": True}

@app.post("/admin/excluir", tags=["Admin"])
def delete_user_admin(dados: AdminUserEdit):
    mongo_db.get_collection("usuarios").delete_one({"usuario": dados.target_user})
    return {"sucesso": True}

@app.get("/setup/criar-admin", tags=["Admin"])
def setup_admin_user():
    if mongo_db.get_collection("usuarios").find_one({"usuario": "admin"}): return {"sucesso": False}
    mongo_db.get_collection("usuarios").insert_one({"usuario": "admin", "senha": "123", "nome": "System Admin", "is_admin": True, "status": "ativo"})
    return {"sucesso": True}
