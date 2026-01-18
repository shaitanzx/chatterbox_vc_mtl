# File: engine.py
# Core TTS model loading and speech generation logic.
import os
os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"
import logging
import random
import numpy as np
import torch
import gc
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from typing import Optional, Tuple
from pathlib import Path
from chatterbox.tts import ChatterboxTTS  # Main TTS engine class
from chatterbox.models.s3gen.const import (
    S3GEN_SR,
)  # Default sample rate from the engine

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

from pathlib import Path
import torch
from safetensors.torch import load_file as load_safetensors
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers import MTLTokenizer
from chatterbox.mtl_tts import Conditionals, SUPPORTED_LANGUAGES
from chatterbox.vc import ChatterboxVC

# === ИМПОРТ ДЛЯ ЗАГРУЗКИ МОДЕЛЕЙ ===
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("huggingface_hub not available. Manual model download required.")

class PatchedChatterboxTTS(ChatterboxMultilingualTTS):
    """
    An inherited class that fixes the attention implementation issue by overriding
    the `from_local` class method.
    """
    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'PatchedChatterboxTTS':
        print("🚀 Using PatchedChatterboxTTS.from_local to load the model.")
        ckpt_dir = Path(ckpt_dir)

        # --- This is the original code from the library ---
        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", map_location=device, weights_only=True)
        )
        ve.to(device).eval()
        
        # --- OUR FIX IS APPLIED HERE ---
        # 1. Create the T3Config
        t3_config = T3Config.multilingual()
        
        # 2. Patch the config object directly
        # This part is slightly different because T3 doesn't take the config directly for attn
        # We'll go back to patching the LlamaConfig inside T3's init, but called from our override
        
        # Let's use the better approach of patching the T3's config logic
        # For simplicity, we directly recreate the T3 object with the fix logic
        from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
        from transformers import LlamaConfig, LlamaModel
        
        hp = T3Config.multilingual()
        cfg = LlamaConfig(**LLAMA_CONFIGS[hp.llama_config_name])
        cfg._attn_implementation = "eager" # Our patch
        
        # We need to manually recreate T3 since we can't inject the patched cfg easily
        # A simpler way is to just call the original method and then fix the model...
        # Let's try a cleaner override. We will replicate the method entirely.

        # The T3 class init needs to be fixed. So we create our own T3.
        class PatchedT3(T3):
            def __init__(self, hp=None):
                super().__init__(hp)
                # Override the transformer model with a patched config
                cfg = self.cfg
                cfg._attn_implementation = "eager"
                self.tfmr = LlamaModel(cfg)

        t3 = PatchedT3(T3Config.multilingual()) # Use our patched T3
        
        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", map_location=device, weights_only=True)
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)


# === ФУНКЦИИ ДЛЯ АВТОМАТИЧЕСКОЙ ЗАГРУЗКИ МОДЕЛЕЙ ===
def download_tts_models() -> bool:
    """Скачивает TTS модели в локальный кэш"""
    if not HF_AVAILABLE:
        logger.error("huggingface_hub not available. Cannot download TTS models.")
        logger.error("Please install: pip install huggingface-hub")
        return False
    
    logger.info("--- Starting TTS Models Download ---")
    
    # Получаем путь к кэшу из конфига
    model_cache_path_str = config_manager.get_string("paths.model_cache", "./model_cache")
    model_cache_path = Path(model_cache_path_str).resolve()
    
    # Получаем ID репозитория из конфига
    model_repo_id = config_manager.get_string("model.repo_id", "ResembleAI/chatterbox")
    
    logger.info(f"Target repository: {model_repo_id}")
    logger.info(f"Local directory: {model_cache_path}")
    
    # Создаем папку если не существует
    try:
        model_cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {model_cache_path}")
    except Exception as e:
        logger.error(f"Cannot create directory: {e}")
        return False
    
    # Список файлов для загрузки (такой же как в download_model.py)
    tts_model_files = [
        "ve.pt",  # Voice Encoder model
        "t3_cfg.pt",  # T3 model (Transformer Text-to-Token)
        "s3gen.pt",  # S3Gen model (Token-to-Waveform)
        "tokenizer.json",  # Text tokenizer configuration
        "conds.pt",  # Default conditioning data (e.g., for default voice)
        "Cangjie5_TC.json",
        "s3gen.safetensors",
        "grapheme_mtl_merged_expanded_v1.json",
        "t3_mtl23ls_v2.safetensors"
    ]
    
    success_count = 0
    total_files = len(tts_model_files)
    
    for filename in tts_model_files:
        logger.info(f"Downloading '{filename}'...")
        try:
            hf_hub_download(
                repo_id=model_repo_id,
                filename=filename,
                local_dir=model_cache_path,
                local_dir_use_symlinks=False,
                force_download=False,
                resume_download=True,
            )
            success_count += 1
            logger.info(f"✓ Successfully downloaded '{filename}'")
        except Exception as e:
            logger.error(f"✗ Failed to download '{filename}': {e}")
    
    if success_count == total_files:
        logger.info(f"✅ All {total_files} TTS files downloaded successfully!")
        return True
    else:
        logger.warning(f"⚠️ Downloaded {success_count}/{total_files} files")
        # Если скачано больше половины файлов, считаем успехом
        return success_count > total_files // 2

def download_ruaccent_models() -> bool:
    """Скачивает ruaccent модели в отдельную папку"""
    if not HF_AVAILABLE:
        logger.error("huggingface_hub not available. Cannot download ruaccent models.")
        logger.error("Please install: pip install huggingface-hub")
        return False
    
    logger.info("--- Starting RuAccent Models Download ---")
    
    # Получаем путь к кэшу ruaccent из конфига
    ruaccent_cache_path_str = config_manager.get_string("paths.ruaccent_cache", "./ruaccent_cache")
    ruaccent_cache_path = Path(ruaccent_cache_path_str).resolve()
    
    logger.info(f"RuAccent cache directory: {ruaccent_cache_path}")
    
    # Создаем папку если не существует
    try:
        ruaccent_cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured ruaccent directory exists: {ruaccent_cache_path}")
    except Exception as e:
        logger.error(f"Cannot create ruaccent directory: {e}")
        return False
    
    # Репозитории ruaccent моделей
    ruaccent_repos = {
        "Om1n": "ai-forever/ruaccent-om1n",
        "model": "ai-forever/ruaccent-model",
        "rules": "ai-forever/ruaccent-rules",
    }
    
    success_count = 0
    total_repos = len(ruaccent_repos)
    
    for model_name, repo_id in ruaccent_repos.items():
        logger.info(f"Downloading {model_name} from {repo_id}...")
        try:
            # Скачиваем всю модель (snapshot)
            snapshot_download(
                repo_id=repo_id,
                local_dir=ruaccent_cache_path / model_name,
                local_dir_use_symlinks=False,
                resume_download=True,
                force_download=False,
            )
            success_count += 1
            logger.info(f"✓ Successfully downloaded {model_name}")
        except Exception as e:
            logger.error(f"✗ Failed to download {model_name}: {e}")
    
    if success_count == total_repos:
        logger.info(f"✅ All {total_repos} ruaccent models downloaded successfully!")
        return True
    else:
        logger.warning(f"⚠️ Downloaded {success_count}/{total_repos} ruaccent models")
        return False

def check_and_download_all_models() -> bool:
    """Проверяет и скачивает все необходимые модели (TTS и ruaccent)"""
    logger.info("🔍 Checking for missing models...")
    
    # 1. Проверяем TTS модели
    model_cache_path_str = config_manager.get_string("paths.model_cache", "./model_cache")
    model_cache_path = Path(model_cache_path_str).resolve()
    
    tts_required_files = [
        "ve.pt",
        "t3_cfg.pt",
        "s3gen.pt",
        "tokenizer.json",
        "conds.pt",
        "Cangjie5_TC.json",
        "s3gen.safetensors",
        "grapheme_mtl_merged_expanded_v1.json",
        "t3_mtl23ls_v2.safetensors"
    ]
    
    missing_tts = []
    for file in tts_required_files:
        file_path = model_cache_path / file
        if not file_path.exists():
            missing_tts.append(file)
    
    # 2. Проверяем ruaccent модели
    ruaccent_cache_path_str = config_manager.get_string("paths.ruaccent_cache", "./ruaccent_cache")
    ruaccent_cache_path = Path(ruaccent_cache_path_str).resolve()
    
    ruaccent_required_dirs = ["Om1n", "model", "rules"]
    missing_ruaccent = []
    for dir_name in ruaccent_required_dirs:
        dir_path = ruaccent_cache_path / dir_name
        if not dir_path.exists():
            missing_ruaccent.append(dir_name)
    
    # 3. Логируем что отсутствует
    if missing_tts:
        logger.warning(f"Missing {len(missing_tts)} TTS files: {', '.join(missing_tts[:3])}{'...' if len(missing_tts) > 3 else ''}")
    
    if missing_ruaccent:
        logger.warning(f"Missing {len(missing_ruaccent)} ruaccent directories: {', '.join(missing_ruaccent)}")
    
    # 4. Если ничего не отсутствует - возвращаем успех
    if not missing_tts and not missing_ruaccent:
        logger.info("✅ All models are present.")
        return True
    
    # 5. Скачиваем если что-то отсутствует
    logger.info("🔄 Some models are missing. Starting automatic download...")
    
    if not HF_AVAILABLE:
        logger.error("❌ huggingface_hub not installed. Please install: pip install huggingface-hub")
        logger.error("Or download models manually.")
        return False
    
    all_success = True
    
    # Скачиваем TTS модели если нужно
    if missing_tts:
        logger.info("📥 Downloading TTS models...")
        tts_success = download_tts_models()
        if not tts_success:
            logger.error("❌ TTS models download failed")
            all_success = False
    
    # Скачиваем ruaccent модели если нужно
    if missing_ruaccent:
        logger.info("📥 Downloading ruaccent models...")
        ruaccent_success = download_ruaccent_models()
        if not ruaccent_success:
            logger.warning("⚠️ RuAccent models download failed (accentuation may not work)")
            # Не считаем это фатальной ошибкой, так как ruaccent не критичен
    
    if all_success:
        logger.info("✅ All required models downloaded successfully!")
    else:
        logger.warning("⚠️ Some models may not be fully downloaded")
    
    return all_success


# --- Global Module Variables ---
multilingual_model: Optional[PatchedChatterboxTTS] = None
MULTILINGUAL_MODEL_LOADED: bool = False
chatterbox_model: Optional[ChatterboxTTS] = None
MODEL_LOADED: bool = False
model_device: Optional[str] = None
vc_model: Optional[ChatterboxVC] = None
VC_MODEL_LOADED: bool = False

def set_seed(seed_value: int):
    """
    Sets the seed for torch, random, and numpy for reproducibility.
    This is called if a non-zero seed is provided for generation.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Global seed set to: {seed_value}")


def _test_cuda_functionality() -> bool:
    """
    Tests if CUDA is actually functional, not just available.

    Returns:
        bool: True if CUDA works, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.cuda()
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA functionality test failed: {e}")
        return False


def _test_mps_functionality() -> bool:
    """
    Tests if MPS is actually functional, not just available.

    Returns:
        bool: True if MPS works, False otherwise.
    """
    if not torch.backends.mps.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.to("mps")
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"MPS functionality test failed: {e}")
        return False


def load_model() -> bool:
    """
    Loads the multilingual TTS model by default from local cache.
    Automatically downloads models if missing.
    """
    global chatterbox_model, MODEL_LOADED, model_device, multilingual_model, MULTILINGUAL_MODEL_LOADED
    global vc_model, VC_MODEL_LOADED

    if MODEL_LOADED:
        logger.info("TTS model is already loaded.")
        return True

    try:
        # === ШАГ 1: ПРОВЕРЯЕМ И СКАЧИВАЕМ МОДЕЛИ (ЕСЛИ НУЖНО) ===
        logger.info("🔄 Checking and downloading models (if needed)...")
        if not check_and_download_all_models():
            logger.error("❌ Model download/check failed. Please check internet connection.")
            return False
        
        # === ШАГ 2: ОПРЕДЕЛЯЕМ УСТРОЙСТВО ===
        device_setting = config_manager.get_string("tts_engine.device", "auto")
        if device_setting == "auto":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA functionality test passed. Using CUDA.")
            elif _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS functionality test passed. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.info("CUDA and MPS not functional or not available. Using CPU.")

        elif device_setting == "cuda":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA requested and functional. Using CUDA.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "CUDA was requested in config but functionality test failed. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "mps":
            if _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS requested and functional. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "MPS was requested in config but functionality test failed. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "cpu":
            resolved_device_str = "cpu"
            logger.info("CPU device explicitly requested in config. Using CPU.")

        else:
            logger.warning(
                f"Invalid device setting '{device_setting}' in config. "
                f"Defaulting to auto-detection."
            )
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
            elif _test_mps_functionality():
                resolved_device_str = "mps"
            else:
                resolved_device_str = "cpu"
            logger.info(f"Auto-detection resolved to: {resolved_device_str}")
        
        model_device = resolved_device_str
        logger.info(f"Final device selection: {model_device}")

        # === ШАГ 3: ЗАГРУЖАЕМ TTS МОДЕЛЬ ИЗ ЛОКАЛЬНОГО КЭША ===
        model_cache_path_str = config_manager.get_string("paths.model_cache", "./model_cache")
        model_cache_path = Path(model_cache_path_str).resolve()
        
        logger.info(f"📁 Loading TTS model from local cache: {model_cache_path}")
        
        # Загружаем основную TTS модель
        multilingual_model = PatchedChatterboxTTS.from_local(
            ckpt_dir=model_cache_path, 
            device=model_device
        )
        
        chatterbox_model = multilingual_model
        MULTILINGUAL_MODEL_LOADED = True
        MODEL_LOADED = True

        logger.info(f"✅ PatchedChatterboxTTS model loaded successfully from local cache on {model_device}.")
        logger.info("Multilingual model is now the default for ALL languages.")
        
        # === ШАГ 4: ПРОВЕРЯЕМ НАЛИЧИЕ RUACCENT МОДЕЛЕЙ ===
        ruaccent_cache_path_str = config_manager.get_string("paths.ruaccent_cache", "./ruaccent_cache")
        ruaccent_cache_path = Path(ruaccent_cache_path_str).resolve()
        
        if all((ruaccent_cache_path / dir_name).exists() for dir_name in ["Om1n", "model", "rules"]):
            logger.info("✅ RuAccent models are available in cache")
        else:
            logger.warning("⚠️ RuAccent models not fully available. Accentuation may not work.")
        
        # === ШАГ 5: ПРОБУЕМ ЗАГРУЗИТЬ VC МОДЕЛЬ ===
        try:
            logger.info(f"Attempting to load Voice Conversion model from local cache...")
            
            # Проверяем необходимые для VC файлы
            vc_required_files = ["ve.pt"]  # Voice encoder is shared
            
            vc_missing = []
            for file in vc_required_files:
                if not (model_cache_path / file).exists():
                    vc_missing.append(file)
            
            if vc_missing:
                logger.warning(f"Missing VC model files: {vc_missing}")
                logger.warning("Voice Conversion will not be available.")
                vc_model = None
                VC_MODEL_LOADED = False
            else:
                # Загружаем VC модель
                vc_model = ChatterboxVC.from_local(
                    ckpt_dir=model_cache_path,
                    device=model_device
                )
                VC_MODEL_LOADED = True
                logger.info(f"✅ Voice Conversion model loaded successfully from local cache on {model_device}.")
                
        except Exception as vc_e:
            logger.warning(f"⚠️ Failed to load Voice Conversion model: {vc_e}")
            logger.warning("Voice Conversion tab will not be available.")
            vc_model = None
            VC_MODEL_LOADED = False

        return True

    except Exception as e:
        logger.error(f"❌ Error loading model: {e}", exc_info=True)
        multilingual_model = None
        chatterbox_model = None
        MULTILINGUAL_MODEL_LOADED = False
        MODEL_LOADED = False
        return False


def load_multilingual_model() -> bool:
    """
    Loads the multilingual TTS model from local cache.
    """
    global multilingual_model, MULTILINGUAL_MODEL_LOADED, model_device
    global chatterbox_model, MODEL_LOADED

    if MULTILINGUAL_MODEL_LOADED:
        logger.info("Multilingual TTS model is already loaded and set as default.")
        return True

    if model_device is None:
        logger.error("Main model device not determined. Load main model first.")
        return False

    if chatterbox_model is not None:
        logger.info("Unloading the standard ChatterboxTTS model to free up memory...")
        chatterbox_model = None
        MODEL_LOADED = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Standard model unloaded and memory cleared.")

    try:
        # Получаем путь к кэшу моделей
        model_cache_path_str = config_manager.get_string("paths.model_cache", "./model_cache")
        model_cache_path = Path(model_cache_path_str).resolve()
        
        logger.info(f"Loading multilingual model (PatchedChatterboxTTS) from local cache: {model_cache_path} on {model_device}...")

        multilingual_model = PatchedChatterboxTTS.from_local(
            ckpt_dir=model_cache_path, 
            device=model_device
        )

        chatterbox_model = multilingual_model

        MULTILINGUAL_MODEL_LOADED = True
        MODEL_LOADED = True

        logger.info(f"PatchedChatterboxTTS model loaded successfully from local cache on {model_device}.")
        logger.info("This model will now be used for ALL languages, including English.")
        return True

    except Exception as e:
        logger.error(f"Error loading multilingual model: {e}", exc_info=True)
        multilingual_model = None
        chatterbox_model = None
        MULTILINGUAL_MODEL_LOADED = False
        MODEL_LOADED = False
        return False


def load_vc_model() -> bool:
    """
    Loads the Voice Conversion model from local cache.
    """
    global vc_model, VC_MODEL_LOADED, model_device
    
    if VC_MODEL_LOADED:
        logger.info("Voice Conversion model is already loaded.")
        return True
    
    if model_device is None:
        logger.error("Main model device not determined. Load main model first.")
        return False
    
    try:
        # Получаем путь к кэшу моделей
        model_cache_path_str = config_manager.get_string("paths.model_cache", "./model_cache")
        model_cache_path = Path(model_cache_path_str).resolve()
        
        logger.info(f"Loading Voice Conversion model from local cache: {model_cache_path} on {model_device}...")
        
        # Загружаем VC модель из локального кэша
        vc_model = ChatterboxVC.from_local(
            ckpt_dir=model_cache_path,
            device=model_device
        )
        VC_MODEL_LOADED = True
        
        logger.info(f"Voice Conversion model loaded successfully from local cache on {model_device}.")
        return True
        
    except Exception as e:
        logger.error(f"Error loading Voice Conversion model: {e}", exc_info=True)
        vc_model = None
        VC_MODEL_LOADED = False
        return False


def get_or_load_vc_model() -> Optional[ChatterboxVC]:
    """
    Gets or loads the Voice Conversion model from local cache.
    """
    global vc_model, VC_MODEL_LOADED
    
    if not VC_MODEL_LOADED:
        if not load_vc_model():
            return None
    
    return vc_model


def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language: str = "en",
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Synthesizes audio from text using the currently loaded TTS model.
    If the multilingual model is loaded, it handles all languages.
    """
    global chatterbox_model, multilingual_model

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded. Cannot synthesize audio.")
        return None, None
    
    active_model = chatterbox_model

    try:
        if seed != 0:
            logger.info(f"Applying user-provided seed for generation: {seed}")
            set_seed(seed)
        else:
            logger.info("Using default (potentially random) generation behavior as seed is 0.")

        logger.debug(
            f"Synthesizing with params: audio_prompt='{audio_prompt_path}', temp={temperature}, "
            f"exag={exaggeration}, cfg_weight={cfg_weight}, seed_applied_globally_if_nonzero={seed}, "
            f"language={language}"
        )

        is_multilingual = isinstance(active_model, ChatterboxMultilingualTTS)
        
        if is_multilingual:
            logger.info(f"Synthesizing with multilingual model for language: {language}")
            wav_tensor = active_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                language_id=language,
            )
        else:
            logger.info("Synthesizing with standard English model.")
            wav_tensor = active_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        return wav_tensor, active_model.sr

    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}", exc_info=True)
        return None, None


def get_supported_languages() -> list:
    """
    Returns a list of all supported languages ​​for the UI.
    """
    langs = SUPPORTED_LANGUAGES
    if isinstance(langs, dict):
        return list(langs.keys())
    elif isinstance(langs, (list, tuple, set)):
        return list(langs)
    else:
        try:
            return list(getattr(langs, "keys")())
        except Exception:
            return ["en"]


# Функция для ручного запуска загрузки моделей (опционально)
def download_models_manually() -> bool:
    """
    Manual function to download all models.
    Can be called separately if needed.
    """
    logger.info("🚀 Manual model download started...")
    return check_and_download_all_models()


# Если файл запускается напрямую - скачиваем модели
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    print("=" * 60)
    print("🧠 Chatterbox TTS Model Downloader")
    print("=" * 60)
    
    if download_models_manually():
        print("\n✅ All models downloaded successfully!")
        print("You can now run the main application.")
    else:
        print("\n❌ Model download failed.")
        print("Please check your internet connection and try again.")
        exit(1)