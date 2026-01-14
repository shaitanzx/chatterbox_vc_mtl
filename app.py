# File: server_gradio.py
# Полная замена веб-интерфейса на Gradio с сохранением оригинальных названий функций
# Основано на server.py, script.js и index.html

import os
from pathlib import Path
import gradio as gr
import torch
import numpy as np
import tempfile
import time
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
import io
import shutil
import uuid
import librosa
import unicodedata
import re
from datetime import datetime

# --- ОРИГИНАЛЬНЫЕ ИМПОРТЫ ИЗ SERVER.PY ---
# Импортируем config_manager ПЕРВЫМ ДЕЛОМ
from config import (
    config_manager,
    get_host,
    get_port,
    get_log_file_path,
    get_output_path,
    get_reference_audio_path,
    get_predefined_voices_path,
    get_ui_title,
    get_gen_default_temperature,
    get_gen_default_exaggeration,
    get_gen_default_cfg_weight,
    get_gen_default_seed,
    get_gen_default_speed_factor,
    get_gen_default_language,
    get_audio_sample_rate,
    get_full_config_for_template,
    get_audio_output_format,
)

# Получаем путь к кэшу из конфигурации
model_cache_path = config_manager.get_path("paths.model_cache", "./model_cache", ensure_absolute=True)

# Устанавливаем переменные окружения ПЕРЕД любыми импортами huggingface
os.environ["HF_HOME"] = str(model_cache_path)
os.environ["HF_HUB_CACHE"] = str(model_cache_path)
os.environ["TRANSFORMERS_CACHE"] = str(model_cache_path)
os.environ["TORCH_HOME"] = str(model_cache_path)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(model_cache_path)
os.environ["XDG_CACHE_HOME"] = str(model_cache_path.parent)

# --- ИМПОРТЫ ИЗ ОРИГИНАЛЬНЫХ ФАЙЛОВ ---
import engine  # TTS Engine interface
from models import CustomTTSRequest  # Pydantic models
import utils  # Utility functions

from ruaccent import RUAccent

# --- Logging Configuration ---
log_file_path_obj = get_log_file_path()
log_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(str(log_file_path_obj), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- Global Variables ---
current_config = {}
currentUiState = {}
appPresets = []
initialReferenceFiles = []
initialPredefinedVoices = []
hideChunkWarning = False
hideGenerationWarning = False
currentVoiceMode = 'predefined'
isGenerating = False
wavesurfer = None
currentAudioBlobUrl = None

reference_playing_state = {"is_playing": False, "current_file": None}

# --- SUPPORTED LANGUAGES (из server.py) ---
SUPPORTED_LANGUAGES = [
    "en", "ru", "de", "fr", "es", "pt", "tr", "zh", "ja", "ko",
    "hi", "ar", "sv", "nl", "pl", "it", "fi", "no", "ms", "he",
    "el", "da", "sw"
]

LANGUAGE_LABELS = {
    'en': "English", 'ru': "Russian", 'de': "German", 'fr': "French", 'es': "Spanish",
    'pt': "Portuguese", 'tr': "Turkish", 'zh': "Chinese", 'ja': "Japanese", 'ko': "Korean",
    'hi': "Hindi", 'ar': "Arabic", 'sv': "Swedish", 'nl': "Dutch", 'pl': "Polish",
    'it': "Italian", 'fi': "Finnish", 'no': "Norwegian", 'ms': "Malay", 'he': "Hebrew",
    'el': "Greek", 'da': "Danish", 'sw': "Swahili"
}
# Создаём обратное отображение: "Russian" → "ru"
DISPLAY_TO_CODE = {name: code for code, name in LANGUAGE_LABELS.items()}

def extract_language_code(display_text: str) -> str:
    """
    Извлекает код языка из строки вида 'Russian (ru)' или просто 'Russian'.
    Возвращает код (например, 'ru') или исходную строку, если не найдено.
    """
    # Убираем скобки и всё, что в них — оставляем только название
    if " (" in display_text and display_text.endswith(")"):
        lang_name = display_text.split(" (")[0]
    else:
        lang_name = display_text  # на случай, если скобок нет

    return DISPLAY_TO_CODE.get(lang_name, display_text)

# --- Accentuation Support (из server.py) ---
try:
    accent_model = RUAccent()
    accent_model.load()
except Exception as e:
    logger.error(f"Failed to initialize RUAccent: {e}")
    accent_model = None

def convert_plus_to_accent(text: str) -> str:
    """Convert ruaccent '+vowel' markers to vowel with combining acute"""
    replacements = {
        '+а': 'а́', '+А': 'А́', '+е': 'е́', '+Е': 'Е́',
        '+ё': 'ё́', '+Ё': 'Ё́', '+и': 'и́', '+И': 'И́',
        '+о': 'о́', '+О': 'О́', '+у': 'у́', '+У': 'У́',
        '+ы': 'ы́', '+Ы': 'Ы́', '+э': 'э́', '+Э': 'Э́',
        '+ю': 'ю́', '+Ю': 'Ю́', '+я': 'я́', '+Я': 'Я́',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def load_custom_accents() -> Dict[str, str]:
    """Load custom accent fixes from YAML and dict files (из server.py)"""
    yaml_fixes = {}
    path = Path("accent_fixes.yaml")
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(k, str) and isinstance(v, str):
                            yaml_fixes[k.strip()] = v.strip()
        except Exception as e:
            logger.error(f"Failed to load accent_fixes.yaml: {e}")
    
    logger.info(f"Loaded {len(yaml_fixes)} custom accent fixes")
    return yaml_fixes

CUSTOM_ACCENTS = load_custom_accents()

def apply_custom_fixes(text: str) -> str:
    """Apply custom accent fixes (из server.py)"""
    text = unicodedata.normalize("NFC", text)
    items = [(k, v) for k, v in CUSTOM_ACCENTS.items() 
             if isinstance(k, str) and isinstance(v, str)]
    items.sort(key=lambda kv: len(kv[0]), reverse=True)
    for wrong, correct in items:
        text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
    return text

# --- ОРИГИНАЛЬНЫЕ ФУНКЦИИ ИЗ SERVER.PY ---

def on_accent_click(text: str):
    """Original from server.py - accentuate Russian text"""
    if accent_model is None:
        gr.Error("⚠️ RUAccent model not loaded")
        return text
    
    try:
        raw_text = accent_model.process_all(text)
        accented_text = convert_plus_to_accent(raw_text)
        accented_text = apply_custom_fixes(accented_text)
        gr.Info("✅ Stresses are placed!")
        return accented_text
    except Exception as e:
        logger.error(f"Error in accentuate_text_endpoint: {e}", exc_info=True)
        gr.Error(f"⚠️ Accentuation failed: {str(e)}")
        return text
#def on_accent_click(text: str) -> Tuple[str, Dict[str, str]]:
#    """Обработчик кнопки Stress (аналог из script.js)"""
#    if not text:
#        return text, show_notification("No text to accentuate", "warning")
    
#    result = await accentuate_text_endpoint(text)
#    if result.get("status") == "success":
#        return result["accented_text"], show_notification("✅ Stresses are placed!", "success")
#    else:
#        return text, show_notification(f"⚠️ {result.get('detail', 'Error')}", "error")



def get_ui_initial_data() -> Dict[str, Any]:
    """Original from server.py - get initial UI data"""
    logger.info("+++++++++Request for initial UI data")
    try:
        full_config = get_full_config_for_template()
        print('++++++++',full_config)
        reference_files = utils.get_valid_reference_files()
        print('++++++++',reference_files)
        predefined_voices = utils.get_predefined_voices()
        print('++++++++',predefined_voices)
        
        # Load presets
        loaded_presets = []
        ui_static_path = Path(__file__).parent
        presets_file = ui_static_path / "presets.yaml"
        if presets_file.exists():
            print ('-----------------------------------------')
            with open(presets_file, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)
                if isinstance(yaml_content, list):
                    loaded_presets = yaml_content
        print ('qqqqqqqqqqqqqqqqqqqqqqqqq', loaded_presets)
        return {
            "config": full_config,
            "reference_files": reference_files,
            "predefined_voices": predefined_voices,
            "presets": loaded_presets,
            "languages": SUPPORTED_LANGUAGES,
        }
    except Exception as e:
        logger.error(f"Error preparing initial UI data: {e}", exc_info=True)
        return {"error": "Failed to load initial data"}

def save_settings_endpoint(config_tts_engine_device, reference_audio_path, predefined_voices_path, default_voice_id, default_voice_clone,
                config_paths_model_cache,config_paths_output,temperature_slider, exaggeration_slider,cfg_weight_slider, seed_input,
                speed_factor_slider, language, config_audio_output_format,config_audio_output_sample_rate):
    """Original from server.py - save settings"""
    logger.info("Saving settings")
    try:
        settings_data = {
            "tts_engine": {
                "device": config_tts_engine_device,
                "reference_audio_path": reference_audio_path,
                "predefined_voices_path": predefined_voices_path,
                "default_voice_id": default_voice_id,
                "default_voice_clone": default_voice_clone
            },
            "paths": {
                "model_cache": config_paths_model_cache,
                "output": config_paths_output
            },
            "generation_defaults": {
                "temperature": temperature_slider,
                "exaggeration": exaggeration_slider,
                "cfg_weight": cfg_weight_slider,
                "seed": seed_input,
                "speed_factor": speed_factor_slider,
                "language": extract_language_code(language)
            },
            "audio_output": {
                "format": config_audio_output_format,
                "sample_rate": config_audio_output_sample_rate
            }
        }

        if config_manager.update_and_save(settings_data):
            restart_needed = any(
                key in settings_data
                for key in ["server", "tts_engine", "paths", "model"]
            )
            gr.Info("Settings saved successfully.")
            if restart_needed:
                gr.Info("A server restart may be required.")
            return
        else:
            return 
    except Exception as e:
        logger.error(f"Error saving settings: {e}", exc_info=True)
        return 

async def reset_settings_endpoint() -> Dict[str, Any]:
    """Original from server.py - reset settings"""
    logger.warning("Resetting all configurations to default values")
    try:
        if config_manager.reset_and_save():
            return {
                "message": "Configuration reset to defaults. Please reload.",
                "restart_needed": True
            }
        else:
            return {"error": "Failed to reset configuration"}
    except Exception as e:
        logger.error(f"Error resetting settings: {e}", exc_info=True)
        return {"error": f"Internal error: {str(e)}"}

async def get_reference_files_api() -> List[str]:
    """Original from server.py - get reference files"""
    return utils.get_valid_reference_files()

async def get_predefined_voices_api() -> List[Dict[str, str]]:
    """Original from server.py - get predefined voices"""
    return utils.get_predefined_voices()

def upload_reference_audio_endpoint(files: List[gr.File]) -> Dict[str, Any]:
    #Original from server.py - upload reference audio
    ref_path = get_reference_audio_path(ensure_absolute=True)
    uploaded_filenames = []
    errors = []
    
    for file_info in files:
        if not file_info:
            continue
            
        # Extract filename from Gradio file object
        filename = os.path.basename(file_info)
        safe_filename = utils.sanitize_filename(filename)
        destination_path = ref_path / safe_filename
        
        try:
            if destination_path.exists():
                logger.info(f"File '{safe_filename}' already exists.")
                uploaded_filenames.append(safe_filename)
                continue
            
            # Copy file
            shutil.copy2(file_info, destination_path)
            logger.info(f"Saved uploaded file to: {destination_path}")
            
            # Validate
            max_duration = config_manager.get_int(
                "audio_output.max_reference_duration_sec", 600
            )
            is_valid, validation_msg = utils.validate_reference_audio(
                destination_path, max_duration
            )
            if not is_valid:
                destination_path.unlink(missing_ok=True)
                errors.append({"filename": safe_filename, "error": validation_msg})
            else:
                uploaded_filenames.append(safe_filename)
                
        except Exception as e:
            errors.append({"filename": filename, "error": str(e)})
    
    all_files = utils.get_valid_reference_files()
    return {
        "message": f"Processed {len(files)} file(s)",
        "uploaded_files": uploaded_filenames,
        "all_reference_files": all_files,
        "errors": errors
    }
"""
async def upload_predefined_voice_endpoint(files: List[gr.File]) -> Dict[str, Any]:
    #Original from server.py - upload predefined voice
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)
    uploaded_filenames = []
    errors = []
    
    for file_info in files:
        if not file_info:
            continue
            
        filename = os.path.basename(file_info)
        safe_filename = utils.sanitize_filename(filename)
        destination_path = predefined_voices_path / safe_filename
        
        try:
            if destination_path.exists():
                logger.info(f"Voice file '{safe_filename}' already exists.")
                uploaded_filenames.append(safe_filename)
                continue
            
            shutil.copy2(file_info, destination_path)
            
            # Basic validation
            is_valid, validation_msg = utils.validate_reference_audio(
                destination_path, max_duration_sec=None
            )
            if not is_valid:
                destination_path.unlink(missing_ok=True)
                errors.append({"filename": safe_filename, "error": validation_msg})
            else:
                uploaded_filenames.append(safe_filename)
                
        except Exception as e:
            errors.append({"filename": filename, "error": str(e)})
    
    all_voices = utils.get_predefined_voices()
    return {
        "message": f"Processed {len(files)} voice file(s)",
        "uploaded_files": uploaded_filenames,
        "all_predefined_voices": all_voices,
        "errors": errors
    }
"""
# --- ОСНОВНАЯ TTS ФУНКЦИЯ (аналог custom_tts_endpoint из server.py) ---
def custom_tts_endpoint(
    text: str,
    voice_mode: str,
    predefined_voice_id: Optional[str] = None,
    reference_audio_filename: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    speed_factor: float = 1.0,
    seed: int = 0,
    language: str = "en",
    split_text: bool = False,
    chunk_size: int = 120,
    output_format: str = "mp3",
    audio_name: Optional[str] = None
) -> Tuple[Optional[str], str]:  # (audio_file_path, status_message)
    """Original TTS generation function from server.py"""
    
    global isGenerating
    
    if isGenerating:
        return None, "Generation is already in progress."
    
    isGenerating = True
    start_time = time.time()
    
    try:
        # Проверка модели (аналог строки 597 server.py)
        if not engine.MODEL_LOADED:
            return None, "TTS engine model is not currently loaded or available."
        
        # Определение пути к аудиопромпту (аналог строк 609-648 server.py)
        audio_prompt_path = None
        if voice_mode == "predefined":
            if not predefined_voice_id:
                return None, "Missing 'predefined_voice_id' for 'predefined' voice mode."
            voices_dir = get_predefined_voices_path(ensure_absolute=True)
            potential_path = voices_dir / predefined_voice_id
            if not potential_path.is_file():
                return None, f"Predefined voice file '{predefined_voice_id}' not found."
            audio_prompt_path = potential_path
            
        elif voice_mode == "clone":
            if not reference_audio_filename:
                return None, "Missing 'reference_audio_filename' for 'clone' voice mode."
            ref_dir = get_reference_audio_path(ensure_absolute=True)
            potential_path = ref_dir / reference_audio_filename
            if not potential_path.is_file():
                return None, f"Reference audio file '{reference_audio_filename}' not found."
            max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 600)
            is_valid, msg = utils.validate_reference_audio(potential_path, max_dur)
            if not is_valid:
                return None, f"Invalid reference audio: {msg}"
            audio_prompt_path = potential_path
        
        # Разделение текста на чанки (аналог строк 666-680 server.py)
        if split_text and len(text) > (chunk_size * 1.5):
            text_chunks = utils.chunk_text_by_sentences(text, chunk_size)
        else:
            text_chunks = [text]
        
        # Генерация аудио по чанкам (аналог строк 686-726 server.py)
        all_audio_segments_np = []
        engine_output_sample_rate = None
        
        for i, chunk in enumerate(text_chunks):
            try:
                chunk_audio_tensor, chunk_sr = engine.synthesize(
                    text=chunk,
                    audio_prompt_path=str(audio_prompt_path) if audio_prompt_path else None,
                    temperature=temperature,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    seed=seed,
                    language=language,
                )
                
                if chunk_audio_tensor is None or chunk_sr is None:
                    return None, f"TTS engine failed to synthesize audio for chunk {i+1}."
                
                if engine_output_sample_rate is None:
                    engine_output_sample_rate = chunk_sr
                
                processed_audio_np = chunk_audio_tensor.cpu().numpy().squeeze()
                all_audio_segments_np.append(processed_audio_np)
                
            except Exception as e:
                return None, f"Error processing audio chunk {i+1}: {str(e)}"
        
        # Объединение и обработка аудио (аналог строк 728-800 server.py)
        if not all_audio_segments_np:
            return None, "Audio generation resulted in no output."
        
        final_audio_np = (
            np.concatenate(all_audio_segments_np)
            if len(all_audio_segments_np) > 1
            else all_audio_segments_np[0]
        )
        
        # Применение аудио-обработки
        if config_manager.get_bool("audio_processing.enable_silence_trimming", False):
            final_audio_np = utils.trim_lead_trail_silence(
                final_audio_np, engine_output_sample_rate
            )
        
        # Применение скорости
        if speed_factor != 1.0:
            try:
                import torch
                final_audio_tensor = torch.from_numpy(final_audio_np.astype(np.float32))
                
                # Используем оригинальную функцию из utils
                final_audio_tensor, _ = utils.apply_speed_factor(
                    final_audio_tensor, 
                    engine_output_sample_rate, 
                    speed_factor
                )
                final_audio_np = final_audio_tensor.cpu().numpy()
            except Exception as e:
                logger.error(f"Failed to apply speed factor: {e}", exc_info=True)
        
        # Кодирование аудио (аналог строк 802-815 server.py)
        output_format_str = output_format if output_format else get_audio_output_format()
        final_output_sample_rate = get_audio_sample_rate()
        
        encoded_audio_bytes = utils.encode_audio(
            audio_array=final_audio_np,
            sample_rate=engine_output_sample_rate,
            output_format=output_format_str,
            target_sample_rate=final_output_sample_rate,
        )
        
        if encoded_audio_bytes is None:
            return None, "Failed to encode audio to requested format."
        
        # Сохранение файла (аналог строк 817-840 server.py)
        outputs_dir = get_output_path(ensure_absolute=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        suggested_filename_base = audio_name or f"tts_output_{timestamp_str}"
        file_name = utils.sanitize_filename(f"{suggested_filename_base}.{output_format_str}")
        file_path = outputs_dir / file_name
        
        with open(file_path, "wb") as f:
            f.write(encoded_audio_bytes)
        
        generation_time = time.time() - start_time
        
        return str(file_path), f"✅ Audio generated successfully in {generation_time:.2f}s"
        
    except Exception as e:
        logger.error(f"Error in TTS generation: {e}", exc_info=True)
        return None, f"❌ Error: {str(e)}"
    finally:
        isGenerating = False

# --- ФУНКЦИИ ИЗ SCRIPT.JS (адаптированные для Gradio) ---

def show_notification(message: str, type: str = "info") -> Dict[str, str]:
    """Аналог showNotification из script.js"""
    icon = {
        "success": "✅",
        "error": "❌", 
        "warning": "⚠️",
        "info": "ℹ️"
    }.get(type, "ℹ️")
    
    return {
        "message": f"{icon} {message}",
        "type": type,
        "timestamp": time.strftime("%H:%M:%S")
    }

def getTTSFormData(
    text: str,
    voice_mode: str,
    predefined_voice: str,
    reference_file: str,
    temperature: float,
    exaggeration: float,
    cfg_weight: float,
    speed_factor: float,
    seed: int,
    language: str,
    split_text: bool,
    chunk_size: int,
    output_format: str
) -> Dict[str, Any]:
    """Аналог getTTSFormData из script.js"""
    return {
        "text": text,
        "temperature": temperature,
        "exaggeration": exaggeration,
        "cfg_weight": cfg_weight,
        "speed_factor": speed_factor,
        "seed": seed,
        "language": language,
        "voice_mode": voice_mode,
        "split_text": split_text,
        "chunk_size": chunk_size,
        "output_format": output_format,
        "predefined_voice_id": predefined_voice if voice_mode == "predefined" and predefined_voice != "none" else None,
        "reference_audio_filename": reference_file if voice_mode == "clone" and reference_file != "none" else None
    }

def toggleVoiceOptionsDisplay(voice_mode: str) -> Tuple[Dict, Dict]:
    """Аналог toggleVoiceOptionsDisplay из script.js"""
    return (
        gr.update(visible=(voice_mode == "predefined")),
        gr.update(visible=(voice_mode == "clone"))
    )

def toggleChunkControlsVisibility(split_enabled: bool) -> Tuple[Dict, Dict]:
    """Аналог toggleChunkControlsVisibility из script.js"""
    return (
        gr.update(visible=split_enabled),
        gr.update(visible=split_enabled)
    )

def updateSpeedFactorWarning(speed_factor: float) -> str:
    """Аналог updateSpeedFactorWarning из script.js"""
    if speed_factor != 1.0:
        return f"⚠️ Speed factor is {speed_factor}. Normal is 1.0"
    return ""

def populatePredefinedVoices() -> List[str]:
    """Аналог populatePredefinedVoices из script.js"""
    voices = utils.get_predefined_voices()
    return [voice.get("filename", "") for voice in voices]

def populateReferenceFiles() -> List[str]:
    """Аналог populateReferenceFiles из script.js"""
    files = utils.get_valid_reference_files()
    return files

def populatePresets() -> List[Dict[str, Any]]:
    """Аналог populatePresets из script.js"""
    ui_static_path = Path(__file__).parent / "ui"
    presets_file = ui_static_path / "presets.yaml"
    if presets_file.exists():
        with open(presets_file, "r", encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f)
            if isinstance(yaml_content, list):
                return yaml_content
    return []

def applyPreset(preset_name: str, presets: List[Dict[str, Any]]) -> tuple:
    # Поиск пресета
    for preset in presets:
        if preset.get("name") == preset_name:
            # Извлекаем prompt (может быть на верхнем уровне)
            
            
            # Извлекаем параметры из вложенного словаря 'params'
            params = preset.get("params", {})
            
            temperature = float(params.get("temperature", 0.7))
            exaggeration = float(params.get("exaggeration", 1.0))
            cfg_weight = float(params.get("cfg_weight", 7.0))
            speed_factor = float(params.get("speed_factor", 1.0))
            seed = int(params.get("seed", -1))
            
            return (temperature, exaggeration, cfg_weight, speed_factor, seed)
    
    # Если пресет не найден — значения по умолчанию
    return (0.7, 1.0, 7.0, 1.0, -1)

# --- ОБРАБОТЧИКИ СОБЫТИЙ КНОПОК (аналог событий из script.js) ---

def on_generate_click(
    text: str,
    voice_mode: str,
    predefined_voice: str,
    reference_file: str,
    temperature: float,
    exaggeration: float,
    cfg_weight: float,
    speed_factor: float,
    seed: int,
    language: str,
    split_text: bool,
    chunk_size: int,
    output_format: str,
    audio_name: str
) -> Tuple[Optional[str], str, Dict[str, str]]:
    """Основной обработчик кнопки Generate (аналог из script.js)"""
    
    # Валидация (аналог строк 545-560 script.js)
    if not text or text.strip() == "":
        return None
    
    if voice_mode == "predefined" and predefined_voice == "none":
        return None
    
    if voice_mode == "clone" and reference_file == "none":
        return None
    
    # Проверка предупреждений (аналог строк 562-570 script.js)
    # (в Gradio можно добавить чекбоксы для отключения предупреждений)
    
    # Вызов TTS генерации
    language=extract_language_code(language)
    audio_file, message = custom_tts_endpoint(
        text=text,
        voice_mode=voice_mode,
        predefined_voice_id=predefined_voice if predefined_voice != "none" else None,
        reference_audio_filename=reference_file if reference_file != "none" else None,
        temperature=temperature,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        speed_factor=speed_factor,
        seed=seed,
        language=language,
        split_text=split_text,
        chunk_size=chunk_size,
        output_format=output_format,
        audio_name=audio_name
    )
    gr.Info(message)
    return gr.update (value=audio_file, visible=True)
    #if audio_file:
    #    notification = show_notification("Audio generated successfully!", "success")
    #    return audio_file, f"✅ {message}", notification
    #else:
    #    notification = show_notification(f"Generation failed: {message}", "error")
    #    return None, f"❌ {message}", notification


def on_text_input(text: str) -> str:
    """Обработчик ввода текста (аналог из script.js)"""
    return str(len(text))

async def on_restart_click() -> Dict[str, str]:
    """Обработчик кнопки Restart Server (аналог из script.js)"""
    # В Gradio просто показываем сообщение
    return show_notification("🔄 Server restart initiated...", "info")

def on_reference_upload(files: List[gr.File]):
    """
    Обработчик загрузки референсных файлов.
    Автоматически обновляет список файлов после загрузки.
    """
    #if not files:
    #    return populateReferenceFiles(), show_notification("⚠️ No files selected", "warning")
    
    try:
        # Вызываем оригинальную функцию загрузки
        result =  upload_reference_audio_endpoint(files)
        
        #if "errors" in result and result["errors"]:
        #    error_msg = result["errors"][0].get("error", "Upload failed")
        #    return populateReferenceFiles(), show_notification(f"❌ {error_msg}", "error")
        
        # Получаем обновленный список файлов
        all_files = result.get("all_reference_files", [])
        uploaded_files = result.get("uploaded_files", [])
        
        if uploaded_files:
            # Выбираем первый загруженный файл по умолчанию
            default_selection = uploaded_files[0] if uploaded_files else "none"
            updated_options = all_files

            #notification = show_notification(
            #    f"✅ Uploaded: {', '.join(uploaded_files[:3])}" + 
            #    ("..." if len(uploaded_files) > 3 else ""),
            #    "success"
            #)
            
            return gr.update(choices=updated_options,value=default_selection)
        else:
            return gr.update(choices=populateReferenceFiles())
            
    except Exception as e:
        logger.error(f"Error in reference upload: {e}", exc_info=True)
        return populateReferenceFiles(), show_notification(f"❌ Upload failed: {str(e)}", "error")
def toggle_voice_audio(selected_file: str, voice_mode: str) -> Tuple[Optional[str], str, Dict, Dict]:
    """
    Универсальная функция для воспроизведения файлов из обоих режимов.
    voice_mode: "predefined" или "clone"
    """
    global reference_playing_state
    
    if not selected_file:
        gr.Warning("⚠️ Please select a file")
        return None, "▶️ Play/Stop", gr.update(visible=False), gr.update(visible=False)
    
    # Определяем путь в зависимости от режима
    if voice_mode == "predefined":
        base_path = get_predefined_voices_path(ensure_absolute=True)
    else:  # clone
        base_path = get_reference_audio_path(ensure_absolute=True)
    
    file_path = base_path / selected_file
    
    # Проверяем существует ли файл
    if not file_path.exists():
        gr.Error(f"❌ File not found: {selected_file}")
        return None, "▶️ Play/Stop", gr.update(visible=False), gr.update(visible=False)
    
    # Создаем уникальный ключ для файла
    current_key = f"{voice_mode}_{selected_file}"
    
    # Если уже воспроизводится этот файл - останавливаем
    if reference_playing_state["is_playing"] and reference_playing_state["current_key"] == current_key:
        reference_playing_state = {"is_playing": False, "current_key": None}
        gr.Info(f"⏸️ Stopped: {selected_file}")
        return None, "▶️ Play/Stop", gr.update(visible=False), gr.update(visible=False)
    
    # Начинаем воспроизведение
    reference_playing_state = {"is_playing": True, "current_key": current_key}
    gr.Info(f"🎵 Playing: {selected_file}")
    
    return (
        str(file_path),  # путь к файлу
        "⏸️ Play/Stop",  # текст кнопки
        gr.update(visible=True),  # показываем плеер
        gr.update(value=str(file_path), autoplay=True)  # устанавливаем файл и автозапуск
    )
def reset_playback_on_mode_change(voice_mode: str) -> Tuple[str, str, Dict]:
    """
    Сбрасывает воспроизведение при смене режима голоса.
    """
    global reference_playing_state
    reference_playing_state = {"is_playing": False, "current_key": None}
    return "▶️ Play/Stop", "▶️ Play/Stop", gr.update(visible=False)
def voice_conversion(input_audio_path, target_voice_audio_path, chunk_sec=60, overlap_sec=0.1, disable_watermark=True, pitch_shift=0):
    vc_model = get_or_load_vc_model()
    model_sr = vc_model.sr

    wav, sr = sf.read(input_audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != model_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=model_sr)
        sr = model_sr

    total_sec = len(wav) / model_sr

    if total_sec <= chunk_sec:
        wav_out = vc_model.generate(
            input_audio_path,
            target_voice_path=target_voice_audio_path,
            apply_watermark=not disable_watermark,
            pitch_shift=pitch_shift
        )
        out_wav = wav_out.squeeze(0).numpy()
        return model_sr, out_wav

    # chunking logic for long files
    chunk_samples = int(chunk_sec * model_sr)
    overlap_samples = int(overlap_sec * model_sr)
    step_samples = chunk_samples - overlap_samples

    out_chunks = []
    for start in range(0, len(wav), step_samples):
        end = min(start + chunk_samples, len(wav))
        chunk = wav[start:end]
        temp_chunk_path = f"temp_vc_chunk_{start}_{end}.wav"
        sf.write(temp_chunk_path, chunk, model_sr)
        out_chunk = vc_model.generate(
            temp_chunk_path,
            target_voice_path=target_voice_audio_path,
            apply_watermark=not disable_watermark,
            pitch_shift=pitch_shift
        )
        out_chunk_np = out_chunk.squeeze(0).numpy()
        out_chunks.append(out_chunk_np)
        os.remove(temp_chunk_path)

    # Crossfade join as before...
    result = out_chunks[0]
    for i in range(1, len(out_chunks)):
        overlap = min(overlap_samples, len(out_chunks[i]), len(result))
        if overlap > 0:
            fade_out = np.linspace(1, 0, overlap)
            fade_in = np.linspace(0, 1, overlap)
            result[-overlap:] = result[-overlap:] * fade_out + out_chunks[i][:overlap] * fade_in
            result = np.concatenate([result, out_chunks[i][overlap:]])
        else:
            result = np.concatenate([result, out_chunks[i]])
    return model_sr, result

#def on_reference_selection_change(selected_file: str) -> Tuple[str, Dict, Dict]:
#    """
#    При изменении выбора файла в dropdown останавливает воспроизведение.
#    """
#    global reference_playing_state
#    reference_playing_state = {"is_playing": False, "current_file": None}
#    return "▶️ Play/Stop", gr.update(visible=False, autoplay=False), gr.update(visible=False)

# --- СОЗДАНИЕ GRADIO ИНТЕРФЕЙСА ---

def create_gradio_interface():
    """Создание полного интерфейса Gradio на основе index.html"""
    
    # Загружаем начальные данные
    initial_data = get_ui_initial_data()
    print('zzzzzzzzzzzzzzz',initial_data)
    if isinstance(initial_data, dict):
        current_config = initial_data.get("config", {})
        appPresets = initial_data.get("presets", [])
        languages = initial_data.get("languages", ["en"])
    else:
        current_config = {}
        appPresets = []
        languages = ["en"]
    
    # Генерация опций для языков
    language_options = []
    for lang_code in languages:
        label = LANGUAGE_LABELS.get(lang_code, lang_code)
        language_options.append(f"{label} ({lang_code})")
    
    with gr.Blocks(title="Chatterbox Server") as demo:
        
        # Заголовок (аналог navbar из index.html)
        gr.Markdown(f"# 🎤 {get_ui_title()}")
        with gr.Tabs():

            # === VC TAB: Voice Conversion Tab ===
            with gr.Tab("🎤 Voice Conversion (VC)"):
                gr.Markdown("## Voice Conversion\nConvert one speaker's voice to sound like another speaker using a target/reference voice audio.")
                with gr.Row():
                    vc_input_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input Audio (to convert)")
                    vc_target_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Target Voice Audio")
                vc_pitch_shift = gr.Number(value=0, label="Pitch", step=0.5, interactive=True)
                disable_watermark_checkbox = gr.Checkbox(label="Disable Perth Watermark", value=True, visible=False)
                vc_convert_btn = gr.Button("Run Voice Conversion")
                vc_output_files = gr.Files(label="Converted VC Audio File(s)")
                vc_output_audio = gr.Audio(label="VC Output Preview", interactive=True)

                def _vc_wrapper(input_audio_path, target_voice_audio_path, disable_watermark, pitch_shift):
                    # Defensive: None means Gradio didn't get file yet
                    if not input_audio_path or not os.path.exists(input_audio_path):
                        raise gr.Error("Please upload or record an input audio file.")
                    if not target_voice_audio_path or not os.path.exists(target_voice_audio_path):
                        raise gr.Error("Please upload or record a target/reference voice audio file.")

                    sr, out_wav = voice_conversion(
                        input_audio_path,
                        target_voice_audio_path,
                        disable_watermark=disable_watermark,
                        pitch_shift=pitch_shift
                    )
                    os.makedirs("output", exist_ok=True)
                    base = os.path.splitext(os.path.basename(input_audio_path))[0]
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")[:-3]
                    out_path = f"output/{base}_vc_{timestamp}.wav"
                    sf.write(out_path, out_wav, sr)
                    return [out_path], out_path  # Files and preview

                vc_convert_btn.click(
                    fn=_vc_wrapper,
                    inputs=[vc_input_audio, vc_target_audio, disable_watermark_checkbox, vc_pitch_shift],
                    outputs=[vc_output_files, vc_output_audio],
                )

            with gr.Tab("🎵 MTL Generation"):
        
                with gr.Row():
                        gr.Markdown("### Text to synthesize")
                with gr.Row():
                        gr.Markdown("Enter the text you want to convert to speech. For audiobooks, you can paste long chapters.")
                with gr.Row():    
                        text_area = gr.Textbox(
                            label="",
                            value=current_config.get("ui_state", {}).get("last_text", "\u041A\u043E\u0433\u0434\u0430\u0301-\u0442\u043E \u0433\u0435\u0440\
    \u043E\u0301\u0439 \u0431\u044B\u043B \u043E\u0301\u0444\u0438\u0441\u043D\u044B\
    \u043C \u0441\u043E\u0442\u0440\u0443\u0301\u0434\u043D\u0438\u043A\u043E\u043C\
    , \u043D\u0435 \u0448\u0438\u0301\u0431\u043A\u043E \u0440\u0432\u0430\u0301\u0432\
    \u0448\u0438\u043C\u0441\u044F \u0447\u0435\u0433\u043E\u0301-\u0442\u043E \u0434\
    \u043E\u0441\u0442\u0438\u0433\u0430\u0301\u0442\u044C \u0432 \u0436\u0438\u0301\
    \u0437\u043D\u0438."),
                            placeholder="Enter text here...",
                            lines=8,
                            max_lines=15,
                            show_copy_button=False,
                            elem_id="text"
                        )
                with gr.Row():        
                        char_count = gr.Textbox(
                                label="Characters",
                                value="94",
                                interactive=False,
                                scale=1,
                                elem_id="char-count"
                            )   
                    
                # Кнопки действий (аналог flex-wrap из index.html)
                with gr.Row():
                        generate_btn = gr.Button("🎵 Generate Speech",elem_id="generate-btn")
                        accent_btn = gr.Button("🇷🇺 Stress")
                with gr.Row():        
                    # Уведомления (аналог popup-msg)
                        notification_display = gr.JSON(
                            label="Notifications",
                            value={},
                            visible=False
                        )
                with gr.Group():
                    with gr.Row():                
                # Настройки разделения текста (аналог Split text into chunks)
                            split_text_toggle = gr.Checkbox(
                                label="Split text into chunks",
                                value=True
                            )
                    with gr.Row():            
                            chunk_size_slider = gr.Slider(
                                minimum=50,
                                maximum=1000,
                                value=120,
                                step=10,
                                label="Chunk Size",
                                visible=True
                            )
                with gr.Row():                
                    # Режим голоса (аналог Voice Mode)
                        with gr.Accordion("🗣 Voice Mode", open=True):
                            voice_mode_radio = gr.Radio(
                                choices=["predefined", "clone"],
                                value="predefined",
                                label="Select Voice Mode"
                            )
                    
                    # Предопределенные голоса
                            with gr.Group(visible=True) as predefined_group:
                                with gr.Row():
                                    predefined_voice_select = gr.Dropdown(
                                        choices=populatePredefinedVoices(),
                                        value=current_config.get("ui_state", {}).get("last_predefined_voice", "none"),
                                        label="Predefined Voices",
                                        interactive=True
                                    )
                                with gr.Row():    
                                    predefined_play_btn = gr.Button("▶️ Play/Stop")
                    
                    # Референсные файлы для клонирования
                            with gr.Group(visible=False) as clone_group:
                                with gr.Row():
                                    reference_file_select = gr.Dropdown(
                                        choices=populateReferenceFiles(),
                                        value=current_config.get("ui_state", {}).get("last_reference_file", "none"),
                                        label="Reference Audio Files",
                                        interactive=True
                                    )
                                with gr.Row(): 
                                    reference_play_btn = gr.Button("▶️ Play/Stop")
                        # Кнопки для работы с референсными файлами ТОЛЬКО ЗДЕСЬ
                                with gr.Row():
                                    reference_upload_btn = gr.UploadButton("📁 Upload Reference Audio",
                                        file_types=[".wav", ".mp3"],
                                        file_count="single",
                                        visible=True
                                    )

                            reference_audio_player = gr.Audio(
                                    visible=False,
                                    label="",
                                    interactive=False,
                                    show_label=False,
                                    elem_id="reference-audio-player",
                                    autoplay=False  # изначально выключено
                                )  
                            reference_audio_trigger = gr.Audio(
                                    visible=False,
                                    elem_id="reference-audio-trigger"
                                )      




                with gr.Row():
                    # Настройки генерации (аналог Generation Parameters из index.html)
                        with gr.Accordion("🎛 Generation Parameters", open=True):
                            with gr.Row():
                                with gr.Column():
                                    temperature_slider = gr.Slider(
                                        minimum=0.0,
                                        maximum=1.5,
                                        value=get_gen_default_temperature(),
                                        step=0.01,
                                        label="Temperature"
                                        )   
                                    cfg_weight_slider = gr.Slider(
                                        minimum=0.0,
                                        maximum=2.0,
                                        value=get_gen_default_cfg_weight(),
                                        step=0.01,
                                        label="CFG Weight"
                                        )
                                    seed_input = gr.Number(
                                        value=get_gen_default_seed(),
                                        label="Generation Seed (0 or -1 for random)"
                                        )
                                with gr.Column():
                                    exaggeration_slider = gr.Slider(
                                        minimum=0.0,
                                        maximum=2.0,
                                        value=get_gen_default_exaggeration(),
                                        step=0.01,
                                        label="Exaggeration"
                                        )
                                    speed_factor_slider = gr.Slider(
                                        minimum=0.25,
                                        maximum=4.0,
                                        value=get_gen_default_speed_factor(),
                                        step=0.05,
                                        label="Speed Factor"
                                        )
                                    language_select = gr.Dropdown(
                                        choices=language_options,
                                        value=current_config.get("generation_defaults", {}).get("language", "English (en)"),
                                        label="Language",
                                        interactive=True
                                        )

                with gr.Accordion("📚 Example Presets", open=False):
                    with gr.Row():
                        if appPresets:
                            preset_buttons = []
                            for preset in appPresets:
                                btn = gr.Button(
                                    preset.get("name", "Unnamed"),
                                    size="sm",
                                    variant="secondary"
                                )
                                btn.click(
                                    fn=lambda p=preset: applyPreset(p.get("name", ""), appPresets),
                                    inputs=[],
                                    outputs=[temperature_slider, exaggeration_slider, 
                                            cfg_weight_slider, speed_factor_slider, seed_input]
                                    )

                with gr.Row():                
                # Имя аудиофайла
                        with gr.Accordion("📁 Audio File Name", open=False):
                            audio_name_input = gr.Textbox(
                                label="Custom Audio Name",
                                placeholder="Enter custom name (without extension)",
                                value=""
                            )
        
        # Секция с результатами
                with gr.Row():
                    with gr.Column():
                # Аудиоплеер
                        audio_output = gr.Audio(
                            label="Generated Audio",
                            type="filepath",
                            interactive=False,
                            visible=False
                        )
                
              

        
        # Секция с информацией
                with gr.Accordion("💡 Tips & Tricks", open=False):
                    gr.Markdown("""
                    - For **Audiobooks**, use **MP3** format, enable **Split text**, and set a chunk size of ~250-500.
                    - Use **Predefined Voices** for consistent, high-quality output.
                    - For **Voice Cloning**, upload clean reference audio (`.wav`/`.mp3`). Quality of reference is key.
                    - Experiment with **Temperature** and other generation parameters to fine-tune output.
                    """)

            with gr.Tab("⚙️ Server Configuration"):
        # Секция конфигурации сервера (аналог Server Configuration из index.html)
            #with gr.Accordion("⚙️ Server Configuration", open=False):
                gr.Markdown("""
                These settings are loaded from `config.yaml` via an API call.
                **Restart the server** to apply changes to Host, Port, Model, or Path settings if modified.
                """)
                with gr.Row():
                    with gr.Column():
            
                        config_paths_model_cache = gr.Textbox(
                            label="Model Cache Path",
                            value=current_config.get("paths", {}).get("model_cache", "./model_cache"),
                            interactive=False
                            )      


                        config_tts_engine_reference_audio_path = gr.Textbox(
                            label="Reference Audio Path",
                            value=current_config.get("tts_engine", {}).get("reference_audio_path", "./reference_audio"),
                            interactive=True
                            )
                        config_tts_engine_predefined_voices_path = gr.Textbox(
                            label="Predefined Voices Path",
                            value=current_config.get("tts_engine", {}).get("predefined_voices_path", "./voices"),
                            interactive=True
                            )
                        config_paths_output = gr.Textbox(
                            label="Output Path",
                            value=current_config.get("paths", {}).get("output", "./outputs"),
                            interactive=True
                            )
                    with gr.Column():  
                        config_tts_engine_device = gr.Textbox(
                            label="TTS Device",
                            value=current_config.get("tts_engine", {}).get("device", "cpu"),
                            interactive=False
                            )                   
                        config_tts_engine_default_voice_id = gr.Textbox(
                            label="Predefined Voice",
                            value=current_config.get("tts_engine", {}).get("default_voice_id", ""),
                            interactive=True
                            )
                        config_tts_engine_default_voice_clone = gr.Textbox(
                            label="Clone Voice",
                            value=current_config.get("tts_engine", {}).get("default_voice_clone", ""),
                            interactive=True
                            )


                        config_audio_output_format = gr.Dropdown(
                                    choices=["wav", "mp3", "opus"],
                                    value=current_config.get("audio_output", {}).get("format", "mp3"),
                                    label="Audio Output Format",
                                    interactive=True
                                    )


                        config_audio_output_sample_rate = gr.Number(
                            label="Audio Sample Rate",
                            value=current_config.get("audio_output", {}).get("sample_rate", 24000),
                            precision=0,
                            interactive=True
                            )
            
            # Кнопки управления конфигурацией
                with gr.Row():
                    save_config_btn = gr.Button("💾 Save Server Configuration", variant="primary")
                #restart_server_btn = gr.Button("🔄 Restart Server", variant="secondary", visible=False)
            
            # Статус конфигурации
            #config_status = gr.Textbox(
            #    label="Configuration Status",
            #    value="",
            #    interactive=False,
            #    visible=False
            #)        








        # --- ПРИВЯЗКА ОБРАБОТЧИКОВ СОБЫТИЙ ---
        predefined_play_btn.click(
            fn=lambda file: toggle_voice_audio(file, "predefined"),
            inputs=[predefined_voice_select],
            outputs=[
                reference_audio_player,  # основной аудиоплеер
                predefined_play_btn,     # текст кнопки
                reference_audio_player,  # видимость
                reference_audio_player   # autoplay
            ]
        )
        save_config_btn.click(
            fn=save_settings_endpoint,
            inputs=[config_tts_engine_device, config_tts_engine_reference_audio_path, config_tts_engine_predefined_voices_path, 
                config_tts_engine_default_voice_id, config_tts_engine_default_voice_clone,
                config_paths_model_cache,config_paths_output,temperature_slider, exaggeration_slider,cfg_weight_slider, seed_input,
                speed_factor_slider, language_select, config_audio_output_format,config_audio_output_sample_rate]
        )
        
        reference_play_btn.click(
            fn=lambda file: toggle_voice_audio(file, "clone"),
            inputs=[reference_file_select],
            outputs=[
                reference_audio_player,  # основной аудиоплеер
                reference_play_btn,      # текст кнопки
                reference_audio_player,  # видимость
                reference_audio_player   # autoplay
            ]
        )
        reference_upload_btn.upload(
            fn=on_reference_upload,
            inputs=[reference_upload_btn],
            outputs=[reference_file_select]
        )
        # Основная кнопка Generate
        generate_btn.click(lambda: (gr.update(interactive=False)),outputs=[generate_btn]) \
            .then(fn=on_generate_click,inputs=[
                text_area,
                voice_mode_radio,
                predefined_voice_select,
                reference_file_select,
                temperature_slider,
                exaggeration_slider,
                cfg_weight_slider,
                speed_factor_slider,
                seed_input,
                language_select,
                split_text_toggle,
                chunk_size_slider,
                config_audio_output_format,
                audio_name_input
            ],outputs=[audio_output]) \
            .then (lambda: (gr.update(interactive=True)),outputs=[generate_btn])
        
        # Кнопки управления текстом

        accent_btn.click(
            fn=on_accent_click,
            inputs=[text_area],
            outputs=[text_area]
        )
        

        
        # Обновление счетчика символов
        text_area.change(
            fn=on_text_input,
            inputs=[text_area],
            outputs=[char_count]
        )
        
        # Переключение режимов голоса
        voice_mode_radio.change(
            fn=toggleVoiceOptionsDisplay,
            inputs=[voice_mode_radio],
            outputs=[predefined_group, clone_group]
        )
        
#        # Переключение видимости настроек чанкинга
#        split_text_toggle.change(
#            fn=toggleChunkControlsVisibility,
#            inputs=[split_text_toggle],
#            outputs=[chunk_size_slider, chunk_size_value_display]
#        )
        
        # Автоматическое скрытие уведомлений через 3 секунды
        def hide_notification():
            time.sleep(3)
            return gr.update(visible=False)
        
        # Добавляем скрытие уведомлений
        notification_display.change(
            fn=lambda: gr.update(visible=True),
            outputs=[notification_display]
        ).then(
            fn=hide_notification,
            outputs=[notification_display]
        )
    
    return demo

# --- ЗАПУСК СЕРВЕРА ---

def main():
    """Запуск Gradio сервера"""
    
    # Загрузка TTS модели
    logger.info("Initializing TTS Server...")
    
    if not engine.load_model():
        logger.critical("CRITICAL: TTS Model failed to load on startup.")
        return
    
    logger.info("TTS Model loaded successfully via engine.")
    
    # Создание интерфейса
    demo = create_gradio_interface()
    
    # Конфигурация запуска
    server_host = get_host()
    server_port = get_port()
    
    logger.info(f"Starting TTS Server on http://{server_host}:{server_port}")
    logger.info(f"Web UI available at http://{server_host}:{server_port}")
    
    # Запуск Gradio
    demo.launch(share=True)

if __name__ == "__main__":
    main()