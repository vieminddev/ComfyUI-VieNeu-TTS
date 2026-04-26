import os
import tempfile

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Voice list — populated dynamically after first model load.
# ---------------------------------------------------------------------------
_VOICE_IDS: list = [
    "Bích Ngọc (Nữ - Miền Bắc)",
    "Phạm Tuyên (Nam - Miền Bắc)",
    "Thục Đoan (Nữ - Miền Nam)",
    "Xuân Vĩnh (Nam - Miền Nam)",
]

# ---------------------------------------------------------------------------
# Singleton model loader with config-aware reload
# ---------------------------------------------------------------------------
_tts_instance = None
_tts_config: dict = {}

MODE_OPTIONS = ["turbo", "turbo_gpu", "standard"]
DEVICE_OPTIONS = ["cpu", "cuda", "mps"]


def _get_tts(mode: str = "turbo", device: str = "cpu"):
    global _tts_instance, _VOICE_IDS, _tts_config

    new_config = {"mode": mode, "device": device}

    if _tts_instance is not None and _tts_config != new_config:
        print(f"[VieNeu-TTS] Config changed ({_tts_config} → {new_config}), reloading…")
        try:
            _tts_instance.close()
        except Exception:
            pass
        _tts_instance = None

    if _tts_instance is None:
        try:
            from vieneu import Vieneu
        except ImportError as exc:
            raise RuntimeError(
                "[VieNeu-TTS] vieneu package not found. Run: pip install vieneu"
            ) from exc

        _tts_config = new_config
        kwargs: dict = {}

        if mode == "standard":
            # PyTorch base model — supports backbone_device / codec_device
            kwargs["backbone_device"] = device
            kwargs["codec_device"] = device
        elif mode == "turbo_gpu":
            # GGUF on GPU
            kwargs["device"] = device

        print(f"[VieNeu-TTS] Loading model (mode={mode}, device={device})…")
        _tts_instance = Vieneu(mode=mode, **kwargs)

        try:
            _VOICE_IDS = [vid for _, vid in _tts_instance.list_preset_voices()]
            print(f"[VieNeu-TTS] Loaded {len(_VOICE_IDS)} voices: {_VOICE_IDS}")
        except Exception as e:
            print(f"[VieNeu-TTS] Could not refresh voice list: {e}")

        print("[VieNeu-TTS] Model ready.")

    return _tts_instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_to_comfy(audio_np: np.ndarray, sample_rate: int) -> dict:
    waveform = torch.from_numpy(audio_np).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.T.contiguous()
    waveform = waveform.unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


def _comfy_to_wav_file(audio: dict, path: str) -> None:
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    wav = waveform[0]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    audio_np = wav.squeeze(0).numpy().astype(np.float32)
    sf.write(path, audio_np, sr, subtype="PCM_16")


def _apply_speed(audio: dict, speed: float) -> dict:
    if abs(speed - 1.0) < 1e-3:
        return audio
    waveform = audio["waveform"].float()
    new_T = max(1, int(round(waveform.shape[-1] / speed)))
    stretched = F.interpolate(waveform, size=new_T, mode="linear", align_corners=False)
    return {"waveform": stretched, "sample_rate": audio["sample_rate"]}


def _auto_transcribe(audio_path: str) -> str:
    """Auto-transcribe reference audio using Whisper model directly (no pipeline)."""
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
    except ImportError:
        raise RuntimeError(
            "[VieNeu-TTS] Auto-transcription requires transformers. "
            "Run: pip install transformers  —  or fill in ref_text manually."
        )
    print("[VieNeu-TTS] ref_text empty — auto-transcribing with Whisper…")

    # Load and resample to 16 kHz mono (Whisper requirement)
    audio_np, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    if sr != 16000:
        wav = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
        new_len = int(len(audio_np) * 16000 / sr)
        wav = F.interpolate(wav, size=new_len, mode="linear", align_corners=False)
        audio_np = wav.squeeze().numpy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)

    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    print(f"[VieNeu-TTS] Transcript: {transcript}")
    return transcript


def _run_infer_to_comfy(tts, infer_kwargs: dict) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        audio_obj = tts.infer(**infer_kwargs)
        tts.save(audio_obj, tmp_path)
        audio_np, sr = sf.read(tmp_path, dtype="float32")
        return _numpy_to_comfy(audio_np, sr)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Node: VieNeuTTS  (preset voices)
# ---------------------------------------------------------------------------

class VieNeuTTSNode:
    """Generate speech from text using VieNeu-TTS preset voices."""

    CATEGORY = "VieNeu/TTS"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        voices = _VOICE_IDS if _VOICE_IDS else ["(load model first)"]
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Xin chào, tôi là VieNeu TTS.",
                    "placeholder": "Nhập văn bản cần đọc…",
                }),
                "voice": (voices, {"default": voices[0]}),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0,
                    "step": 0.05, "display": "slider",
                }),
            },
            "optional": {
                "mode": (MODE_OPTIONS, {"default": "turbo"}),
                "device": (DEVICE_OPTIONS, {"default": "cpu"}),
            },
        }

    def generate(self, text: str, voice: str, speed: float,
                 mode: str = "turbo", device: str = "cpu"):
        if not text.strip():
            raise ValueError("[VieNeu-TTS] Text input is empty.")
        tts = _get_tts(mode, device)
        voice_data = tts.get_preset_voice(voice)
        audio = _run_infer_to_comfy(tts, {"text": text, "voice": voice_data})
        return (_apply_speed(audio, speed),)


# ---------------------------------------------------------------------------
# Node: VieNeuTTSClone  (zero-shot voice cloning)
# ---------------------------------------------------------------------------

class VieNeuTTSCloneNode:
    """Clone a speaker's voice from a short reference audio clip."""

    CATEGORY = "VieNeu/TTS"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone_voice"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Đây là giọng nói được clone từ audio mẫu.",
                    "placeholder": "Nhập văn bản cần đọc…",
                }),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Transcription of reference audio (optional)…",
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0,
                    "step": 0.05, "display": "slider",
                }),
            },
            "optional": {
                "mode": (MODE_OPTIONS, {"default": "turbo"}),
                "device": (DEVICE_OPTIONS, {"default": "cpu"}),
            },
        }

    def clone_voice(self, text: str, ref_audio: dict, ref_text: str,
                    speed: float, mode: str = "turbo", device: str = "cpu"):
        if not text.strip():
            raise ValueError("[VieNeu-TTS] Text input is empty.")

        tts = _get_tts(mode, device)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref_path = f.name
        try:
            _comfy_to_wav_file(ref_audio, ref_path)

            if mode == "standard":
                # standard mode: encode audio → ref_codes first, then infer
                transcript = ref_text.strip() or _auto_transcribe(ref_path)
                ref_codes = tts.encode_reference(ref_path)
                infer_kwargs: dict = {
                    "text": text,
                    "ref_codes": ref_codes,
                    "ref_text": transcript,
                }
            else:
                # turbo / turbo_gpu: pass ref_audio path directly
                infer_kwargs = {"text": text, "ref_audio": ref_path}
                if ref_text.strip():
                    infer_kwargs["ref_text"] = ref_text.strip()

            audio = _run_infer_to_comfy(tts, infer_kwargs)
        finally:
            try:
                os.unlink(ref_path)
            except OSError:
                pass

        return (_apply_speed(audio, speed),)


# ---------------------------------------------------------------------------
# Node: VieNeuTTSUnloadModel
# ---------------------------------------------------------------------------

class VieNeuTTSUnloadModelNode:
    """Unload the VieNeu-TTS model from memory."""

    CATEGORY = "VieNeu/TTS"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "unload"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def unload(self):
        global _tts_instance, _tts_config
        if _tts_instance is not None:
            try:
                _tts_instance.close()
            except Exception:
                pass
            _tts_instance = None
            _tts_config = {}
            print("[VieNeu-TTS] Model unloaded.")
        return ()
