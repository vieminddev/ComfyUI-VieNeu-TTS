# ComfyUI-VieNeu-TTS

A production-ready ComfyUI custom node that integrates
[VieNeu-TTS](https://huggingface.co/pnnbao-ump/VieNeu-TTS) — a 0.6 B Vietnamese + English
text-to-speech model with zero-shot voice cloning.

---

## Features

- Text-to-speech in **Vietnamese and English** (mixed input supported)
- **6 preset speaker voices** — male/female, north/south Vietnamese accents
- **Zero-shot voice cloning** from a 3–5 s reference audio clip
- **Speed control** (0.5×–2.0×)
- Native **ComfyUI `AUDIO`** output — connects directly to `Save Audio`, `Preview Audio`, etc.
- Lazy model loading — model downloads and loads only on first inference

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/ComfyUI-VieNeu-TTS
cd ComfyUI-VieNeu-TTS
pip install -r requirements.txt
```

Restart ComfyUI. The VieNeu-TTS model weights will be downloaded automatically from
HuggingFace on the first inference.

> **Windows note:** if `pip install vieneu` fails, add the llama-cpp-python index:
> ```bash
> pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/
> ```

---

## Nodes

### `VieNeu TTS`
Synthesises speech from text using one of the built-in preset voices.

| Input  | Type   | Description |
|--------|--------|-------------|
| `text` | STRING | Text to speak (Vietnamese / English / mixed) |
| `voice` | COMBO | Preset speaker (see table below) |
| `speed` | FLOAT  | Playback speed multiplier — `1.0` = normal, `0.5`–`2.0` |

**Output:** `AUDIO`

---

### `VieNeu TTS (Voice Clone)`
Clones any speaker's voice from a short reference clip.

| Input      | Type   | Description |
|------------|--------|-------------|
| `text`     | STRING | Text to synthesise |
| `ref_audio`| AUDIO  | Reference audio (3–5 s recommended) |
| `ref_text` | STRING | Transcription of reference audio *(optional but improves quality)* |
| `speed`    | FLOAT  | Playback speed multiplier |

**Output:** `AUDIO`

---

### `VieNeu TTS – Unload Model`
Frees VRAM/RAM by calling `tts.close()` and dropping the model singleton.
Add to the end of a workflow when you want to reclaim memory.

---

## Preset Voices

| Display name | ID | Gender | Accent |
|--------------|----|--------|--------|
| Bình (Nam – Bắc) | `binh` | Male | North Vietnamese |
| Tuyên (Nam – Bắc) | `tuyen` | Male | North Vietnamese |
| Nguyên (Nam – Nam) | `nguyen` | Male | South Vietnamese |
| Hương (Nữ – Bắc) | `huong` | Female | North Vietnamese |
| Ngọc (Nữ – Bắc) | `ngoc` | Female | North Vietnamese |
| Đoạn (Nữ – Nam) | `doan` | Female | South Vietnamese |

---

## Example Workflows

### Basic TTS
```
[Text (multiline)] → [VieNeu TTS] → [Preview Audio]
```

### Voice Cloning
```
[Load Audio] → [VieNeu TTS (Voice Clone)] ← [Text]
                       ↓
                [Save Audio]
```

---

## Requirements

| Package | Purpose |
|---------|---------|
| `vieneu` | VieNeu-TTS Python SDK |
| `soundfile` | WAV encode/decode |
| `torch` | Tensor ops (installed with ComfyUI) |

---

## Model

[pnnbao-ump/VieNeu-TTS](https://huggingface.co/pnnbao-ump/VieNeu-TTS) — 0.6 B parameters,
BF16, fine-tuned from NeuTTS Air on the VieNeu-TTS-1000h dataset (421 k samples).

## License

Apache-2.0
