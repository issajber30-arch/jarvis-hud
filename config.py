"""Hardware, audio, model and colour settings (tuned for Apple M4, 16 GB). Only the main thread edits this file."""
from pathlib import Path

ROOT = Path(__file__).parent

# Audio
SAMPLE_RATE = 16000
BLOCK_SIZE = 1280            # 80 ms — openWakeWord frame; VAD reads it as 512-sample chunks
WAKE_MODEL = "hey_jarvis"
WAKE_FRAMEWORK = "tflite"    # ONNX path returns ~0 scores on macOS arm64 (openWakeWord #336)
WAKE_THRESHOLD = 0.5
WAKE_THRESHOLD_SPEAKING = 0.8
VAD_THRESHOLD = 0.5
SILENCE_END_MS = 700
MAX_UTTERANCE_S = 20
FOLLOWUP_WINDOW_S = 8
CONFIRM_TIMEOUT_S = 10

# Models
WHISPER_MODEL = "mlx-community/whisper-small.en-mlx"
EMBED_MODEL = "minishlab/potion-base-8M"
ROUTER_THRESHOLD = 0.85     # 93.8% held-out; 0.80 grabbed greetings ("good morning" -> agenda)
VECTORS_PATH = ROOT / "data" / "command_vectors.npz"
CLAUDE_MODEL = "claude-haiku-5-5"
CLAUDE_MAX_TOKENS = 1024
HISTORY_TURNS = 20
TTS_VOICE = "bm_george"
TTS_LANG = "b"               # British English
TTS_SPEED = 1.0
REFERENCE_VOICE_DIR = ROOT / "data" / "reference_voice"

# HUD
HUD_FPS = 30               # terminals cannot show more; 60 doubled CPU for nothing
TELEMETRY_HZ = 1
LOG_MAX_CHARS = 200

# Palette (gold holographic). Roles, not hues, so a re-theme is one edit here plus hud/jarvis.tcss.
HUD_PRIMARY = "#FFB23F"   # gold lines and text
HUD_DIM = "#6B3F0E"       # back faces, borders, faded text
HUD_GLOW = "#FFF0C8"      # core and label
HUD_SPEAKING = "#FFD98A"  # brighter gold while Jarvis talks
HUD_CONFIRM = "#9FE8FF"   # cool white-cyan: stands out against gold when awaiting a yes/no
HUD_ERROR = "#FF4D4D"
HUD_BG = "#070503"

THEMES = {
    "GOLD": {"HUD_PRIMARY": "#FFB23F", "HUD_DIM": "#6B3F0E", "HUD_GLOW": "#FFF0C8", "HUD_SPEAKING": "#FFD98A",
             "HUD_CONFIRM": "#9FE8FF", "HUD_BG": "#070503"},
    "ARC": {"HUD_PRIMARY": "#4FD1FF", "HUD_DIM": "#1B4F63", "HUD_GLOW": "#E6FBFF", "HUD_SPEAKING": "#BFF0FF",
            "HUD_CONFIRM": "#FFB547", "HUD_BG": "#05080C"},
    "MATRIX": {"HUD_PRIMARY": "#3DFF7A", "HUD_DIM": "#145C2E", "HUD_GLOW": "#D6FFE3", "HUD_SPEAKING": "#9CFFBE",
               "HUD_CONFIRM": "#FFB547", "HUD_BG": "#030805"},
}


def apply_theme(name: str) -> None:
    """Swap the HUD_* role colours; widgets read them at render time. HUD_ERROR stays red in every theme."""
    globals().update(THEMES[name])
