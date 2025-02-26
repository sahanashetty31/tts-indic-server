import enum

from pydantic_settings import BaseSettings

SPEED = 1.0

class StrEnum(str, enum.Enum):
    """Custom implementation of StrEnum for Python versions < 3.11"""
    def __str__(self):
        return str(self.value)

# NOTE: commented out response formats don't work
class ResponseFormat(StrEnum):
    MP3 = "mp3"
    # OPUS = "opus"
    # AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    # PCM = "pcm"

class Config(BaseSettings):
    log_level: str = "info"  # env: LOG_LEVEL
    model: str = "ai4bharat/indic-parler-tts"  # env: MODEL
    max_models: int = 1  # env: MAX_MODELS
    lazy_load_model: bool = False  # env: LAZY_LOAD_MODEL
    voice: str = (
        "Thomas speaks moderately slowly in a sad tone with emphasis and high quality audio."  # env: VOICE
    )
    response_format: ResponseFormat = ResponseFormat.MP3  # env: RESPONSE_FORMAT

config = Config()