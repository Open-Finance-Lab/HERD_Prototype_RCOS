from pathlib import Path
from typing import List, Optional
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_ID: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        description="HF repo id or local path to a model folder"
    )
    REVISION: Optional[str] = None             
    HF_TOKEN: Optional[str] = None             
    TRUST_REMOTE_CODE: bool = False

    DEVICE_MAP_AUTO: bool = True               
    TORCH_DTYPE_AUTO: bool = True              
    TORCH_DTYPE: Optional[str] = None          
    LOCAL_FILES_ONLY_IF_PATH_EXISTS: bool = True

    MAX_NEW_TOKENS: int = 256
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95
    TOP_K: int = 0                                
    REPETITION_PENALTY: float = 1.0
    STOP: List[str] = Field(default_factory=list) 

    class Config:
        env_file = ".env"
        env_prefix = ""   

    @computed_field
    @property
    def MODEL_PATH(self) -> Path:
        return Path(self.MODEL_ID).expanduser().resolve()

    @computed_field
    @property
    def IS_LOCAL(self) -> bool:
        try:
            return self.LOCAL_FILES_ONLY_IF_PATH_EXISTS and self.MODEL_PATH.exists()
        except Exception:
            return False

settings = Settings()
