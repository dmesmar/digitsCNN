from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    api_key: str = Field(validation_alias="API_KEY")
    data_dir: Path = Field(
        default=Path("/mnt/data"), validation_alias="DATA_DIR"
    )
    model_filename: str = Field(
        default="model.h5", validation_alias="MODEL_FILENAME"
    )


    model_config = SettingsConfigDict(
        env_file=".env",          
        env_file_encoding="utf-8",
        case_sensitive=True,      
        populate_by_name=True     
    )


    @property
    def model_path(self) -> Path:
        return self.data_dir / self.model_filename

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"



settings = Settings()