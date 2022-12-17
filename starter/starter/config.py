from pydantic import BaseSettings


class Settings(BaseSettings):
    environment: str
    model_save_path: str
    pipeline_save_path: str
    label_encoder_save_path: str

    class Config:
        env_file = ".env", ".test.env"
