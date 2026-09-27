import os
from functools import lru_cache
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import ConfigDict, field_validator
from pydantic_settings import BaseSettings


if os.path.exists(".env"):
    load_dotenv(dotenv_path=".env")


class EnvironmentVariables(BaseSettings):
    """
    EnvironmentVariables is a configuration class for managing application
    environment variables.
    Attributes:
        APP_NAME (str): The name of the application.
        APP_ENVIRONMENT (str): The current environment
            (one of: local, dev, uat, prod, test).
        APP_DEBUG (bool): Flag to enable or disable debug mode.
        APP_MAINTENANCE (bool): Flag to enable or disable maintenance mode.
            Defaults to False.
        ALLOWED_ORIGINS (str): Comma-separated list of allowed CORS origins.
        LOG_LEVEL (str): The logging level (e.g., INFO, DEBUG).
        LOG_CHANNEL (str): The logging channel to use.
        LOG_DIR (str): Directory path for storing log files.
        STORAGE_TYPE (str): The storage type to use. Defaults to "local".
        STORAGE_BUCKET (Optional[str]): The storage bucket name. Optional.
        STORAGE_PATH (str): The storage path. Defaults to "storage/core".
    Class Attributes:
        model_config: Configuration for environment file loading.
    """

    APP_NAME: str
    APP_ENVIRONMENT: Literal["local", "dev", "uat", "prod", "test"]
    APP_VERSION: Optional[str] = "unknown"
    APP_RUNTIME: Optional[str] = "lambda"
    APP_DEBUG: bool
    ALLOWED_ORIGINS: str
    APP_MAINTENANCE: bool = False

    LOG_LEVEL: str
    LOG_CHANNEL: str
    LOG_DIR: str
    LOG_SAMPLE_RATE: Optional[str] = "1.0"

    STORAGE_TYPE: str = "local"
    STORAGE_BUCKET: Optional[str] = None
    STORAGE_PATH: str = "storage/core"

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("APP_ENVIRONMENT")
    def validate_app_environment(cls, v):
        """Validate APP_ENVIRONMENT is one of the expected values."""
        valid_environments = ["local", "dev", "uat", "prod", "test"]
        if v not in valid_environments:
            raise ValueError(
                f"APP_ENVIRONMENT must be one of {valid_environments}, got '{v}'"
            )
        return v

    @field_validator("STORAGE_BUCKET", mode="before")
    def convert_empty_to_none(cls, v):
        if v == "":
            return None
        return v


@lru_cache()
def env(var_name: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
    """
    Create and return an instance of EnvironmentVariables or a specific
    environment variable.

    This function initializes and returns an EnvironmentVariables object,
    which is used to manage and access environment variables for the
    application. If a variable name is provided, it returns the value of
    that specific environment variable. If the variable is not found, it
    returns the provided default value.

    Args:
        var_name (Optional[str]): The name of the environment variable to
            retrieve.
        default (Optional[str]): The default value to return if the
            variable is not found. Defaults to None.

    Returns:
        EnvironmentVariables or Optional[str]: An instance of the
        EnvironmentVariables class or the value of the specified
        environment variable, or the default value if not found.
    """
    env_vars = EnvironmentVariables()
    if var_name:
        return getattr(env_vars, var_name, default)
    return env_vars
