"""
Configuration management for CrewMaster.
"""

import os
import yaml
import time
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field
from pathlib import Path


class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str = Field(default_factory=lambda: f"sqlite:///{Path.home()}/.crewmaster/crewmaster.db")
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10

class LLMConfig(BaseModel):
    """LLM configuration."""
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: Optional[str] = None
    base_url: Optional[str] = None

class MemoryConfig(BaseModel):
    """Memory configuration."""
    enabled: bool = True
    short_term_limit: int = 10
    long_term_limit: int = 100
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

class ToolConfig(BaseModel):
    """Tool configuration."""
    enabled_categories: List[str] = ["web_search", "file_ops", "code_exec", "api_calls"]
    max_tools_per_agent: int = 5
    custom_tools_path: Optional[str] = None

class CrewMasterConfig(BaseModel):
    """Main CrewMaster configuration."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    
    # Agent generation settings
    max_agents_per_crew: int = 5
    default_agent_verbose: bool = True
    default_agent_max_iter: int = 5
    
    # Crew execution settings
    default_process: str = "sequential"
    execution_timeout: int = 300  # seconds
    
    # Logging and debugging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    debug_mode: bool = False

class Config:
    """Configuration manager for CrewMaster."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration."""
        self.config_path = config_path or self._get_default_config_path()
        self._config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        config_dir = Path.home() / ".crewmaster"
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / "config.yaml")
    
    def _load_config(self) -> CrewMasterConfig:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                return CrewMasterConfig(**config_data)
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
                print("Using default configuration.")
        
        # Create default config and save it
        config = CrewMasterConfig()
        self.save_config(config)
        return config
    
    def save_config(self, config: Optional[CrewMasterConfig] = None):
        """Save configuration to file."""
        config_to_save = config or self._config
        
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(config_to_save.model_dump(), f, default_flow_style=False)
        except Exception as e:
            print(f"Warning: Failed to save config to {self.config_path}: {e}")
    
    def get(self, key: str = None) -> Any:
        """Get configuration value or entire config."""
        if key is None:
            return self._config
        
        # Support nested key access like "database.url"
        value = self._config
        for part in key.split('.'):
            value = getattr(value, part, None)
            if value is None:
                break
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        # This would need more complex logic for nested updates
        # For now, we'll keep it simple
        if hasattr(self._config, key):
            setattr(self._config, key, value)
            self.save_config()
    
    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration."""
        return self._config.database
    
    @property
    def llm(self) -> LLMConfig:
        """Get LLM configuration."""
        return self._config.llm
    
    @property
    def memory(self) -> MemoryConfig:
        """Get memory configuration."""
        return self._config.memory
    
    @property
    def tools(self) -> ToolConfig:
        """Get tools configuration."""
        return self._config.tools
    
    def update_from_env(self):
        """Update configuration from environment variables."""
        env_mappings = {
            'OPENAI_API_KEY': 'llm.api_key',
            'CREWMASTER_DATABASE_URL': 'database.url',
            'CREWMASTER_LLM_MODEL': 'llm.model',
            'CREWMASTER_LLM_BASE_URL': 'llm.base_url',
            'CREWMASTER_DEBUG': 'debug_mode',
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if config_key.endswith('debug_mode'):
                    env_value = env_value.lower() in ('true', '1', 'yes', 'on')
                elif config_key.endswith('temperature'):
                    env_value = float(env_value)
                elif config_key.endswith(('max_tokens', 'pool_size', 'max_overflow')):
                    env_value = int(env_value)
                
                # Set nested configuration values
                parts = config_key.split('.')
                if len(parts) == 2:
                    section, key = parts
                    section_obj = getattr(self._config, section)
                    setattr(section_obj, key, env_value)
                else:
                    setattr(self._config, config_key, env_value)