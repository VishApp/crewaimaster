"""
LLM Provider factory for supporting multiple AI providers.
"""

import os
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def get_llm_config(self, config) -> Dict[str, Any]:
        """Get the LLM configuration for this provider."""
        pass
    
    @abstractmethod
    def validate_config(self, config) -> bool:
        """Validate provider-specific configuration."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def get_llm_config(self, config) -> Dict[str, Any]:
        return {
            "model": config.llm.model or "gpt-4",
            "api_key": config.llm.api_key,
            "base_url": config.llm.base_url
        }
    
    def validate_config(self, config) -> bool:
        return bool(config.llm.api_key)


class GoogleProvider(LLMProvider):
    """Google Gemini/Vertex AI provider implementation."""
    
    def get_llm_config(self, config) -> Dict[str, Any]:
        return {
            "model": config.llm.model or "gemini-pro",
            "api_key": config.llm.api_key,
            "project_id": config.llm.project_id,
            "region": config.llm.region or 'us-central1',
            "auth_file": config.llm.auth_file
        }
    
    def validate_config(self, config) -> bool:
        return bool(config.llm.api_key or config.llm.auth_file)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""
    
    def get_llm_config(self, config) -> Dict[str, Any]:
        return {
            "model": config.llm.model or "claude-3-sonnet-20240229",
            "api_key": config.llm.api_key,
            "base_url": config.llm.base_url
        }
    
    def validate_config(self, config) -> bool:
        return bool(config.llm.api_key)


class DeepSeekProvider(LLMProvider):
    """DeepSeek provider implementation."""
    
    def get_llm_config(self, config) -> Dict[str, Any]:
        return {
            "model": config.llm.model or "deepseek-chat",
            "api_key": config.llm.api_key,
            "base_url": config.llm.base_url or "https://api.deepseek.com/v1"
        }
    
    def validate_config(self, config) -> bool:
        return bool(config.llm.api_key)


class CustomProvider(LLMProvider):
    """Custom provider for any OpenAI-compatible API."""
    
    def get_llm_config(self, config) -> Dict[str, Any]:
        return {
            "model": config.llm.model,
            "api_key": config.llm.api_key,
            "base_url": config.llm.base_url
        }
    
    def validate_config(self, config) -> bool:
        return bool(config.llm.api_key and config.llm.base_url)


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    _providers = {
        "openai": OpenAIProvider(),
        "google": GoogleProvider(),
        "anthropic": AnthropicProvider(),
        "deepseek": DeepSeekProvider(),
        "custom": CustomProvider()
    }
    
    @classmethod
    def get_provider(cls, provider_name: str) -> LLMProvider:
        """Get a provider by name."""
        provider = cls._providers.get(provider_name.lower())
        if not provider:
            raise ValueError(f"Unsupported provider: {provider_name}")
        return provider
    
    @classmethod
    def get_llm_config(cls, config) -> Dict[str, Any]:
        """Get LLM configuration for the specified provider."""
        provider = cls.get_provider(config.llm.provider)
        return provider.get_llm_config(config)
    
    @classmethod
    def validate_config(cls, config) -> bool:
        """Validate configuration for the specified provider."""
        try:
            provider = cls.get_provider(config.llm.provider)
            return provider.validate_config(config)
        except ValueError:
            return False
    
    @classmethod
    def list_providers(cls) -> list:
        """List all available providers."""
        return list(cls._providers.keys())


def get_llm_config_for_crewai(config) -> Dict[str, Any]:
    """
    Get LLM configuration formatted for CrewAI usage.
    This function adapts provider configs to CrewAI's expected format.
    """
    base_config = LLMProviderFactory.get_llm_config(config)
    
    # Format model name with provider prefix for LiteLLM
    provider = config.llm.provider.lower()
    model = base_config.get("model", "")
    
    # Apply provider prefix if not already present
    if provider == "anthropic" and not model.startswith("anthropic/"):
        base_config["model"] = f"anthropic/{model}"
    elif provider == "openai" and not model.startswith("openai/") and not model.startswith("gpt"):
        base_config["model"] = f"openai/{model}"
    elif provider == "google" and not model.startswith("google/") and not model.startswith("gemini"):
        base_config["model"] = f"google/{model}"
    elif provider == "deepseek" and not model.startswith("deepseek/"):
        base_config["model"] = f"deepseek/{model}"
    elif provider == "custom":
        # For custom providers, use model name as-is since custom base_url handles routing
        base_config["model"] = model
        # Ensure LiteLLM gets the custom configuration
        base_config["api_base"] = base_config.get("base_url")
    
    # Add common CrewAI parameters
    base_config.update({
        "provider": provider,
        "temperature": config.llm.temperature,
        "max_tokens": config.llm.max_tokens
    })
    
    return base_config