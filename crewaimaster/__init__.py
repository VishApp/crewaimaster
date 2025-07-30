"""
CrewAIMaster: A Python package for building intelligent multi-agent systems using CrewAI.

This package provides a CLI and framework for automatically generating, managing,
and executing multi-agent crews based on natural language tasks.
"""

import warnings
import os

# Suppress common deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Pydantic.*deprecated.*")
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince20.*")
warnings.filterwarnings("ignore", message=".*extra keyword arguments.*")
warnings.filterwarnings("ignore", message=".*Field.*deprecated.*")
warnings.filterwarnings("ignore", message=".*event loop.*")

# Set environment variable
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

__version__ = "0.1.2"
__author__ = "CrewAIMaster Team"
__email__ = "vishnuprasadapp@gmail.com"

from .core.master_agent_crew import MasterAgentCrew
from .core.file_generator import CrewFileGenerator

__all__ = [
    "MasterAgentCrew",
    "CrewFileGenerator",
]