"""
Database models for CrewMaster using SQLAlchemy.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import uuid4
from sqlalchemy import (
    Column, String, Text, DateTime, Boolean, JSON, ForeignKey, 
    Integer, Table, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.types import String
from pydantic import BaseModel

Base = declarative_base()

# Association tables for many-to-many relationships
agent_tool_association = Table(
    'agent_tools',
    Base.metadata,
    Column('agent_id', String, ForeignKey('agents.id'), primary_key=True),
    Column('tool_id', String, ForeignKey('tools.id'), primary_key=True)
)

crew_agent_association = Table(
    'crew_agents', 
    Base.metadata,
    Column('crew_id', String, ForeignKey('crews.id'), primary_key=True),
    Column('agent_id', String, ForeignKey('agents.id'), primary_key=True)
)

class AgentModel(Base):
    """Database model for agents."""
    __tablename__ = 'agents'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)
    goal = Column(Text, nullable=False)
    backstory = Column(Text, nullable=False)
    
    # Agent configuration
    verbose = Column(Boolean, default=True)
    allow_delegation = Column(Boolean, default=False)
    max_iter = Column(Integer, default=5)
    max_execution_time = Column(Integer, nullable=True)
    
    # Memory configuration
    memory_enabled = Column(Boolean, default=False)
    memory_type = Column(String, default="short_term")  # short_term, long_term, shared
    memory_config = Column(JSON, nullable=True)
    
    # Guardrails and constraints
    guardrails = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime, nullable=True)
    
    # Performance metrics
    avg_execution_time = Column(Integer, nullable=True)  # in seconds
    success_rate = Column(Integer, default=100)  # percentage
    
    # Relationships
    tools = relationship("ToolModel", secondary=agent_tool_association, back_populates="agents")
    crews = relationship("CrewModel", secondary=crew_agent_association, back_populates="agents")
    executions = relationship("ExecutionLogModel", back_populates="agent")

class CrewModel(Base):
    """Database model for crews."""
    __tablename__ = 'crews'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String, nullable=False)
    task = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    
    # Crew configuration
    process_type = Column(String, default="sequential")  # sequential, parallel, hierarchical
    verbose = Column(Boolean, default=True)
    memory_enabled = Column(Boolean, default=False)
    
    # Task specifications
    expected_output = Column(Text, nullable=True)
    task_config = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    execution_count = Column(Integer, default=0)
    last_executed = Column(DateTime, nullable=True)
    
    # Performance metrics
    avg_execution_time = Column(Integer, nullable=True)  # in seconds
    success_rate = Column(Integer, default=100)  # percentage
    
    # Relationships
    agents = relationship("AgentModel", secondary=crew_agent_association, back_populates="crews")
    tasks = relationship("TaskModel", back_populates="crew")
    executions = relationship("ExecutionLogModel", back_populates="crew")

class TaskModel(Base):
    """Database model for tasks."""
    __tablename__ = 'tasks'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    crew_id = Column(String, ForeignKey('crews.id'), nullable=False)
    agent_id = Column(String, ForeignKey('agents.id'), nullable=True)
    
    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    expected_output = Column(Text, nullable=False)
    
    # Task configuration
    context = Column(JSON, nullable=True)  # Input context for the task
    tools_override = Column(JSON, nullable=True)  # Task-specific tools
    dependencies = Column(JSON, nullable=True)  # Task dependencies
    
    # Task flow
    order_index = Column(Integer, default=0)
    is_async = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    crew = relationship("CrewModel", back_populates="tasks")

class ToolModel(Base):
    """Database model for tools."""
    __tablename__ = 'tools'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=False)
    category = Column(String, nullable=False)  # web_search, file_ops, code_exec, etc.
    
    # Tool configuration
    tool_type = Column(String, nullable=False)  # built_in, custom, api
    config = Column(JSON, nullable=True)  # Tool-specific configuration
    api_spec = Column(JSON, nullable=True)  # For API tools
    
    # Requirements and constraints
    required_env_vars = Column(JSON, nullable=True)
    permissions = Column(JSON, nullable=True)
    rate_limits = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)
    usage_count = Column(Integer, default=0)
    
    # Relationships
    agents = relationship("AgentModel", secondary=agent_tool_association, back_populates="tools")

class KnowledgeBaseModel(Base):
    """Database model for knowledge bases."""
    __tablename__ = 'knowledge_bases'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # Knowledge source
    source_type = Column(String, nullable=False)  # file, url, api, manual
    source_path = Column(String, nullable=True)
    source_config = Column(JSON, nullable=True)
    
    # Vector store configuration
    embedding_model = Column(String, default="sentence-transformers/all-MiniLM-L6-v2")
    vector_store_type = Column(String, default="faiss")
    vector_store_path = Column(String, nullable=True)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_status = Column(String, default="pending")  # pending, processing, completed, failed
    document_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_accessed = Column(DateTime, nullable=True)

class ExecutionLogModel(Base):
    """Database model for execution logs."""
    __tablename__ = 'execution_logs'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    crew_id = Column(String, ForeignKey('crews.id'), nullable=True)
    agent_id = Column(String, ForeignKey('agents.id'), nullable=True)
    
    # Execution details
    task_input = Column(Text, nullable=True)
    task_output = Column(Text, nullable=True)
    execution_time = Column(Integer, nullable=True)  # in seconds
    status = Column(String, nullable=False)  # started, completed, failed, cancelled
    
    # Error handling
    error_message = Column(Text, nullable=True)
    error_type = Column(String, nullable=True)
    
    # Performance metrics
    tokens_used = Column(Integer, nullable=True)
    api_calls_made = Column(Integer, nullable=True)
    tools_used = Column(JSON, nullable=True)
    
    # Metadata
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    crew = relationship("CrewModel", back_populates="executions")
    agent = relationship("AgentModel", back_populates="executions")

# Pydantic models for API serialization
class AgentCreate(BaseModel):
    name: str
    role: str
    goal: str
    backstory: str
    verbose: bool = True
    allow_delegation: bool = False
    max_iter: int = 5
    max_execution_time: Optional[int] = None
    memory_enabled: bool = False
    memory_type: str = "short_term"
    memory_config: Optional[Dict[str, Any]] = None
    guardrails: Optional[List[str]] = None

class AgentResponse(BaseModel):
    id: str
    name: str
    role: str
    goal: str
    backstory: str
    verbose: bool
    memory_enabled: bool
    memory_type: str
    tools: List[str]
    created_at: datetime
    usage_count: int
    
    class Config:
        from_attributes = True

class CrewCreate(BaseModel):
    name: str
    task: str
    description: Optional[str] = None
    process_type: str = "sequential"
    verbose: bool = True
    memory_enabled: bool = False
    expected_output: Optional[str] = None
    agent_ids: List[str]

class CrewResponse(BaseModel):
    id: str
    name: str
    task: str
    description: Optional[str]
    process_type: str
    agents: List[AgentResponse]
    created_at: datetime
    execution_count: int
    
    class Config:
        from_attributes = True

class ExecutionResult(BaseModel):
    crew_id: str
    status: str
    output: Optional[str] = None
    execution_time: Optional[int] = None
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True