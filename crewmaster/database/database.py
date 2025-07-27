"""
Database operations and session management for CrewMaster.
"""

import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager

from .models import Base, AgentModel, CrewModel, ToolModel, KnowledgeBaseModel, ExecutionLogModel

class Database:
    """Database manager for CrewMaster."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database connection."""
        if database_url is None:
            # Default to SQLite in user's home directory
            db_dir = os.path.expanduser("~/.crewmaster")
            os.makedirs(db_dir, exist_ok=True)
            database_url = f"sqlite:///{db_dir}/crewmaster.db"
        
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
            echo=False
        )
        
        # Enable foreign key constraints for SQLite
        if "sqlite" in database_url:
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)
    
    def reset_database(self):
        """Reset the database by dropping and recreating all tables."""
        self.drop_tables()
        self.create_tables()
    
    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_session_sync(self) -> Session:
        """Get a synchronous database session."""
        return self.SessionLocal()

class AgentRepository:
    """Repository for agent operations."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create_agent(self, agent_data: Dict[str, Any]) -> AgentModel:
        """Create a new agent."""
        with self.db.get_session() as session:
            agent = AgentModel(**agent_data)
            session.add(agent)
            session.flush()
            session.refresh(agent)
            return agent
    
    def get_agent(self, agent_id: str) -> Optional[AgentModel]:
        """Get an agent by ID."""
        with self.db.get_session() as session:
            return session.query(AgentModel).filter(AgentModel.id == agent_id).first()
    
    def get_agents(self, limit: int = 100, offset: int = 0) -> List[AgentModel]:
        """Get all agents with pagination."""
        with self.db.get_session() as session:
            return session.query(AgentModel).offset(offset).limit(limit).all()
    
    def search_agents(self, role: Optional[str] = None, tools: Optional[List[str]] = None) -> List[AgentModel]:
        """Search agents by role and tools."""
        with self.db.get_session() as session:
            query = session.query(AgentModel)
            
            if role:
                query = query.filter(AgentModel.role.ilike(f"%{role}%"))
            
            if tools:
                # This would need a more complex query for tool matching
                pass
            
            return query.all()
    
    def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> Optional[AgentModel]:
        """Update an agent."""
        with self.db.get_session() as session:
            agent = session.query(AgentModel).filter(AgentModel.id == agent_id).first()
            if agent:
                for key, value in updates.items():
                    setattr(agent, key, value)
                session.flush()
                session.refresh(agent)
            return agent
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        with self.db.get_session() as session:
            agent = session.query(AgentModel).filter(AgentModel.id == agent_id).first()
            if agent:
                session.delete(agent)
                return True
            return False
    
    def increment_usage(self, agent_id: str):
        """Increment agent usage count."""
        with self.db.get_session() as session:
            agent = session.query(AgentModel).filter(AgentModel.id == agent_id).first()
            if agent:
                agent.usage_count += 1
                agent.last_used = datetime.now(timezone.utc)

class CrewRepository:
    """Repository for crew operations."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create_crew(self, crew_data: Dict[str, Any], agent_ids: List[str]) -> CrewModel:
        """Create a new crew with associated agents."""
        with self.db.get_session() as session:
            crew = CrewModel(**crew_data)
            
            # Add agents to crew
            agents = session.query(AgentModel).filter(AgentModel.id.in_(agent_ids)).all()
            crew.agents.extend(agents)
            
            session.add(crew)
            session.flush()
            session.refresh(crew)
            return crew
    
    def get_crew(self, crew_id: str) -> Optional[CrewModel]:
        """Get a crew by ID."""
        with self.db.get_session() as session:
            return session.query(CrewModel).filter(CrewModel.id == crew_id).first()
    
    def get_crews(self, limit: int = 100, offset: int = 0) -> List[CrewModel]:
        """Get all crews with pagination."""
        with self.db.get_session() as session:
            return session.query(CrewModel).offset(offset).limit(limit).all()
    
    def search_crews(self, task_keywords: Optional[List[str]] = None) -> List[CrewModel]:
        """Search crews by task keywords."""
        with self.db.get_session() as session:
            query = session.query(CrewModel)
            
            if task_keywords:
                for keyword in task_keywords:
                    query = query.filter(CrewModel.task.ilike(f"%{keyword}%"))
            
            return query.all()
    
    def update_crew(self, crew_id: str, updates: Dict[str, Any]) -> Optional[CrewModel]:
        """Update a crew."""
        with self.db.get_session() as session:
            crew = session.query(CrewModel).filter(CrewModel.id == crew_id).first()
            if crew:
                for key, value in updates.items():
                    setattr(crew, key, value)
                session.flush()
                session.refresh(crew)
            return crew
    
    def delete_crew(self, crew_id: str) -> bool:
        """Delete a crew."""
        with self.db.get_session() as session:
            crew = session.query(CrewModel).filter(CrewModel.id == crew_id).first()
            if crew:
                session.delete(crew)
                return True
            return False
    
    def increment_execution(self, crew_id: str):
        """Increment crew execution count."""
        with self.db.get_session() as session:
            crew = session.query(CrewModel).filter(CrewModel.id == crew_id).first()
            if crew:
                crew.execution_count += 1
                crew.last_executed = datetime.now(timezone.utc)

class ToolRepository:
    """Repository for tool operations."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create_tool(self, tool_data: Dict[str, Any]) -> ToolModel:
        """Create a new tool."""
        with self.db.get_session() as session:
            tool = ToolModel(**tool_data)
            session.add(tool)
            session.flush()
            session.refresh(tool)
            return tool
    
    def get_tool(self, tool_id: str) -> Optional[ToolModel]:
        """Get a tool by ID."""
        with self.db.get_session() as session:
            return session.query(ToolModel).filter(ToolModel.id == tool_id).first()
    
    def get_tool_by_name(self, name: str) -> Optional[ToolModel]:
        """Get a tool by name."""
        with self.db.get_session() as session:
            return session.query(ToolModel).filter(ToolModel.name == name).first()
    
    def get_tools(self, category: Optional[str] = None, active_only: bool = True) -> List[ToolModel]:
        """Get tools by category."""
        with self.db.get_session() as session:
            query = session.query(ToolModel)
            
            if active_only:
                query = query.filter(ToolModel.is_active == True)
            
            if category:
                query = query.filter(ToolModel.category == category)
            
            return query.all()
    
    def get_tools_for_task(self, task_description: str) -> List[ToolModel]:
        """Get recommended tools for a task based on keywords."""
        # This would implement keyword matching logic
        with self.db.get_session() as session:
            return session.query(ToolModel).filter(ToolModel.is_active == True).all()

class ExecutionLogRepository:
    """Repository for execution log operations."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create_log(self, log_data: Dict[str, Any]) -> ExecutionLogModel:
        """Create a new execution log."""
        with self.db.get_session() as session:
            log = ExecutionLogModel(**log_data)
            session.add(log)
            session.flush()
            session.refresh(log)
            return log
    
    def update_log(self, log_id: str, updates: Dict[str, Any]) -> Optional[ExecutionLogModel]:
        """Update an execution log."""
        with self.db.get_session() as session:
            log = session.query(ExecutionLogModel).filter(ExecutionLogModel.id == log_id).first()
            if log:
                for key, value in updates.items():
                    setattr(log, key, value)
                session.flush()
                session.refresh(log)
            return log
    
    def get_logs_for_crew(self, crew_id: str, limit: int = 50) -> List[ExecutionLogModel]:
        """Get execution logs for a crew."""
        with self.db.get_session() as session:
            return (session.query(ExecutionLogModel)
                   .filter(ExecutionLogModel.crew_id == crew_id)
                   .order_by(ExecutionLogModel.started_at.desc())
                   .limit(limit)
                   .all())
    
    def get_logs_for_agent(self, agent_id: str, limit: int = 50) -> List[ExecutionLogModel]:
        """Get execution logs for an agent."""
        with self.db.get_session() as session:
            return (session.query(ExecutionLogModel)
                   .filter(ExecutionLogModel.agent_id == agent_id)
                   .order_by(ExecutionLogModel.started_at.desc())
                   .limit(limit)
                   .all())