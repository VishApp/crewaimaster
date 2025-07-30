"""
Master Agent for CrewAIMaster.

This is the main orchestrator that ties together task analysis, crew design,
agent management, and execution.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from .config import Config
from .task_analyzer import TaskAnalyzer
from .crew_designer import CrewDesigner
from ..database.database import Database, AgentRepository, CrewRepository, ExecutionLogRepository
from ..database.models import CrewModel, AgentModel, ExecutionResult

class CrewExecutionResult:
    """Result of crew execution."""
    
    def __init__(self, crew_id: str, output: str, execution_time: int, 
                 logs: List[Dict[str, Any]], status: str = "completed"):
        self.crew_id = crew_id
        self.output = output
        self.execution_time = execution_time
        self.logs = logs
        self.status = status

class LogEntry:
    """Log entry for execution tracking."""
    
    def __init__(self, timestamp: datetime, message: str, level: str = "INFO"):
        self.timestamp = timestamp
        self.message = message
        self.level = level

class MasterAgent:
    """Main orchestrator for CrewAIMaster operations."""
    
    def __init__(self, config: Config):
        """Initialize the master agent."""
        self.config = config
        self.config.update_from_env()  # Load environment variables
        
        # Initialize database
        self.database = Database(self.config.database.url)
        
        # Initialize repositories
        self.agent_repo = AgentRepository(self.database)
        self.crew_repo = CrewRepository(self.database)
        self.execution_repo = ExecutionLogRepository(self.database)
        
        # Initialize core components
        self.task_analyzer = TaskAnalyzer()
        self.crew_designer = CrewDesigner(config, self.database)
    
    def create_crew(self, task_description: str, crew_name: Optional[str] = None,
                   reuse_agents: bool = True, verbose: bool = False) -> CrewModel:
        """Create a new crew for the given task."""
        if verbose:
            print(f"🔍 Analyzing task: {task_description}")
        
        # Analyze the task
        crew_spec = self.task_analyzer.analyze_task(task_description)
        
        # Override name if provided
        if crew_name:
            crew_spec.name = crew_name
        
        if verbose:
            print(f"📋 Crew specification created:")
            print(f"   Name: {crew_spec.name}")
            print(f"   Complexity: {crew_spec.complexity.value}")
            print(f"   Agents: {[agent.name for agent in crew_spec.agents]}")
            print(f"   Process: {crew_spec.process_type}")
        
        # Create the crew
        crew_model = self.crew_designer.create_crew_from_spec(crew_spec, reuse_agents)
        
        if verbose:
            print(f"✅ Crew created with ID: {crew_model.id}")
        
        return crew_model
    
    def _update_crew_execution_stats(self, crew_id: str, execution_time: int, status: str):
        """Update crew execution statistics in cache."""
        import pickle
        import os
        from datetime import datetime
        
        stats_file = "/tmp/crewaimaster_execution_stats.pkl"
        
        # Load existing stats
        try:
            if os.path.exists(stats_file):
                with open(stats_file, 'rb') as f:
                    stats = pickle.load(f)
            else:
                stats = {}
        except Exception:
            stats = {}
        
        # Initialize crew stats if not exists
        if crew_id not in stats:
            stats[crew_id] = {
                'total_executions': 0,
                'successful_executions': 0,
                'total_execution_time': 0,
                'last_executed': None,
                'execution_history': []
            }
        
        # Update stats
        crew_stats = stats[crew_id]
        crew_stats['total_executions'] += 1
        if status == "completed":
            crew_stats['successful_executions'] += 1
        crew_stats['total_execution_time'] += execution_time
        crew_stats['last_executed'] = datetime.now().isoformat()
        crew_stats['execution_history'].append({
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'status': status
        })
        
        # Keep only last 50 executions
        if len(crew_stats['execution_history']) > 50:
            crew_stats['execution_history'] = crew_stats['execution_history'][-50:]
        
        # Save stats
        try:
            with open(stats_file, 'wb') as f:
                pickle.dump(stats, f)
        except Exception:
            pass  # Ignore save errors
    
    def _get_crew_execution_stats(self, crew_id: str) -> Dict[str, Any]:
        """Get crew execution statistics from cache."""
        import pickle
        import os
        
        stats_file = "/tmp/crewaimaster_execution_stats.pkl"
        
        try:
            if os.path.exists(stats_file):
                with open(stats_file, 'rb') as f:
                    stats = pickle.load(f)
                    return stats.get(crew_id, {
                        'total_executions': 0,
                        'successful_executions': 0,
                        'total_execution_time': 0,
                        'last_executed': None,
                        'execution_history': []
                    })
        except Exception:
            pass
        
        return {
            'total_executions': 0,
            'successful_executions': 0,
            'total_execution_time': 0,
            'last_executed': None,
            'execution_history': []
        }

    def execute_crew(self, crew_id: str, input_data: Optional[str] = None, 
                    verbose: bool = False) -> CrewExecutionResult:
        """Execute a crew by its ID."""
        # Get crew from cache first, then database
        crew_model = self.crew_designer.get_crew_from_cache(crew_id)
        if not crew_model:
            crew_model = self.crew_repo.get_crew(crew_id)
        if not crew_model:
            raise ValueError(f"Crew with ID {crew_id} not found")
        
        # Get CrewAI instance
        crewai_crew = self.crew_designer.get_crewai_instance(crew_id)
        if not crewai_crew:
            raise ValueError(f"CrewAI instance for crew {crew_id} not found")
        
        # Start execution logging (disabled for now due to database issues)
        start_time = datetime.now(timezone.utc)
        execution_log = None  # Skip database logging
        
        logs = []
        if verbose:
            logs.append(LogEntry(start_time, f"Starting execution of crew {crew_model.name}"))
        
        try:
            # Prepare input for CrewAI - inject user input directly into Task descriptions
            if input_data:
                if verbose:
                    logs.append(LogEntry(datetime.now(timezone.utc), f"Injecting user input into tasks: {input_data}"))
                # Update task descriptions to include user input
                for task in crewai_crew.tasks:
                    original_description = task.description
                    enhanced_task = f"{original_description}\n\nADDITIONAL CONTEXT: {input_data}\n\nIMPORTANT: Incorporate the ADDITIONAL CONTEXT above into your task execution. This input provides specific instructions, focus areas, or parameters that should guide your work."
                    task.description = enhanced_task
                    if verbose:
                        logs.append(LogEntry(datetime.now(timezone.utc), f"Enhanced task description for: {task.description[:100]}..."))
                inputs = {}
            else:
                inputs = {}
            
            if verbose:
                logs.append(LogEntry(datetime.now(timezone.utc), "Executing CrewAI crew"))
            
            # Execute the crew
            result = crewai_crew.kickoff(inputs=inputs)
            
            # Calculate execution time
            end_time = datetime.now(timezone.utc)
            execution_time = int((end_time - start_time).total_seconds())
            
            if verbose:
                logs.append(LogEntry(end_time, f"Crew execution completed in {execution_time}s"))
            
            # Update execution log (disabled for now)
            if execution_log:  # Only update if we have a log
                self.execution_repo.update_log(execution_log.id, {
                    "status": "completed",
                    "task_output": str(result),
                    "execution_time": execution_time,
                    "completed_at": end_time
                })
            
            # Update statistics using cache-based tracking
            self._update_crew_execution_stats(crew_id, execution_time, "completed")
            
            return CrewExecutionResult(
                crew_id=crew_id,
                output=str(result),
                execution_time=execution_time,
                logs=[{"timestamp": log.timestamp, "message": log.message} for log in logs],
                status="completed"
            )
            
        except Exception as e:
            # Handle execution error
            end_time = datetime.now(timezone.utc)
            execution_time = int((end_time - start_time).total_seconds())
            
            error_message = str(e)
            
            # Update execution log with error (disabled for now)
            if execution_log:  # Only update if we have a log
                self.execution_repo.update_log(execution_log.id, {
                    "status": "failed",
                    "error_message": error_message,
                    "error_type": type(e).__name__,
                    "execution_time": execution_time,
                    "completed_at": end_time
                })
            
            if verbose:
                logs.append(LogEntry(end_time, f"Crew execution failed: {error_message}", "ERROR"))
            
            # Update statistics for failed execution
            self._update_crew_execution_stats(crew_id, execution_time, "failed")
            
            return CrewExecutionResult(
                crew_id=crew_id,
                output=f"Execution failed: {error_message}",
                execution_time=execution_time,
                logs=[{"timestamp": log.timestamp, "message": log.message} for log in logs],
                status="failed"
            )
    
    def list_crews(self, limit: int = 100, offset: int = 0) -> List[CrewModel]:
        """List all crews."""
        return self.crew_repo.get_crews(limit=limit, offset=offset)
    
    def get_crew(self, crew_id: str) -> Optional[CrewModel]:
        """Get a specific crew by ID."""
        return self.crew_repo.get_crew(crew_id)
    
    def delete_crew(self, crew_id: str) -> bool:
        """Delete a crew."""
        return self.crew_repo.delete_crew(crew_id)
    
    def list_agents(self, limit: int = 100, offset: int = 0) -> List[AgentModel]:
        """List all agents."""
        return self.agent_repo.get_agents(limit=limit, offset=offset)
    
    def get_agent(self, agent_id: str) -> Optional[AgentModel]:
        """Get a specific agent by ID."""
        return self.agent_repo.get_agent(agent_id)
    
    def search_similar_crews(self, task_description: str, limit: int = 5) -> List[CrewModel]:
        """Search for crews with similar tasks."""
        # Simple keyword-based search for now
        # In a more sophisticated implementation, we would use semantic similarity
        keywords = task_description.lower().split()[:3]  # Take first 3 words
        
        return self.crew_repo.search_crews(task_keywords=keywords)
    
    def get_crew_performance(self, crew_id: str) -> Dict[str, Any]:
        """Get performance metrics for a crew."""
        return self.crew_designer.get_crew_performance_metrics(crew_id)
    
    def clone_crew(self, crew_id: str, new_name: Optional[str] = None) -> Optional[CrewModel]:
        """Clone an existing crew."""
        return self.crew_designer.clone_crew(crew_id, new_name)
    
    def update_crew_config(self, crew_id: str, config_updates: Dict[str, Any]) -> bool:
        """Update crew configuration."""
        return self.crew_designer.update_crew_config(crew_id, config_updates)
    
    def get_execution_history(self, crew_id: Optional[str] = None, 
                            agent_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get execution history for crews or agents."""
        if crew_id:
            logs = self.execution_repo.get_logs_for_crew(crew_id, limit)
        elif agent_id:
            logs = self.execution_repo.get_logs_for_agent(agent_id, limit)
        else:
            # Get all recent logs - would need to implement this method
            logs = []
        
        return [
            {
                "id": log.id,
                "crew_id": log.crew_id,
                "agent_id": log.agent_id,
                "status": log.status,
                "execution_time": log.execution_time,
                "started_at": log.started_at,
                "completed_at": log.completed_at,
                "error_message": log.error_message
            }
            for log in logs
        ]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        crews = self.list_crews()
        agents = self.list_agents()
        
        total_executions = sum(crew.execution_count for crew in crews)
        active_crews = len([crew for crew in crews if crew.execution_count > 0])
        active_agents = len([agent for agent in agents if agent.usage_count > 0])
        
        return {
            "total_crews": len(crews),
            "total_agents": len(agents),
            "active_crews": active_crews,
            "active_agents": active_agents,
            "total_executions": total_executions,
            "avg_agents_per_crew": len(agents) / len(crews) if crews else 0,
            "database_url": self.config.database.url,
            "llm_model": self.config.llm.model
        }
    
    def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old execution logs and unused agents/crews."""
        # This would implement cleanup logic
        # For now, return placeholder statistics
        return {
            "logs_deleted": 0,
            "crews_deleted": 0,
            "agents_deleted": 0
        }
    
    def export_crew_config(self, crew_id: str) -> Optional[Dict[str, Any]]:
        """Export crew configuration for backup or sharing."""
        crew = self.get_crew(crew_id)
        if not crew:
            return None
        
        return {
            "crew": {
                "name": crew.name,
                "task": crew.task,
                "description": crew.description,
                "process_type": crew.process_type,
                "expected_output": crew.expected_output,
                "config": crew.task_config
            },
            "agents": [
                {
                    "name": agent.name,
                    "role": agent.role,
                    "goal": agent.goal,
                    "backstory": agent.backstory,
                    "tools": [tool.name for tool in agent.tools] if agent.tools else [],
                    "memory_type": agent.memory_type,
                    "config": {
                        "max_iter": agent.max_iter,
                        "allow_delegation": agent.allow_delegation,
                        "verbose": agent.verbose
                    }
                }
                for agent in crew.agents
            ],
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "crewaimaster_version": "0.1.0"
        }
    
    def import_crew_config(self, config_data: Dict[str, Any]) -> Optional[CrewModel]:
        """Import crew configuration from backup or sharing."""
        # This would implement import logic
        # For now, this is a placeholder
        return None