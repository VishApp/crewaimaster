"""
MasterAgent implementation using CrewAI crew architecture.

This replaces the traditional MasterAgent with a crew-based approach where
intelligent AI agents handle task analysis, agent design, and crew orchestration.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from .config import Config
from .llm_provider import LLMProviderFactory, get_llm_config_for_crewai
from ..agents.crew_orchestrator_agent import CrewOrchestratorAgent, CrewOrchestrationRequest
from ..agents.task_analyzer_agent import TaskAnalyzerAgent
from ..agents.agent_designer_agent import AgentDesignerAgent
from .crew_designer import CrewDesigner, CrewModel, AgentModel
from .task_analyzer import CrewSpec


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


class MasterAgentCrew:
    """CrewAI-based MasterAgent implementation."""
    
    def __init__(self, config: Config):
        """Initialize the master agent crew."""
        self.config = config
        # Environment variable loading is disabled - use .crewmaster/config.yaml only
        
        # Initialize AI agents with provider-specific configuration
        try:
            llm_config = get_llm_config_for_crewai(self.config)
        except ValueError as e:
            # Fallback to OpenAI if provider is unsupported
            print(f"Warning: {e}. Falling back to OpenAI configuration.")
            llm_config = {
                "model": self.config.llm.model,
                "api_key": self.config.llm.api_key,
                "base_url": getattr(self.config.llm, 'base_url', None)
            }
        
        self.crew_orchestrator = CrewOrchestratorAgent(llm_config)
        self.task_analyzer = TaskAnalyzerAgent(llm_config)
        self.agent_designer = AgentDesignerAgent(llm_config)
        
        # Initialize crew designer for file-based operations (no database needed)
        self.crew_designer = CrewDesigner(config)
        
        # Track operation modes
        self._use_ai_agents = True  # Flag to enable/disable AI agent usage
        
        # Analysis cache for reusing task analysis results
        self._analysis_cache = {}
        self._load_analysis_cache()
    
    def create_crew(self, task_description: str, crew_name: Optional[str] = None,
                   reuse_agents: bool = True, verbose: bool = False, 
                   use_ai_orchestration: bool = True) -> CrewModel:
        """
        Create a new crew for the given task using AI orchestration.
        
        Args:
            task_description: Natural language description of the task
            crew_name: Optional custom crew name
            reuse_agents: Whether to reuse existing agents
            verbose: Enable verbose output
            use_ai_orchestration: Use AI agents for crew creation
            
        Returns:
            CrewModel: Created crew model
        """
        if verbose:
            print(f"🔍 Creating crew with AI orchestration: {task_description}")
        
        # Check if we have cached analysis for this task
        cached_analysis = self._get_cached_analysis(task_description)
        if cached_analysis and verbose:
            print("📄 Found cached analysis for this task - optimizing crew creation")
        
        if use_ai_orchestration and self._use_ai_agents:
            return self._create_crew_with_ai(task_description, crew_name, reuse_agents, verbose)
        else:
            return self._create_crew_legacy(task_description, crew_name, reuse_agents, verbose)
    
    def _create_crew_with_ai(self, task_description: str, crew_name: Optional[str] = None,
                            reuse_agents: bool = True, verbose: bool = False) -> CrewModel:
        """Create crew using AI orchestration."""
        if verbose:
            print("🤖 Using AI orchestration for crew creation")
        
        # Create orchestration request
        request = CrewOrchestrationRequest(
            task_description=task_description,
            crew_name=crew_name,
            preferences={"reuse_agents": reuse_agents, "verbose": verbose},
            constraints=[],
            resources={"database": True, "tools": True}
        )
        
        try:
            # Execute AI orchestration
            orchestration_result = self.crew_orchestrator.orchestrate_crew_creation(request)
            
            # Debug the orchestration result
            print(f"🔧 DEBUG: Orchestration result agents: {orchestration_result.crew_spec.get('agents', [])}")
            
            if verbose:
                print("📋 AI orchestration completed:")
                print(f"   Crew: {orchestration_result.crew_spec['name']}")
                print(f"   Agents: {len(orchestration_result.crew_spec['agents'])}")
                print(f"   Complexity: {orchestration_result.crew_spec['complexity']}")
                for log_entry in orchestration_result.orchestration_log[-3:]:  # Show last 3 logs
                    print(f"   {log_entry}")
            
            # Convert orchestration result to CrewSpec
            crew_spec = self._convert_orchestration_to_spec(orchestration_result)
            
            # Create the crew using the existing crew designer
            crew_model = self.crew_designer.create_crew_from_spec(crew_spec, reuse_agents)
            
            if verbose:
                print(f"✅ AI-orchestrated crew created with ID: {crew_model.id}")
                print(f"📊 Predicted performance: {orchestration_result.estimated_performance}")
            
            return crew_model
            
        except Exception as e:
            print(f"🔧 DEBUG: AI orchestration failed: {str(e)}")
            if verbose:
                print(f"❌ AI analysis failed: {str(e)}")
                print("🔄 Using intelligent fallback with enhanced defaults")
            
            # Fallback to legacy creation
            return self._create_crew_legacy(task_description, crew_name, reuse_agents, verbose)
    
    def _create_crew_legacy(self, task_description: str, crew_name: Optional[str] = None,
                           reuse_agents: bool = True, verbose: bool = False) -> CrewModel:
        """Create crew using legacy approach."""
        if verbose:
            print("🔧 Using legacy crew creation approach")
        
        # Use the original task analyzer for legacy mode
        from .task_analyzer import TaskAnalyzer
        legacy_analyzer = TaskAnalyzer()
        
        # Analyze the task
        crew_spec = legacy_analyzer.analyze_task(task_description)
        
        # Override name if provided
        if crew_name:
            crew_spec.name = crew_name
        
        if verbose:
            print(f"📋 Legacy crew specification created:")
            print(f"   Name: {crew_spec.name}")
            print(f"   Complexity: {crew_spec.complexity.value}")
            print(f"   Agents: {[agent.name for agent in crew_spec.agents]}")
            print(f"   Process: {crew_spec.process_type}")
        
        # Create the crew
        crew_model = self.crew_designer.create_crew_from_spec(crew_spec, reuse_agents)
        
        if verbose:
            print(f"✅ Legacy crew created with ID: {crew_model.id}")
        
        return crew_model
    
    def _update_crew_execution_stats(self, crew_id: str, execution_time: int, status: str):
        """Update crew execution statistics in cache."""
        import pickle
        import os
        from datetime import datetime
        
        stats_file = "/tmp/crewmaster_execution_stats.pkl"
        
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
        
        stats_file = "/tmp/crewmaster_execution_stats.pkl"
        
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

    def _convert_orchestration_to_spec(self, orchestration_result) -> CrewSpec:
        """Convert orchestration result to CrewSpec."""
        from .task_analyzer import TaskComplexity, AgentSpec
        
        spec_data = orchestration_result.crew_spec
        
        # Convert agents
        agents = []
        for agent_data in spec_data['agents']:
            # Handle field name variations from AI output
            agent_name = agent_data.get('agentName') or agent_data.get('name', 'DefaultAgent')
            tools = agent_data.get('tools') or agent_data.get('required_tools', [])
            max_iterations = agent_data.get('maxIterations') or agent_data.get('max_iter', 5)
            allow_delegation = agent_data.get('allowDelegation', agent_data.get('allow_delegation', False))
            memory_type = agent_data.get('memoryType', agent_data.get('memory_type', 'short_term'))
            
            agent_spec = AgentSpec(
                role=agent_data['role'],
                name=agent_name,
                goal=agent_data['goal'],
                backstory=agent_data['backstory'],
                required_tools=tools,
                memory_type=memory_type,
                max_iter=max_iterations,
                allow_delegation=allow_delegation
            )
            agents.append(agent_spec)
            print(f"🔧 DEBUG: Created agent spec - Name: {agent_name}, Role: {agent_data['role']}, Tools: {tools}")
        
        # Convert complexity
        complexity_map = {
            "simple": TaskComplexity.SIMPLE,
            "moderate": TaskComplexity.MODERATE,
            "complex": TaskComplexity.COMPLEX
        }
        complexity = complexity_map.get(spec_data['complexity'], TaskComplexity.MODERATE)
        
        return CrewSpec(
            name=spec_data['name'],
            task=spec_data['task'],
            description=spec_data['description'],
            agents=agents,
            expected_output=spec_data['expected_output'],
            complexity=complexity,
            estimated_time=spec_data['estimated_time'],
            process_type=spec_data['process_type']
        )
    
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
        
        # Start execution logging
        start_time = datetime.now(timezone.utc)
        execution_log = None  # Skip database logging for now
        
        logs = []
        if verbose:
            logs.append(LogEntry(start_time, f"Starting execution of crew {crew_model.name}"))
            if hasattr(crew_model, 'ai_enhanced') and crew_model.ai_enhanced:
                logs.append(LogEntry(datetime.now(timezone.utc), "Executing AI-enhanced crew"))
        
        try:
            # Prepare input for CrewAI - inject user input directly into Task descriptions
            if input_data:
                # Update task descriptions to include user input
                for task in crewai_crew.tasks:
                    enhanced_task = f"{task.description}\n\nUSER INPUT: {input_data}\n\nIMPORTANT: Use the information provided in the USER INPUT above. Extract any file paths, parameters, or specific instructions from the USER INPUT and use them in your task execution."
                    task.description = enhanced_task
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
            if execution_log:
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
    
    def analyze_task_with_ai(self, task_description: str, verbose: bool = False) -> Dict[str, Any]:
        """Analyze a task using AI agents with intelligent fallback."""
        # Check if we have a cached analysis for this task
        cached_analysis = self._get_cached_analysis(task_description)
        if cached_analysis:
            if verbose:
                print("📄 Found cached analysis for this task - reusing results")
            return cached_analysis
        
        if verbose:
            print("🔍 Analyzing task with AI...")
        
        try:
            crew_spec = self.task_analyzer.analyze_task(task_description)
            
            analysis_result = {
                "complexity": crew_spec.complexity.value,
                "estimated_time": crew_spec.estimated_time,
                "agent_count": len(crew_spec.agents),
                "agents": [
                    {
                        "role": agent.role,
                        "name": agent.name,
                        "tools": agent.required_tools
                    }
                    for agent in crew_spec.agents
                ],
                "process_type": crew_spec.process_type,
                "expected_output": crew_spec.expected_output
            }
            
            if verbose:
                print(f"📊 AI task analysis complete:")
                print(f"   Complexity: {analysis_result['complexity']}")
                print(f"   Estimated time: {analysis_result['estimated_time']} minutes")
                print(f"   Required agents: {analysis_result['agent_count']}")
            
            # Cache the successful analysis result
            self._cache_analysis_result(task_description, analysis_result)
            
            return analysis_result
            
        except Exception as e:
            if verbose:
                print(f"❌ AI task analysis failed: {str(e)}")
                print("🔄 Using intelligent fallback with enhanced analysis")
            
            # Fallback to simple AI-powered analysis using TaskAnalyzerAgent without LLM
            try:
                if verbose:
                    print("🔄 Using AI-powered fallback analysis (no LLM required)")
                
                # Use the AI agent's parsing logic with a simulated analysis
                crew_spec = self._create_smart_fallback_analysis(task_description)
                
                analysis_result = {
                    "complexity": crew_spec.complexity.value,
                    "estimated_time": crew_spec.estimated_time,
                    "agent_count": len(crew_spec.agents),
                    "agents": [
                        {
                            "role": agent.role,
                            "name": agent.name,
                            "tools": agent.required_tools,
                            "goal": agent.goal,
                            "backstory": agent.backstory
                        }
                        for agent in crew_spec.agents
                    ],
                    "process_type": crew_spec.process_type,
                    "expected_output": crew_spec.expected_output
                }
                
                if verbose:
                    print(f"📊 Smart fallback analysis complete:")
                    print(f"   Complexity: {analysis_result['complexity']}")
                    print(f"   Estimated time: {analysis_result['estimated_time']} minutes")
                    print(f"   Required agents: {analysis_result['agent_count']}")
                
                # Cache the successful fallback analysis result
                self._cache_analysis_result(task_description, analysis_result)
                
                return analysis_result
            
            except Exception as fallback_error:
                if verbose:
                    print(f"❌ Smart fallback analysis also failed: {str(fallback_error)}")
                return {"error": f"Both AI and fallback analysis failed: {str(e)}"}
    
    def _create_smart_fallback_analysis(self, task_description: str) -> 'CrewSpec':
        """Create intelligent fallback analysis using AI agent logic without LLM calls."""
        from .task_analyzer import CrewSpec, AgentSpec, TaskComplexity
        
        # Use the TaskAnalyzerAgent's parsing methods to create a smart analysis
        # This simulates what the AI would generate without requiring API calls
        
        # Normalize the task
        normalized_task = self.task_analyzer._normalize_task_description(task_description)
        
        # Create a simulated comprehensive AI analysis for parsing
        simulated_analysis = self._generate_simulated_ai_analysis(normalized_task)
        
        # Use TaskAnalyzerAgent's parsing methods to extract specifications
        return self.task_analyzer._parse_analysis_result(simulated_analysis, task_description, normalized_task)
    
    def _generate_simulated_ai_analysis(self, task: str) -> str:
        """Generate a simulated AI analysis with complete JSON specification that mimics real AI output."""
        task_lower = task.lower()
        
        # Generate proper JSON that will be parsed by the AI parsing methods
        if any(word in task_lower for word in ['monitor', 'track', 'price', 'watch']):
            return f'''
            ```json
            {{
                "taskComplexity": "moderate",
                "estimatedTime": 15,
                "processType": "sequential",
                "expectedOutput": "Real-time price monitoring data collected from Amazon with automated alerts for price changes",
                "agentSpecifications": [
                    {{
                        "role": "researcher",
                        "agentName": "PriceTracker",
                        "goal": "Actively monitor Amazon product prices and collect real-time price data",
                        "backstory": "You are a specialized price monitoring expert with extensive experience in web scraping and real-time data collection. You excel at tracking product prices across e-commerce platforms and delivering immediate results.",
                        "tools": ["web_scraping", "web_search", "data_processing", "file_operations"],
                        "memoryType": "short_term",
                        "maxIterations": 5,
                        "allowDelegation": false
                    }},
                    {{
                        "role": "analyst",
                        "agentName": "PriceAnalyst",
                        "goal": "Process collected price data and generate actionable insights and alerts",
                        "backstory": "You are a data analyst specialized in price analysis and trend detection. You excel at processing price data streams and identifying significant price changes that require immediate action.",
                        "tools": ["data_processing", "api_calls", "file_operations", "code_execution"],
                        "memoryType": "long_term",
                        "maxIterations": 3,
                        "allowDelegation": true
                    }}
                ],
                "crewNames": ["PriceWatch_Crew", "Amazon_Monitor_Team", "Price_Tracking_Squad"]
            }}
            ```
            '''
        elif any(word in task_lower for word in ['research', 'find', 'paper', 'study']):
            return f'''
            ```json
            {{
                "taskComplexity": "complex",
                "estimatedTime": 30,
                "processType": "sequential",
                "expectedOutput": "Comprehensive research results with collected papers, summaries, and compiled findings",
                "agentSpecifications": [
                    {{
                        "role": "researcher",
                        "agentName": "ResearchScout",
                        "goal": "Actively search and collect the latest research papers and academic materials",
                        "backstory": "You are an expert academic researcher with deep experience in finding, evaluating, and collecting research papers. You excel at navigating academic databases and delivering high-quality research results.",
                        "tools": ["web_search", "document_search", "github_search", "file_operations"],
                        "memoryType": "long_term",
                        "maxIterations": 7,
                        "allowDelegation": false
                    }},
                    {{
                        "role": "analyst",
                        "agentName": "ContentProcessor",
                        "goal": "Extract and synthesize key insights from collected research materials and create data files",
                        "backstory": "You are a content analysis specialist with expertise in processing research data and creating structured files. You excel at analyzing web content, creating CSV/PDF files with findings, and generating actionable insights.",
                        "tools": ["web_search", "code_execution", "file_operations", "data_processing"],
                        "memoryType": "long_term",
                        "maxIterations": 5,
                        "allowDelegation": true
                    }},
                    {{
                        "role": "writer",
                        "agentName": "ReportBuilder",
                        "goal": "Compile research findings into structured, comprehensive reports and create final documents",
                        "backstory": "You are a technical writer specialized in creating research reports and generating final documents. You excel at organizing complex information, creating PDF/DOCX reports, and delivering professional documentation with proper formatting.",
                        "tools": ["file_operations", "code_execution", "web_search", "data_processing"],
                        "memoryType": "short_term",
                        "maxIterations": 3,
                        "allowDelegation": false
                    }}
                ],
                "crewNames": ["Research_Brigade", "Academic_Explorer_Team", "Knowledge_Hunters"]
            }}
            ```
            '''
        else:
            return f'''
            ```json
            {{
                "taskComplexity": "simple",
                "estimatedTime": 10,
                "processType": "sequential",
                "expectedOutput": "Complete task execution results with all deliverables and outcomes",
                "agentSpecifications": [
                    {{
                        "role": "specialist",
                        "agentName": "TaskExecutor",
                        "goal": "Execute the task efficiently and deliver complete results",
                        "backstory": "You are a versatile task execution specialist with broad experience in handling diverse assignments. You excel at understanding requirements and delivering comprehensive results efficiently.",
                        "tools": ["web_search", "file_operations", "data_processing"],
                        "memoryType": "short_term",
                        "maxIterations": 5,
                        "allowDelegation": false
                    }}
                ],
                "crewNames": ["Task_Force", "Execution_Team", "Delivery_Squad"]
            }}
            ```
            '''
    
    def _get_task_cache_key(self, task_description: str) -> str:
        """Generate a cache key from task description."""
        import hashlib
        # Normalize the task description for consistent caching
        normalized_task = task_description.lower().strip()
        return hashlib.md5(normalized_task.encode()).hexdigest()
    
    def _cache_analysis_result(self, task_description: str, analysis_result: Dict[str, Any]):
        """Cache the analysis result for a task."""
        cache_key = self._get_task_cache_key(task_description)
        self._analysis_cache[cache_key] = {
            'analysis': analysis_result,
            'task_description': task_description,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'cache_version': '1.0'
        }
        self._save_analysis_cache()
    
    def _get_cached_analysis(self, task_description: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result for a task."""
        cache_key = self._get_task_cache_key(task_description)
        cached_data = self._analysis_cache.get(cache_key)
        
        if cached_data:
            # Check if cache is not too old (24 hours)
            from datetime import datetime, timedelta
            cache_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now(timezone.utc) - cache_time < timedelta(hours=24):
                return cached_data['analysis']
            else:
                # Remove expired cache entry
                del self._analysis_cache[cache_key]
                self._save_analysis_cache()
        
        return None
    
    def _load_analysis_cache(self):
        """Load analysis cache from file."""
        try:
            import pickle
            import os
            cache_file = "/tmp/crewmaster_analysis_cache.pkl"
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self._analysis_cache = pickle.load(f)
        except Exception:
            self._analysis_cache = {}
    
    def _save_analysis_cache(self):
        """Save analysis cache to file."""
        try:
            import pickle
            cache_file = "/tmp/crewmaster_analysis_cache.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(self._analysis_cache, f)
        except Exception:
            pass  # Ignore cache save errors
    
    def clear_analysis_cache(self) -> Dict[str, Any]:
        """Clear all cached analysis results."""
        cache_count = len(self._analysis_cache)
        self._analysis_cache.clear()
        self._save_analysis_cache()
        
        return {
            "cleared_entries": cache_count,
            "message": f"Cleared {cache_count} cached analysis entries"
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the analysis cache."""
        from datetime import datetime, timedelta
        
        total_entries = len(self._analysis_cache)
        valid_entries = 0
        expired_entries = 0
        
        for cache_key, cached_data in self._analysis_cache.items():
            try:
                cache_time = datetime.fromisoformat(cached_data['timestamp'])
                if datetime.now(timezone.utc) - cache_time < timedelta(hours=24):
                    valid_entries += 1
                else:
                    expired_entries += 1
            except (KeyError, ValueError):
                expired_entries += 1
        
        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cache_file": "/tmp/crewmaster_analysis_cache.pkl"
        }
    
    def list_cached_tasks(self) -> List[Dict[str, Any]]:
        """List all cached task analyses."""
        from datetime import datetime, timedelta
        
        cached_tasks = []
        current_time = datetime.now(timezone.utc)
        
        for cache_key, cached_data in self._analysis_cache.items():
            try:
                cache_time = datetime.fromisoformat(cached_data['timestamp'])
                age_hours = (current_time - cache_time).total_seconds() / 3600
                is_expired = age_hours >= 24
                
                task_info = {
                    "task_description": cached_data.get('task_description', 'Unknown'),
                    "cached_at": cached_data.get('timestamp'),
                    "age_hours": round(age_hours, 1),
                    "is_expired": is_expired,
                    "complexity": cached_data.get('analysis', {}).get('complexity', 'Unknown'),
                    "agent_count": cached_data.get('analysis', {}).get('agent_count', 0)
                }
                cached_tasks.append(task_info)
            except (KeyError, ValueError, TypeError):
                continue
        
        # Sort by most recent first
        cached_tasks.sort(key=lambda x: x['cached_at'], reverse=True)
        return cached_tasks
    
    def clear_expired_cache(self) -> Dict[str, Any]:
        """Clear only expired cache entries."""
        from datetime import datetime, timedelta
        
        expired_keys = []
        current_time = datetime.now(timezone.utc)
        
        for cache_key, cached_data in self._analysis_cache.items():
            try:
                cache_time = datetime.fromisoformat(cached_data['timestamp'])
                if current_time - cache_time >= timedelta(hours=24):
                    expired_keys.append(cache_key)
            except (KeyError, ValueError):
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self._analysis_cache[key]
        
        if expired_keys:
            self._save_analysis_cache()
        
        return {
            "cleared_entries": len(expired_keys),
            "message": f"Cleared {len(expired_keys)} expired cache entries"
        }
    
    def set_ai_mode(self, enabled: bool):
        """Enable or disable AI agent usage."""
        self._use_ai_agents = enabled
    
    def get_ai_mode(self) -> bool:
        """Check if AI agent mode is enabled."""
        return self._use_ai_agents
    
    def modify_with_ai(self, target_type: str, target_name: str, modification_request: str, verbose: bool = False) -> Dict[str, Any]:
        """Use AI agents to intelligently modify a crew or agent."""
        if not self._use_ai_agents:
            return {"success": False, "error": "AI agents are disabled"}
        
        # Check for OpenAI API key availability
        import os
        if not os.getenv('OPENAI_API_KEY'):
            if verbose:
                print("🔧 DEBUG: No OpenAI API key found, using direct modification without AI analysis")
            # Generate modification plan directly without AI
            modification_plan = self._generate_direct_modification_plan(target_type, target_name, modification_request)
            return {
                "success": True,
                "ai_analysis": "Direct modification (no AI analysis - missing OpenAI API key)",
                "modification_plan": modification_plan,
                "target_type": target_type,
                "target_name": target_name
            }
        
        try:
            # Create the modification crew
            modification_task = f"""
            Analyze and implement the following modification request:
            
            Target Type: {target_type}
            Target Name: {target_name}
            Modification Request: {modification_request}
            
            Your task is to:
            1. Understand what needs to be modified
            2. Determine the specific changes needed
            3. Generate a modification plan
            4. Provide clear instructions for implementation
            
            Focus on being specific and actionable in your response.
            """
            
            # Create a specialized crew for modification analysis
            from .task_analyzer import CrewSpec, AgentSpec, TaskComplexity
            import uuid
            
            # Generate unique names to avoid collisions
            unique_id = str(uuid.uuid4())[:8]
            crew_name = f"modification_analysis_{unique_id}"
            agent_name = f"ModificationSpecialist_{unique_id}"
            
            modification_agent = AgentSpec(
                role="modification_specialist",
                name=agent_name,
                goal="Analyze and plan intelligent modifications to crews and agents",
                backstory="You are an expert in understanding natural language modification requests and translating them into specific actionable changes for multi-agent systems.",
                required_tools=["web_search", "file_operations"]
            )
            
            crew_spec = CrewSpec(
                name=crew_name,
                task=modification_task,
                description="Analyze and plan modifications to crews and agents",
                agents=[modification_agent],
                expected_output="A detailed modification plan with specific changes to implement",
                complexity=TaskComplexity.SIMPLE,
                estimated_time=5
            )
            
            # Create and execute the modification crew
            crew_model = self.crew_designer.create_crew_from_spec(crew_spec, reuse_agents=False)
            crewai_crew = self.crew_designer.get_crewai_instance(crew_spec.name)
            
            if not crewai_crew:
                return {"success": False, "error": "Failed to create modification analysis crew"}
            
            # Execute the analysis
            result = crewai_crew.kickoff(inputs={})
            
            # Parse the AI's response to extract modification plan
            ai_analysis = str(result).strip()
            
            # Extract actionable modifications from AI response
            modification_plan = self._parse_ai_modification_response(ai_analysis, target_type, modification_request)
            
            # Clean up the temporary crew
            try:
                if crew_name in self.crew_designer._crews_cache:
                    del self.crew_designer._crews_cache[crew_name]
                if crew_name in self.crew_designer._crewai_instances:
                    del self.crew_designer._crewai_instances[crew_name]
                self.crew_designer._save_cache()
            except Exception:
                pass  # Don't fail if cleanup fails
            
            return {
                "success": True,
                "ai_analysis": ai_analysis,
                "modification_plan": modification_plan,
                "target_type": target_type,
                "target_name": target_name
            }
            
        except Exception as e:
            # Clean up any temporary crews in case of error
            try:
                if 'crew_name' in locals() and crew_name in self.crew_designer._crews_cache:
                    del self.crew_designer._crews_cache[crew_name]
                if 'crew_name' in locals() and crew_name in self.crew_designer._crewai_instances:
                    del self.crew_designer._crewai_instances[crew_name]
                self.crew_designer._save_cache()
            except Exception:
                pass
            
            return {"success": False, "error": str(e)}
    
    def _parse_ai_modification_response(self, ai_response: str, target_type: str, original_request: str) -> Dict[str, Any]:
        """Parse the AI's modification analysis response into actionable plan."""
        plan = {"steps": [], "actions": []}
        
        print(f"🔧 DEBUG: AI Response: {ai_response[:100]}...")
        print(f"🔧 DEBUG: Target type: {target_type}, Request: {original_request}")
        
        # Check if the AI response contains an error (like missing API key)
        if "error" in ai_response.lower() or "failed" in ai_response.lower() or len(ai_response.strip()) < 10:
            print(f"🔧 DEBUG: AI response appears to be an error or empty, using fallback logic")
            # Generate a standard modification plan based on the request
            if target_type == "crew":
                plan["steps"].append(f"Update crew task: {original_request}")
                plan["actions"].append({"type": "update_property", "property": "task", "value": original_request})
                
                plan["steps"].append(f"Update crew description to match new task")
                plan["actions"].append({"type": "update_property", "property": "description", "value": f"AI-updated crew for: {original_request}"})
                
                plan["steps"].append(f"Recreate agents with roles appropriate for new task")
                plan["actions"].append({"type": "recreate_agents_for_task", "value": original_request})
                
                print(f"🔧 DEBUG: Generated {len(plan['actions'])} fallback actions for crew modification")
            else:  # agent
                plan["steps"].append(f"Update agent goal: {original_request}")
                plan["actions"].append({"type": "update_property", "property": "goal", "value": original_request})
        else:
            # Parse actual AI response - look for key modification indicators
            response_lower = ai_response.lower()
            
            # Check if AI suggests updating task/goal
            keywords_found = [word for word in ['task', 'goal', 'objective', 'update', 'change'] if word in response_lower]
            print(f"🔧 DEBUG: Found keywords: {keywords_found}")
            
            if any(word in response_lower for word in ['task', 'goal', 'objective', 'update', 'change']):
                if target_type == "crew":
                    # For crews, update multiple properties
                    plan["steps"].append(f"Update crew task: {original_request}")
                    plan["actions"].append({"type": "update_property", "property": "task", "value": original_request})
                    
                    plan["steps"].append(f"Update crew description to match new task")
                    plan["actions"].append({"type": "update_property", "property": "description", "value": f"AI-updated crew for: {original_request}"})
                    
                    plan["steps"].append(f"Recreate agents with roles appropriate for new task")
                    plan["actions"].append({"type": "recreate_agents_for_task", "value": original_request})
                    
                    print(f"🔧 DEBUG: Generated {len(plan['actions'])} actions for crew modification")
                else:  # agent
                    plan["steps"].append(f"Update agent goal based on AI analysis: {original_request}")
                    plan["actions"].append({"type": "update_property", "property": "goal", "value": original_request})
            else:
                print(f"🔧 DEBUG: No modification keywords found in AI response")
        
        # Note: Tools will be assigned automatically during agent recreation
        print(f"🔧 DEBUG: Final plan has {len(plan['steps'])} steps")
        
        return plan
    
    def _generate_direct_modification_plan(self, target_type: str, target_name: str, modification_request: str) -> Dict[str, Any]:
        """Generate modification plan directly without AI analysis."""
        plan = {"steps": [], "actions": []}
        
        print(f"🔧 DEBUG: Generating direct modification plan")
        print(f"🔧 DEBUG: Target: {target_type} '{target_name}', Request: {modification_request}")
        
        if target_type == "crew":
            # For crews, update task and recreate agents
            plan["steps"].append(f"Update crew task: {modification_request}")
            plan["actions"].append({"type": "update_property", "property": "task", "value": modification_request})
            
            plan["steps"].append(f"Update crew description to match new task")
            plan["actions"].append({"type": "update_property", "property": "description", "value": f"AI-updated crew for: {modification_request}"})
            
            plan["steps"].append(f"Recreate agents with roles appropriate for new task")
            plan["actions"].append({"type": "recreate_agents_for_task", "value": modification_request})
            
            print(f"🔧 DEBUG: Generated {len(plan['actions'])} direct actions for crew modification")
        else:  # agent
            plan["steps"].append(f"Update agent goal: {modification_request}")
            plan["actions"].append({"type": "update_property", "property": "goal", "value": modification_request})
            
            print(f"🔧 DEBUG: Generated {len(plan['actions'])} direct actions for agent modification")
        
        return plan
    
    # Delegate all other methods to the original implementation
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
        keywords = task_description.lower().split()[:3]
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
        
        # Add AI enhancement stats
        ai_enhanced_crews = len([crew for crew in crews if hasattr(crew, 'ai_enhanced') and crew.ai_enhanced])
        
        return {
            "total_crews": len(crews),
            "total_agents": len(agents),
            "active_crews": active_crews,
            "active_agents": active_agents,
            "ai_enhanced_crews": ai_enhanced_crews,
            "total_executions": total_executions,
            "avg_agents_per_crew": len(agents) / len(crews) if crews else 0,
            "database_url": self.config.database.url,
            "llm_model": self.config.llm.model,
            "ai_mode_enabled": self._use_ai_agents
        }
    
    def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old execution logs and unused agents/crews."""
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
        
        config = {
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
            "crewmaster_version": "0.1.0",
            "ai_enhanced": hasattr(crew, 'ai_enhanced') and crew.ai_enhanced
        }
        
        return config
    
    def import_crew_config(self, config_data: Dict[str, Any]) -> Optional[CrewModel]:
        """Import crew configuration from backup or sharing."""
        # This would implement import logic
        return None