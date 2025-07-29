"""
Crew Designer module for CrewMaster.

This module handles the creation and management of CrewAI crews based on 
task analysis and agent specifications.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from crewai import Agent, Crew, Task, Process
from crewai.tools import BaseTool

from .task_analyzer import CrewSpec, AgentSpec
from .config import Config

# Simple data structures to replace database models
class AgentModel:
    def __init__(self):
        self.id = None
        self.name = None
        self.role = None
        self.goal = None
        self.backstory = None
        self.required_tools = []
        self.tools = []
        self.memory_type = 'short_term'
        self.max_iter = 5
        self.allow_delegation = False
        self.verbose = True
        self.memory_enabled = False
        self.usage_count = 0
        self.success_rate = 0.0
        self.avg_execution_time = 0.0

class CrewModel:
    def __init__(self):
        self.id = None
        self.name = None
        self.task = None
        self.description = None
        self.agents = []
        self.expected_output = None
        self.complexity = None
        self.estimated_time = 0
        self.process_type = 'sequential'
        self.verbose = True
        self.memory_enabled = False
        self.execution_count = 0
        self.success_rate = 0.0
        self.avg_execution_time = 0.0
        self.last_executed = None
        self.task_config = {}
        self.ai_enhanced = True

class CrewDesigner:
    """Designs and creates CrewAI crews from specifications."""
    
    def __init__(self, config: Config, database=None):
        """Initialize the crew designer."""
        self.config = config
        self.db = database  # Optional database, will be None when database layer is removed
        
        # Initialize repositories only if database is available
        if database:
            from ..database.database import AgentRepository, CrewRepository, ToolRepository
            self.agent_repo = AgentRepository(database)
            self.crew_repo = CrewRepository(database)
            self.tool_repo = ToolRepository(database)
        else:
            self.agent_repo = None
            self.crew_repo = None
            self.tool_repo = None
            
        from ..tools.registry import ToolRegistry
        self.tool_registry = ToolRegistry()
        
        # CrewAI process mapping
        self.process_mapping = {
            "sequential": Process.sequential,
            "hierarchical": Process.hierarchical,
        }
        
        # In-memory storage for crews (temporary fix for database issues)
        self._crews_cache = {}
        self._crewai_instances = {}
        
        # Try to load from persistent cache
        self._load_cache()
    
    def create_crew_from_spec(self, spec: CrewSpec, reuse_agents: bool = True) -> CrewModel:
        """Create a new crew from a crew specification."""
        print(f"🔧 DEBUG: Creating crew from spec with {len(spec.agents)} agent specifications")
        for i, agent_spec in enumerate(spec.agents):
            print(f"🔧 DEBUG: Agent spec {i+1}: {agent_spec.role} - {agent_spec.name}")
        
        # Create or reuse agents
        agent_models = []
        crewai_agents = []
        
        for agent_spec in spec.agents:
            if reuse_agents:
                # Try to find existing similar agent
                existing_agent = self._find_similar_agent(agent_spec)
                if existing_agent:
                    agent_models.append(existing_agent)
                    self.agent_repo.increment_usage(existing_agent.id)
                    crewai_agents.append(self._create_crewai_agent_from_model(existing_agent))
                    continue
            
            # Create new agent
            agent_model = self._create_new_agent(agent_spec)
            agent_models.append(agent_model)
            crewai_agents.append(self._create_crewai_agent_from_spec(agent_spec))
        
        # Create CrewAI tasks
        print(f"🔧 DEBUG: About to create tasks with {len(crewai_agents)} agents")
        for i, agent in enumerate(crewai_agents):
            print(f"🔧 DEBUG: Agent {i+1}: {agent.role}")
        crewai_tasks = self._create_tasks(spec, crewai_agents)
        
        # Get LLM configuration for CrewAI
        from .llm_provider import get_llm_config_for_crewai
        try:
            llm_config = get_llm_config_for_crewai(self.config.get())
        except Exception as e:
            # Fallback configuration
            llm_config = {
                "model": self.config.get().llm.model,
                "api_key": self.config.get().llm.api_key,
                "base_url": self.config.get().llm.base_url
            }
        
        # Create CrewAI crew with LLM configuration
        process = self.process_mapping.get(spec.process_type, Process.sequential)
        
        # Create CrewAI crew (LLM configuration is now handled at agent level)
        print(f"🔧 DEBUG: Creating CrewAI crew with {len(crewai_agents)} agents and {len(crewai_tasks)} tasks")
        print(f"🔧 DEBUG: Process type: {process}")
        
        crewai_crew = Crew(
            agents=crewai_agents,
            tasks=crewai_tasks,
            process=process,  # Should be Process.sequential for collaboration
            verbose=self.config.get().default_agent_verbose,
            memory=False  # Disable memory for now to avoid CHROMA_OPENAI_API_KEY requirement
        )
        
        # Save crew to database
        crew_data = {
            "name": spec.name,
            "task": spec.task,
            "description": spec.description,
            "process_type": spec.process_type,
            "verbose": self.config.get().default_agent_verbose,
            "memory_enabled": self.config.memory.enabled,
            "expected_output": spec.expected_output,
            "task_config": {
                "complexity": spec.complexity.value,
                "estimated_time": spec.estimated_time
            }
        }
        
        # Extract agent IDs before potential session detachment
        agent_ids = []
        for agent in agent_models:
            agent_ids.append(agent.id)
        
        # Create crew model using local CrewModel class
        
        # Check if crew name already exists
        if any(crew.name == spec.name for crew in self._crews_cache.values()):
            raise ValueError(f"Crew with name '{spec.name}' already exists. Please choose a different name.")
        
        crew_model = CrewModel()
        crew_model.id = spec.name  # Use name as unique identifier
        crew_model.name = spec.name
        crew_model.task = spec.task
        crew_model.description = spec.description
        crew_model.agents = agent_models
        
        # Store in cache for retrieval (using name as key)
        self._crews_cache[spec.name] = crew_model
        
        # Store the CrewAI crew instance for execution
        self._store_crewai_instance(spec.name, crewai_crew)
        
        # Save cache to file for persistence
        self._save_cache()
        
        return crew_model
    
    def get_crew_from_cache(self, crew_id: str) -> Optional[CrewModel]:
        """Get crew from in-memory cache."""
        return self._crews_cache.get(crew_id)
    
    def _find_similar_agent(self, agent_spec: AgentSpec) -> Optional[AgentModel]:
        """Find existing agent with similar role and capabilities."""
        # Disable agent reuse for now to avoid SQLAlchemy session issues
        # TODO: Fix SQLAlchemy session management for proper agent reuse
        return None
    
    def _create_new_agent(self, agent_spec: AgentSpec) -> AgentModel:
        """Create a new agent from specification."""
        # Create agent model using local AgentModel class
        
        agent_model = AgentModel()
        agent_model.id = agent_spec.name  # Use name as unique identifier
        agent_model.name = agent_spec.name
        agent_model.role = agent_spec.role
        agent_model.goal = agent_spec.goal
        agent_model.backstory = agent_spec.backstory
        agent_model.required_tools = agent_spec.required_tools  # Store the tools
        
        return agent_model
    
    def _create_crewai_agent_from_spec(self, agent_spec: AgentSpec) -> Agent:
        """Create a CrewAI Agent from specification with enhanced tool usage instructions."""
        # Get tools for this agent
        tools = self._get_tools_for_agent(agent_spec.required_tools)
        
        # Enhance goal and backstory to emphasize tool usage and current information
        enhanced_goal = self._enhance_goal_for_tool_usage(agent_spec.goal, agent_spec.required_tools)
        enhanced_backstory = self._enhance_backstory_for_current_data(agent_spec.backstory)
        
        # Check if we're in tool-only mode (no LLM API key)
        llm_config = self.config.get().llm
        is_tool_only_mode = llm_config.api_key == "TOOL_ONLY_MODE"
        
        if is_tool_only_mode:
            # Create a special agent that focuses on tool execution
            enhanced_goal = self._create_tool_only_goal(agent_spec.goal, agent_spec.required_tools)
            enhanced_backstory = self._create_tool_only_backstory(agent_spec.backstory)
            
            return Agent(
                role=agent_spec.role,
                goal=enhanced_goal,
                backstory=enhanced_backstory,
                tools=tools,
                verbose=True,  # Force verbose to show tool usage
                allow_delegation=False,  # Disable delegation in tool-only mode
                max_iter=1,  # Single iteration to force tool usage
                memory=False,
                system_template=self._get_tool_only_system_template()
            )
        else:
            # Get LLM configuration for this agent
            from .llm_provider import get_llm_config_for_crewai
            try:
                config = self.config.get()
                provider = config.llm.provider
                print(f"🔧 DEBUG: Provider: {provider}, Model: {config.llm.model}")
                
                from crewai import LLM
                
                if provider == "custom":
                    # For custom providers, we need to specify all parameters
                    agent_llm = LLM(
                        model=config.llm.model,
                        api_key=config.llm.api_key,
                        base_url=config.llm.base_url,
                        temperature=config.llm.temperature,
                        max_tokens=config.llm.max_tokens
                    )
                    print(f"🔧 DEBUG: Created custom LLM with base_url={config.llm.base_url}")
                else:
                    # For standard providers (openai, anthropic, google), let CrewAI auto-detect
                    # It will use OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY env vars
                    agent_llm = LLM(
                        model=config.llm.model,
                        temperature=config.llm.temperature,
                        max_tokens=config.llm.max_tokens
                    )
                    print(f"🔧 DEBUG: Created standard LLM for {provider} with model={config.llm.model}")
            except Exception as e:
                print(f"🔧 DEBUG: Failed to create LLM instance: {e}")
                # Fallback - let CrewAI use default configuration
                agent_llm = None
            
            agent_kwargs = {
                "role": agent_spec.role,
                "goal": enhanced_goal,
                "backstory": enhanced_backstory,
                "tools": tools,
                "verbose": self.config.get().default_agent_verbose,
                "allow_delegation": agent_spec.allow_delegation,
                "max_iter": agent_spec.max_iter,
                "memory": False  # Disable memory for now
            }
            
            # Add LLM if available
            if agent_llm:
                agent_kwargs["llm"] = agent_llm
                
            return Agent(**agent_kwargs)
    
    def _create_crewai_agent_from_model(self, agent_model: AgentModel) -> Agent:
        """Create a CrewAI Agent from database model."""
        # Get tools for this agent
        tool_names = [tool.name for tool in agent_model.tools] if agent_model.tools else []
        tools = self._get_tools_for_agent(tool_names)
        
        # Get LLM configuration for this agent
        from .llm_provider import get_llm_config_for_crewai
        try:
            config = self.config.get()
            provider = config.llm.provider
            print(f"🔧 DEBUG: Model agent - Provider: {provider}, Model: {config.llm.model}")
            
            from crewai import LLM
            
            if provider == "custom":
                # For custom providers, we need to specify all parameters
                agent_llm = LLM(
                    model=config.llm.model,
                    api_key=config.llm.api_key,
                    base_url=config.llm.base_url,
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens
                )
                print(f"🔧 DEBUG: Created custom model LLM with base_url={config.llm.base_url}")
            else:
                # For standard providers (openai, anthropic, google), let CrewAI auto-detect
                agent_llm = LLM(
                    model=config.llm.model,
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens
                )
                print(f"🔧 DEBUG: Created standard model LLM for {provider}")
        except Exception as e:
            print(f"🔧 DEBUG: Failed to create model LLM instance: {e}")
            # Fallback - let CrewAI use default configuration
            agent_llm = None
        
        agent_kwargs = {
            "role": agent_model.role,
            "goal": agent_model.goal,
            "backstory": agent_model.backstory,
            "tools": tools,
            "verbose": agent_model.verbose,
            "allow_delegation": agent_model.allow_delegation,
            "max_iter": agent_model.max_iter,
            "memory": agent_model.memory_enabled
        }
        
        # Add LLM if available
        if agent_llm:
            agent_kwargs["llm"] = agent_llm
            
        return Agent(**agent_kwargs)
    
    def _get_tools_for_agent(self, tool_names: List[str]) -> List[Any]:
        """Get CrewAI tool instances for given tool names using the tool registry."""
        from ..tools.registry import ToolRegistry
        from crewai.tools import tool
        
        # Initialize tool registry
        tool_registry = ToolRegistry()
        tools = []
        
        for tool_name in tool_names:
            # Try to get real tool from registry first
            real_tool = tool_registry.get_tool(tool_name)
            
            if real_tool and not self._is_mock_tool(real_tool):
                # Use the real tool - wrap it to be CrewAI compatible
                print(f"🔧 DEBUG: Using REAL {tool_name} tool: {type(real_tool).__name__}")
                wrapped_tool = self._wrap_tool_for_crewai(real_tool, tool_name)
                tools.append(wrapped_tool)
            else:
                # Fallback to enhanced mock tools with better descriptions
                print(f"🔧 DEBUG: Using MOCK {tool_name} tool (real tool not available)")
                mock_tool = self._create_enhanced_mock_tool(tool_name)
                if mock_tool:
                    tools.append(mock_tool)
        
        return tools
    
    def _is_mock_tool(self, tool_instance) -> bool:
        """Check if a tool is a mock tool."""
        if tool_instance is None:
            return True
        
        tool_class_name = tool_instance.__class__.__name__
        tool_module = tool_instance.__class__.__module__
        
        # Check for explicit mock tools
        if 'Mock' in tool_class_name:
            return True
        
        # Check if it's from the mock tools in registry
        if 'registry' in tool_module and 'Mock' in str(type(tool_instance)):
            return True
        
        # Real tools from langchain_community, crewai_tools, etc. are not mock
        if any(module in tool_module for module in ['langchain_community', 'crewai_tools', 'crewai.']):
            return False
        
        # If it has a proper run method and is not from mock modules, it's likely real
        if hasattr(tool_instance, 'run') and callable(getattr(tool_instance, 'run')):
            return False
            
        return False  # Default to not mock to prefer real tools
    
    def _wrap_tool_for_crewai(self, tool_instance, tool_name: str):
        """Wrap any tool to be CrewAI compatible."""
        from crewai.tools import tool
        
        # Check if it's already a CrewAI tool
        if hasattr(tool_instance, '__class__') and 'crewai' in str(type(tool_instance)).lower():
            return tool_instance
        
        # Handle different tool types
        tool_type = type(tool_instance).__name__
        
        if tool_type == 'DuckDuckGoSearchRun':
            @tool("Web Search Tool")
            def search_web(query: str) -> str:
                """Search the web for information using DuckDuckGo."""
                try:
                    result = tool_instance.run(query)
                    return f"🔍 Web Search Results for '{query}':\n\n{result}"
                except Exception as e:
                    return f"❌ Search failed: {str(e)}"
            return search_web
            
        elif tool_type == 'SerperDevTool':
            @tool("Advanced Web Search Tool")
            def search_web_serper(query: str) -> str:
                """Search the web for information using Serper (more comprehensive results)."""
                try:
                    result = tool_instance.run(search_query=query)
                    return f"🔍 Advanced Search Results for '{query}':\n\n{result}"
                except Exception as e:
                    return f"❌ Advanced search failed: {str(e)}"
            return search_web_serper
            
        elif hasattr(tool_instance, 'run'):
            # Generic CrewAI tool wrapper
            @tool(f"{tool_name.replace('_', ' ').title()}")
            def generic_tool(input_data: str) -> str:
                """Use tool to process input data."""
                try:
                    result = tool_instance.run(input_data)
                    return f"🔧 {tool_name} Results:\n\n{result}"
                except Exception as e:
                    return f"❌ {tool_name} failed: {str(e)}"
            return generic_tool
        
        elif callable(tool_instance):
            # Handle callable tools
            @tool(f"{tool_name.replace('_', ' ').title()}")
            def callable_tool(input_data: str) -> str:
                """Use tool to process input data."""
                try:
                    result = tool_instance(input_data)
                    return f"🔧 {tool_name} Results:\n\n{result}"
                except Exception as e:
                    return f"❌ {tool_name} failed: {str(e)}"
            return callable_tool
        
        # If we can't wrap it, return None so it falls back to mock
        return None
    
    def _enhance_goal_for_tool_usage(self, original_goal: str, tool_names: List[str]) -> str:
        """Enhance agent goal to emphasize tool usage and current information."""
        from datetime import datetime
        
        current_date = datetime.now().strftime("%B %Y")
        current_year = datetime.now().year
        
        # Build tool usage instructions
        tool_instructions = ""
        if "web_search" in tool_names:
            tool_instructions += f"\n- MUST use web search tools to find the most current information from {current_year}"
        if "document_search" in tool_names:
            tool_instructions += f"\n- MUST use document search tools when analyzing papers or documents"
        if "github_search" in tool_names:
            tool_instructions += f"\n- MUST use GitHub search for code and repository information"
        
        enhanced_goal = f"""{original_goal}

CRITICAL INSTRUCTIONS (MUST FOLLOW):
- Current date: {current_date}
- ALWAYS search for information from {current_year} when looking for "latest" or "recent" content
- NEVER rely on pre-existing knowledge - ALWAYS use your tools to get current information
- When searching, include "{current_year}" in your search queries for latest results{tool_instructions}
- If tools don't return {current_year} data, explicitly mention this in your response"""

        return enhanced_goal
    
    def _enhance_backstory_for_current_data(self, original_backstory: str) -> str:
        """Enhance backstory to emphasize tool usage and avoiding outdated information."""
        from datetime import datetime
        
        current_year = datetime.now().year
        
        enhanced_backstory = f"""{original_backstory}

IMPORTANT: You are a tool-using agent who NEVER relies on pre-existing knowledge alone. You ALWAYS:
- Use available tools to gather the most current information
- Search specifically for {current_year} data when looking for "latest" or "recent" information  
- Clearly state when information is from previous years vs current year
- Prioritize fresh, real-time data over any cached knowledge
- Include the current date ({current_year}) in search queries to get the most recent results"""

        return enhanced_backstory
    
    def _create_enhanced_mock_tool(self, tool_name: str):
        """Create enhanced mock tools that provide better feedback about missing functionality."""
        from crewai.tools import tool
        
        tool_descriptions = {
            "web_search": {
                "name": "Web Search Tool",
                "desc": "Search the web for current information. Note: Real search requires API key configuration.",
                "func_name": "search_web"
            },
            "web_scraping": {
                "name": "Web Scraping Tool", 
                "desc": "Scrape and extract data from websites. Note: Real scraping requires API key configuration.",
                "func_name": "scrape_website"
            },
            "document_search": {
                "name": "Document Search Tool",
                "desc": "Search within documents (PDF, DOCX, etc). Note: Real search requires file access.",
                "func_name": "search_documents"
            },
            "github_search": {
                "name": "GitHub Search Tool",
                "desc": "Search GitHub repositories and code. Note: Real search requires API key configuration.",
                "func_name": "search_github"
            },
            "youtube_search": {
                "name": "YouTube Search Tool",
                "desc": "Search YouTube videos and channels. Note: Real search requires API key configuration.",
                "func_name": "search_youtube"
            },
            "vision": {
                "name": "Vision Tool",
                "desc": "Analyze images and generate images. Note: Real functionality requires API key configuration.",
                "func_name": "process_vision"
            },
            "database_search": {
                "name": "Database Search Tool",
                "desc": "Query PostgreSQL databases. Note: Real queries require database configuration.",
                "func_name": "search_database"
            },
            "browser_automation": {
                "name": "Browser Automation Tool",
                "desc": "Automate browser interactions. Note: Real automation requires API key configuration.",
                "func_name": "automate_browser"
            },
            "file_operations": {
                "name": "File Operations Tool",
                "desc": "Read and write files. Note: Limited to basic operations in mock mode.",
                "func_name": "handle_files"
            },
            "code_execution": {
                "name": "Code Execution Tool",
                "desc": "Execute Python code safely. Note: Mock mode provides simulated results.",
                "func_name": "execute_code"
            },
            "data_processing": {
                "name": "Data Processing Tool",
                "desc": "Process and analyze data. Note: Mock mode provides simulated analysis.",
                "func_name": "process_data"
            },
            "api_calls": {
                "name": "API Calls Tool",
                "desc": "Make HTTP API calls. Note: Mock mode provides simulated responses.",
                "func_name": "make_api_call"
            }
        }
        
        if tool_name not in tool_descriptions:
            return None
        
        tool_info = tool_descriptions[tool_name]
        
        # Create a properly formatted mock tool that explains limitations
        def create_mock_function(tool_name, tool_info):
            def mock_function(input_data: str = "") -> str:
                f"""Mock {tool_name} tool for testing. {tool_info['desc']}"""
                return f"""🔧 {tool_info['name']} (Mock Mode)
                
Input: {input_data}

⚠️ This is a mock tool providing simulated results. {tool_info['desc']}

For real functionality:
1. Configure required API keys in your environment
2. Install required dependencies 
3. Restart CrewMaster

Mock result: Simulated {tool_name} operation completed successfully with input: {input_data[:100]}{"..." if len(input_data) > 100 else ""}"""
            
            # Add docstring attribute
            mock_function.__doc__ = f"""Mock {tool_name} tool for testing. {tool_info['desc']}"""
            return mock_function
        
        # Use the @tool decorator to create a proper CrewAI tool
        mock_func = create_mock_function(tool_name, tool_info)
        mock_func.__name__ = tool_info['func_name']
        
        return tool(tool_info['name'])(mock_func)
    
    def _create_tasks(self, spec: CrewSpec, agents: List[Agent]) -> List[Task]:
        """Create CrewAI tasks for the crew with proper agent collaboration using context."""
        from datetime import datetime
        
        current_year = datetime.now().year
        current_date = datetime.now().strftime("%B %d, %Y")
        
        tasks = []
        
        if not agents:
            return tasks
        
        # Create main task for the first agent
        enhanced_task_description = f"""{spec.task}

MANDATORY EXECUTION REQUIREMENTS:
- Current date: {current_date}
- When searching for "latest", "recent", or "current" information, specifically look for {current_year} data
- You MUST use your available tools - do NOT rely on pre-existing knowledge
- For research tasks: Search with queries like "reinforcement learning {current_year}" or "latest papers {current_year}"
- Always verify information is from {current_year} when possible
- If you find only older information, clearly state the publication dates you found

INPUT PARSING REQUIREMENTS:
- ALWAYS check for USER INPUT in your task description
- Extract any file paths that start with / (like /Users/path/file.txt)
- Look for patterns like "file_path:", "read file", "analyze file"
- Extract any specific parameters or instructions from user input
- Use extracted file paths directly with your file reading tools

TOOLS AVAILABLE TO YOU:
{self._get_tool_descriptions_for_task(agents)}

COLLABORATION NOTE: You are the first agent in a collaborative team. Your output will be used by subsequent agents.
START BY USING YOUR TOOLS TO GATHER CURRENT INFORMATION BEFORE GENERATING ANY RESPONSE."""

        # Enhanced expected output to emphasize current data
        enhanced_expected_output = f"""{spec.expected_output}

IMPORTANT: The output must include:
- Publication dates for any papers or sources cited
- Clear indication when information is from {current_year} vs previous years
- Evidence that tools were used to gather current information
- If no {current_year} data is found, explanation of what was searched and what was available
- Clear, structured output for the next agent to build upon"""
        
        main_task = Task(
            description=enhanced_task_description,
            expected_output=enhanced_expected_output,
            agent=agents[0]
        )
        
        tasks.append(main_task)
        print(f"🔧 DEBUG: Created main task for {agents[0].role}")
        
        # Create collaborative tasks for additional agents with proper context linking
        if len(agents) > 1:
            print(f"🔧 DEBUG: Creating {len(agents)-1} additional collaborative tasks with context")
            
            for i, agent in enumerate(agents[1:], 1):
                # Create task description that emphasizes using previous agent's output
                agent_specific_task = self._get_agent_specific_task_description(agent.role, spec.task, current_date, current_year)
                
                collaborative_task = Task(
                    description=agent_specific_task,
                    expected_output=f"""As a {agent.role}, provide your specialized expertise to enhance the work done by previous agents. Your output should:
- Build directly upon the previous agent's findings
- Add your unique {agent.role} perspective and insights
- Provide complementary analysis or deliverables
- Create a comprehensive result when combined with previous work""",
                    agent=agent,
                    context=[tasks[i-1]]  # This is the key fix - reference previous task for context
                )
                
                tasks.append(collaborative_task)
                print(f"🔧 DEBUG: Created collaborative task for {agent.role} with context from {tasks[i-1].agent.role}")
        
        print(f"🔧 DEBUG: Total tasks created: {len(tasks)} with proper context linking")
        for i, task in enumerate(tasks):
            try:
                context_count = len(task.context) if hasattr(task, 'context') and task.context and hasattr(task.context, '__len__') else 0
                context_info = f" (uses context from {context_count} previous tasks)" if context_count > 0 else ""
                print(f"🔧 DEBUG: Task {i+1}: Agent = {task.agent.role}{context_info}")
            except (TypeError, AttributeError):
                print(f"🔧 DEBUG: Task {i+1}: Agent = {task.agent.role} (context info unavailable)")
        
        return tasks
    
    def _get_agent_specific_task_description(self, agent_role: str, original_task: str, current_date: str, current_year: int) -> str:
        """Generate agent-specific task descriptions based on their role."""
        base_requirements = f"""
CONTEXT COLLABORATION:
- You will receive the output from the previous agent automatically via context
- Use their findings as the foundation for your specialized work
- DO NOT repeat their work - build upon it with your {agent_role} expertise

MANDATORY EXECUTION REQUIREMENTS:
- Current date: {current_date}
- You MUST use your available tools for additional research and analysis
- Verify and expand upon previous findings where appropriate
- Add your unique {agent_role} perspective to the overall solution"""

        role_specific_tasks = {
            "researcher": f"""As a researcher, your task is to expand and verify the research conducted by the previous agent.

RESEARCH FOCUS:
- Validate findings from the previous agent's research
- Conduct additional searches to fill any gaps
- Look for more recent {current_year} studies or developments
- Cross-reference sources to ensure accuracy
- Find complementary research that supports or challenges initial findings

YOUR RESEARCH TASK: {original_task}
{base_requirements}""",

            "analyst": f"""As an analyst, your task is to analyze and synthesize the data/information provided by the previous agent.

ANALYSIS FOCUS:
- Perform deep analysis of the research and data collected
- Identify patterns, trends, and key insights
- Create summaries and actionable recommendations
- Quantify findings where possible
- Compare different approaches or solutions

YOUR ANALYSIS TASK: {original_task}
{base_requirements}""",

            "writer": f"""As a writer, your task is to create comprehensive documentation from the previous agent's work.

WRITING FOCUS:
- Organize findings into clear, structured documents
- Create executive summaries and detailed reports
- Ensure content is accessible to target audience
- Add proper citations and references
- Format content professionally

YOUR WRITING TASK: {original_task}
{base_requirements}""",

            "specialist": f"""As a specialist, your task is to provide expert-level insights on the previous agent's work.

SPECIALIST FOCUS:
- Apply domain expertise to evaluate findings
- Provide technical recommendations and best practices
- Identify potential issues or limitations
- Suggest implementation strategies
- Add professional insights from your field

YOUR SPECIALIST TASK: {original_task}
{base_requirements}""",

            "critic": f"""As a critic, your task is to critically evaluate and improve the previous agent's work.

CRITICAL FOCUS:
- Identify strengths and weaknesses in previous work
- Suggest improvements and alternative approaches
- Validate accuracy and completeness
- Provide constructive feedback
- Recommend next steps or additional research

YOUR CRITICAL EVALUATION TASK: {original_task}
{base_requirements}"""
        }
        
        return role_specific_tasks.get(agent_role.lower(), f"""As a {agent_role}, build upon the previous agent's work to complete: {original_task}
{base_requirements}""")
    
    def _get_tool_descriptions_for_task(self, agents: List[Agent]) -> str:
        """Generate tool descriptions for task instructions."""
        all_tools = set()
        for agent in agents:
            if agent.tools:
                for tool in agent.tools:
                    tool_name = getattr(tool, 'name', str(type(tool)))
                    all_tools.add(tool_name)
        
        tool_descriptions = {
            'Web Search Tool': 'Search the internet for current information and research papers',
            'Document Search Tool': 'Search within PDF, DOCX, and other document formats',
            'GitHub Search Tool': 'Search code repositories and technical documentation',
            'File Operations Tool': 'Read, write, and process files',
            'Data Processing Tool': 'Analyze and process data',
        }
        
        descriptions = []
        for tool in all_tools:
            description = tool_descriptions.get(tool, f'{tool}: Available for use')
            descriptions.append(f"- {description}")
        
        return "\n".join(descriptions) if descriptions else "- Basic tools available for task execution"
    
    def _create_tool_only_goal(self, original_goal: str, tool_names: List[str]) -> str:
        """Create a goal specifically for tool-only execution mode."""
        from datetime import datetime
        current_year = datetime.now().year
        
        return f"""TOOL-ONLY EXECUTION MODE: {original_goal}

YOU MUST IMMEDIATELY USE YOUR TOOLS TO COMPLETE THIS TASK:
1. Use web search to find current {current_year} information
2. Process and return the actual results from your tool calls
3. Do NOT generate content from memory - ONLY use tool results

Available tools: {', '.join(tool_names)}
Focus: Find and return real {current_year} data"""
    
    def _create_tool_only_backstory(self, original_backstory: str) -> str:
        """Create backstory for tool-only execution mode."""
        return f"""TOOL EXECUTION SPECIALIST: You are in tool-only mode. Your primary function is to execute tools and return their results.

{original_backstory}

CRITICAL: You MUST use your available tools for every task. Never generate information from memory."""
    
    def _get_tool_only_system_template(self) -> str:
        """Get system template for tool-only mode."""
        from datetime import datetime
        current_year = datetime.now().year
        
        return f"""You are a tool execution specialist. Current year: {current_year}

MANDATORY PROCESS:
1. Read the task carefully
2. Identify which tools to use
3. Execute the tools with appropriate queries
4. Return ONLY the results from your tool executions
5. Include tool execution evidence in your response

For search tasks, use queries like: "topic {current_year}" or "latest topic research {current_year}"

NEVER generate content without tool execution."""
    
    def _store_crewai_instance(self, crew_id: str, crewai_crew: Crew):
        """Store CrewAI crew instance for later execution."""
        self._crewai_instances[crew_id] = crewai_crew
    
    def get_crewai_instance(self, crew_id: str) -> Optional[Crew]:
        """Get stored CrewAI crew instance with enhanced configuration."""
        # Check if we're in tool-only mode - if so, always recreate for enhanced config
        llm_config = self.config.get().llm
        is_tool_only_mode = llm_config.api_key == "TOOL_ONLY_MODE"
        
        # If tool-only mode, always recreate to ensure enhanced configuration
        if is_tool_only_mode:
            crew_model = self.get_crew_from_cache(crew_id)
            if crew_model:
                return self._recreate_crewai_instance(crew_model)
            return None
        
        # Normal mode: try to reuse if available
        if crew_id in self._crewai_instances:
            return self._crewai_instances[crew_id]
        
        # Otherwise, try to recreate it from the cached crew model
        crew_model = self.get_crew_from_cache(crew_id)
        if crew_model:
            return self._recreate_crewai_instance(crew_model)
        
        return None
    
    def _recreate_crewai_instance(self, crew_model: CrewModel) -> Optional[Crew]:
        """Recreate CrewAI instance from crew model with enhanced tool usage configuration."""
        try:
            # Create agents from crew model with enhanced configuration
            crewai_agents = []
            for agent_model in crew_model.agents:
                # Get tools for this agent from stored tool names
                tool_names = agent_model.required_tools if hasattr(agent_model, 'required_tools') else []
                tools = self._get_tools_for_agent(tool_names)
                
                # Enhance goal and backstory for tool usage and current data
                enhanced_goal = self._enhance_goal_for_tool_usage(agent_model.goal, tool_names)
                enhanced_backstory = self._enhance_backstory_for_current_data(agent_model.backstory)
                
                # Check if we're in tool-only mode
                llm_config = self.config.get().llm
                is_tool_only_mode = llm_config.api_key == "TOOL_ONLY_MODE"
                
                if is_tool_only_mode:
                    # Create enhanced agent for tool-only mode
                    enhanced_goal = self._create_tool_only_goal(agent_model.goal, tool_names)
                    enhanced_backstory = self._create_tool_only_backstory(agent_model.backstory)
                    
                    agent = Agent(
                        role=agent_model.role,
                        goal=enhanced_goal,
                        backstory=enhanced_backstory,
                        tools=tools,
                        verbose=True,  # Force verbose to show tool usage
                        allow_delegation=False,  # Disable delegation in tool-only mode
                        max_iter=1,  # Single iteration to force tool usage
                        memory=False,
                        system_template=self._get_tool_only_system_template()
                    )
                else:
                    agent = Agent(
                        role=agent_model.role,
                        goal=enhanced_goal,
                        backstory=enhanced_backstory,
                        tools=tools,
                        verbose=True,
                        allow_delegation=False,
                        memory=False
                    )
                
                crewai_agents.append(agent)
            
            # Create enhanced tasks with proper collaboration
            from crewai import Task
            tasks = []
            
            if crewai_agents:
                # Create main task for first agent
                enhanced_task_description = f"""{crew_model.task}

MANDATORY EXECUTION REQUIREMENTS:
- Current date: {datetime.now().strftime("%B %d, %Y")}
- When searching for "latest", "recent", or "current" information, specifically look for {datetime.now().year} data
- You MUST use your available tools - do NOT rely on pre-existing knowledge
- For research tasks: Search with queries like "reinforcement learning {datetime.now().year}" or "latest papers {datetime.now().year}"
- Always verify information is from {datetime.now().year} when possible
- If you find only older information, clearly state the publication dates you found

COLLABORATION NOTE: You are the first agent in a collaborative team. Your output will be used by subsequent agents.
START BY USING YOUR TOOLS TO GATHER CURRENT INFORMATION BEFORE GENERATING ANY RESPONSE."""
            
                main_task = Task(
                    description=enhanced_task_description,
                    expected_output=f"Current {datetime.now().year} results and tool-verified information for: {crew_model.task}",
                    agent=crewai_agents[0]
                )
                tasks.append(main_task)
                
                # Create collaborative tasks for additional agents
                for i, agent in enumerate(crewai_agents[1:], 1):
                    agent_specific_task = self._get_agent_specific_task_description(
                        agent.role, crew_model.task, 
                        datetime.now().strftime("%B %d, %Y"), 
                        datetime.now().year
                    )
                    
                    collaborative_task = Task(
                        description=agent_specific_task,
                        expected_output=f"Specialized {agent.role} analysis building upon previous work",
                        agent=agent,
                        context=[tasks[i-1]]  # Link to previous task
                    )
                    tasks.append(collaborative_task)
            else:
                # Fallback for no agents
                tasks = []
            
            # Create crew
            crewai_crew = Crew(
                agents=crewai_agents,
                tasks=tasks,  # Use the properly linked tasks
                process=Process.sequential,
                verbose=True,
                memory=False
            )
            
            # Store in memory for future use
            self._crewai_instances[crew_model.id] = crewai_crew
            return crewai_crew
            
        except Exception as e:
            print(f"Error recreating CrewAI instance: {e}")
            return None
    
    def _load_cache(self):
        """Load cache from file."""
        try:
            import pickle
            import os
            cache_file = "/tmp/crewmaster_cache.pkl"
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self._crews_cache = data.get('crews', {})
                    # Note: CrewAI instances are not pickleable, so we skip them
        except Exception:
            pass  # Ignore cache load errors
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            import pickle
            cache_file = "/tmp/crewmaster_cache.pkl"
            data = {'crews': self._crews_cache}
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            pass  # Ignore cache save errors
    
    def update_crew_config(self, crew_id: str, config_updates: Dict[str, Any]) -> bool:
        """Update crew configuration."""
        crew_model = self.crew_repo.get_crew(crew_id)
        if not crew_model:
            return False
        
        # Update database model
        self.crew_repo.update_crew(crew_id, config_updates)
        
        # If CrewAI instance exists, update it as well
        crewai_crew = self.get_crewai_instance(crew_id)
        if crewai_crew:
            # Update relevant properties
            if 'verbose' in config_updates:
                crewai_crew.verbose = config_updates['verbose']
            if 'memory_enabled' in config_updates:
                crewai_crew.memory = config_updates['memory_enabled']
        
        return True
    
    def clone_crew(self, crew_id: str, new_name: Optional[str] = None) -> Optional[CrewModel]:
        """Clone an existing crew with a new name."""
        original_crew = self.crew_repo.get_crew(crew_id)
        if not original_crew:
            return None
        
        # Create new crew data
        crew_data = {
            "name": new_name or f"{original_crew.name}_clone",
            "task": original_crew.task,
            "description": original_crew.description,
            "process_type": original_crew.process_type,
            "verbose": original_crew.verbose,
            "memory_enabled": original_crew.memory_enabled,
            "expected_output": original_crew.expected_output,
            "task_config": original_crew.task_config
        }
        
        agent_ids = [agent.id for agent in original_crew.agents]
        
        cloned_crew = self.crew_repo.create_crew(crew_data, agent_ids)
        
        # Clone the CrewAI instance if it exists
        original_crewai = self.get_crewai_instance(crew_id)
        if original_crewai:
            # Create a new CrewAI crew with the same configuration
            new_agents = []
            for agent in original_crew.agents:
                new_agents.append(self._create_crewai_agent_from_model(agent))
            
            new_tasks = self._create_tasks_from_crew_model(original_crew, new_agents)
            
            process = self.process_mapping.get(original_crew.process_type, Process.sequential)
            new_crewai_crew = Crew(
                agents=new_agents,
                tasks=new_tasks,
                process=process,
                verbose=original_crew.verbose,
                memory=original_crew.memory_enabled
            )
            
            self._store_crewai_instance(cloned_crew.id, new_crewai_crew)
        
        return cloned_crew
    
    def _create_tasks_from_crew_model(self, crew_model: CrewModel, agents: List[Agent]) -> List[Task]:
        """Create CrewAI tasks from a crew database model."""
        # This is a simplified version - in practice, you'd want to store
        # task specifications in the database as well
        
        main_task = Task(
            description=crew_model.task,
            expected_output=crew_model.expected_output or "Complete task successfully",
            agent=agents[0] if agents else None
        )
        
        return [main_task]
    
    def get_crew_performance_metrics(self, crew_id: str) -> Dict[str, Any]:
        """Get performance metrics for a crew."""
        crew_model = self.crew_repo.get_crew(crew_id)
        if not crew_model:
            return {}
        
        return {
            "crew_id": crew_id,
            "execution_count": crew_model.execution_count,
            "success_rate": crew_model.success_rate,
            "avg_execution_time": crew_model.avg_execution_time,
            "last_executed": crew_model.last_executed,
            "agent_count": len(crew_model.agents),
            "agent_performance": [
                {
                    "agent_id": agent.id,
                    "name": agent.name,
                    "role": agent.role,
                    "usage_count": agent.usage_count,
                    "success_rate": agent.success_rate,
                    "avg_execution_time": agent.avg_execution_time
                }
                for agent in crew_model.agents
            ]
        }