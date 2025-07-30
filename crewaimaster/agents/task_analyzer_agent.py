"""
TaskAnalyzerAgent - AI-powered agent for intelligent task analysis.

This agent uses CrewAI to analyze natural language task descriptions
and generate detailed crew specifications with intelligent reasoning.
"""

from typing import Dict, List, Any, Optional
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from ..core.task_analyzer import TaskComplexity, AgentRole, TaskRequirement, AgentSpec, CrewSpec


class TaskAnalysisResult(BaseModel):
    """Structured result from task analysis."""
    complexity: str = Field(description="Task complexity: simple, moderate, or complex")
    required_roles: List[str] = Field(description="List of required agent roles")
    tools_needed: List[str] = Field(description="List of tools needed for execution")
    output_format: str = Field(description="Expected output format")
    estimated_time: int = Field(description="Estimated time in minutes")
    process_type: str = Field(description="Process type: sequential, hierarchical")
    agent_specifications: List[Dict[str, Any]] = Field(description="Detailed agent specs")



class TaskAnalyzerAgent:
    """AI-powered task analyzer using CrewAI."""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the TaskAnalyzerAgent."""
        self.llm_config = llm_config or {}
        
        # Create the analysis agent
        agent_kwargs = {
            "role": "Task Analysis Specialist",
            "goal": "Analyze natural language task descriptions and determine the optimal crew structure, agent roles, and tools needed for successful execution",
            "backstory": """You are an expert AI task analyzer with deep understanding of multi-agent systems, 
            workflow orchestration, and task decomposition. You excel at breaking down complex requests into 
            actionable components and determining the most efficient agent configuration for any given task.
            
            Your expertise includes:
            - Understanding task complexity and breaking down requirements
            - Identifying the optimal agent roles and responsibilities
            - Selecting appropriate tools and resources
            - Estimating execution time and process flow
            - Ensuring efficient coordination between agents
            """,
            "verbose": True,
            "allow_delegation": False,
            "tools": [],
            "max_iter": 3
        }
        
        # Add LLM configuration if provided
        if self.llm_config:
            agent_kwargs["llm"] = self._create_llm_instance()
            
        self.analyzer_agent = Agent(**agent_kwargs)
        
        # Create the specification generator agent
        spec_kwargs = {
            "role": "Agent Specification Designer",
            "goal":"Create detailed agent specifications based on task analysis including goals, backstories, and tool assignments",
            "backstory":"""You are a specialist in designing AI agents with the perfect balance of capabilities, 
            personality, and tool access. You understand how to craft compelling agent personas that work 
            effectively in team environments while maintaining their individual expertise areas.
            
            Your skills include:
            - Crafting engaging agent backstories and personalities
            - Defining clear, actionable goals for each agent
            - Selecting the optimal tool set for each role
            - Ensuring proper agent coordination and delegation
            - Balancing specialization with collaboration
            """,
            "verbose": True,
            "allow_delegation": False,
            "max_iter": 3
        }

        # Add LLM configuration if provided
        if self.llm_config:
            spec_kwargs["llm"] = self._create_llm_instance()
            
        self.spec_agent = Agent(**spec_kwargs)
    
    def _create_llm_instance(self):
        """Create LLM instance using CrewAI's LLM class."""
        try:
            from crewai import LLM
            provider = self.llm_config.get("provider", "openai")
            
            if provider == "custom":
                # For custom providers, specify all parameters
                return LLM(
                    model=self.llm_config.get("model", "gpt-4"),
                    api_key=self.llm_config.get("api_key"),
                    base_url=self.llm_config.get("base_url"),
                    temperature=self.llm_config.get("temperature", 0.7),
                    max_tokens=self.llm_config.get("max_tokens", 2000)
                )
            else:
                # For standard providers, pass API key explicitly
                return LLM(
                    model=self.llm_config.get("model", "gpt-4"),
                    api_key=self.llm_config.get("api_key"),
                    base_url=self.llm_config.get("base_url"),
                    temperature=self.llm_config.get("temperature", 0.7),
                    max_tokens=self.llm_config.get("max_tokens", 2000)
                )
        except Exception as e:
            print(f"Warning: Could not create LLM instance: {e}")
            return None
    
    def _normalize_task_description(self, task_description: str) -> str:
        """
        Normalize meta-task descriptions to direct task descriptions.
        
        Converts "Build a crew that does X" to "Do X"
        """
        import re
        
        # Patterns that indicate meta-tasks (explaining how to build rather than doing)
        meta_patterns = [
            # Direct patterns for crew/team/system/agent/ai/bot
            (r'^build\s+(?:a\s+)?(?:crew|team|system|agent|ai|bot)\s+that\s+(.+)$', r'\1'),
            (r'^create\s+(?:a\s+)?(?:crew|team|system|agent|ai|bot)\s+that\s+(.+)$', r'\1'),
            (r'^develop\s+(?:a\s+)?(?:crew|team|system|agent|ai|bot)\s+that\s+(.+)$', r'\1'),
            (r'^design\s+(?:a\s+)?(?:crew|team|system|agent|ai|bot)\s+that\s+(.+)$', r'\1'),
            (r'^make\s+(?:a\s+)?(?:crew|team|system|agent|ai|bot)\s+that\s+(.+)$', r'\1'),
            
            # Handle "Build an [anything] assistant/tool/helper that does Y" pattern
            (r'^build\s+(?:an?\s+)?(.+?)\s+(?:assistant|tool|helper|service|application|app|software|solution)\s+that\s+(.+)$', r'\2'),
            (r'^create\s+(?:an?\s+)?(.+?)\s+(?:assistant|tool|helper|service|application|app|software|solution)\s+that\s+(.+)$', r'\2'),
            (r'^develop\s+(?:an?\s+)?(.+?)\s+(?:assistant|tool|helper|service|application|app|software|solution)\s+that\s+(.+)$', r'\2'),
            (r'^design\s+(?:an?\s+)?(.+?)\s+(?:assistant|tool|helper|service|application|app|software|solution)\s+that\s+(.+)$', r'\2'),
            (r'^make\s+(?:an?\s+)?(.+?)\s+(?:assistant|tool|helper|service|application|app|software|solution)\s+that\s+(.+)$', r'\2'),
            
            # Handle "Build an AI/artificial intelligence X that does Y" pattern  
            (r'^build\s+(?:an?\s+)?(?:ai|artificial intelligence)\s+(.+?)\s+that\s+(.+)$', r'\2'),
            (r'^create\s+(?:an?\s+)?(?:ai|artificial intelligence)\s+(.+?)\s+that\s+(.+)$', r'\2'),
            (r'^develop\s+(?:an?\s+)?(?:ai|artificial intelligence)\s+(.+?)\s+that\s+(.+)$', r'\2'),
            (r'^design\s+(?:an?\s+)?(?:ai|artificial intelligence)\s+(.+?)\s+that\s+(.+)$', r'\2'),
            (r'^make\s+(?:an?\s+)?(?:ai|artificial intelligence)\s+(.+?)\s+that\s+(.+)$', r'\2'),
            
            # Handle reverse pattern: "Build X agent/crew/system that does Y"
            (r'^(?:build|create|develop|design|make)\s+(?:an?\s+)?(.+?)\s+(?:crew|team|system|agent|ai|bot)(?:\s+that\s+(.+))?$', r'\1 \2'),
        ]
        
        # Normalize to lowercase for pattern matching
        normalized_desc = task_description.lower().strip()
        
        # Apply normalization patterns
        for pattern, replacement in meta_patterns:
            match = re.match(pattern, normalized_desc, re.IGNORECASE)
            if match:
                # Extract the actual task from the meta description
                direct_task = match.expand(replacement).strip()
                
                # Clean up the extracted task
                direct_task = re.sub(r'^(can|will|should|that)\s+', '', direct_task)
                direct_task = re.sub(r'\s+', ' ', direct_task)
                
                # Ensure it starts with an action verb
                if not re.match(r'^(monitor|analyze|research|find|track|watch|collect|process|generate|create|write|build|develop)', direct_task, re.IGNORECASE):
                    # If no action verb, try to infer from context
                    if any(word in direct_task for word in ['price', 'monitor', 'track', 'watch']):
                        direct_task = f"monitor {direct_task}"
                    elif any(word in direct_task for word in ['research', 'paper', 'study', 'find']):
                        direct_task = f"research {direct_task}"
                    elif any(word in direct_task for word in ['analyze', 'analysis', 'data']):
                        direct_task = f"analyze {direct_task}"
                    elif any(word in direct_task for word in ['report', 'document', 'write']):
                        direct_task = f"create {direct_task}"
                    else:
                        direct_task = f"execute {direct_task}"
                
                # Capitalize first letter
                direct_task = direct_task[0].upper() + direct_task[1:] if direct_task else direct_task
                
                return direct_task
        
        # If no meta-pattern matched, return original task
        return task_description
    
    def analyze_task(self, task_description: str) -> CrewSpec:
        """
        Analyze a task description using optimized AI and return a detailed crew specification.
        
        This method is optimized for speed and efficiency using a single AI agent instead of a crew.
        
        Args:
            task_description: Natural language description of the task
            
        Returns:
            CrewSpec: Detailed specification for creating the crew
        """
        # First, normalize the task description to avoid meta-task confusion
        normalized_task = self._normalize_task_description(task_description)
        
        # Create a single, comprehensive analysis task for efficiency
        analysis_task = Task(
            description=f"""
            Analyze this task and provide a complete crew specification for EXECUTING the task, not explaining how to do it.
            
            TASK TO EXECUTE: {normalized_task}
            
            CRITICAL: Design agents that will ACTIVELY EXECUTE this task, not create plans or strategies.
            
            Provide a comprehensive analysis with the following structure:
            
            {{
                "taskComplexity": "simple|moderate|complex",
                "estimatedTime": <minutes>,
                "processType": "sequential|hierarchical",
                "expectedOutput": "<actual_execution_results_description>",
                "agentSpecifications": [
                    {{
                        "role": "<role_name>",
                        "agentName": "<specific_name>",
                        "goal": "<direct_execution_goal_that_delivers_results>",
                        "backstory": "<expertise_focused_on_doing_the_work>",
                        "tools": ["<tool1>", "<tool2>", "<tool3>"],
                        "memoryType": "short_term|long_term",
                        "maxIterations": <number>,
                        "allowDelegation": true|false
                    }}
                ],
                "crewNames": ["<name1>", "<name2>", "<name3>"]
            }}
            
            EXECUTION-FOCUSED Guidelines:
            - Design agents that DO the work, not explain how to do it
            - For monitoring tasks: agents should monitor and return actual data
            - For research tasks: agents should find and return actual results
            - For analysis tasks: agents should analyze and return insights
            - For development tasks: agents should build and return working solutions
            - Goals should be action-oriented: "Monitor prices and return data" not "Create a strategy for monitoring"
            - Expected output should be actual results: "Price data collected" not "Plan for price monitoring"
            - Tools should be selected from actual CrewAI tools: SerperDevTool, FileReadTool, ScrapeWebsiteTool, GithubSearchTool, YoutubeVideoSearchTool, YoutubeChannelSearchTool, CodeInterpreterTool, PDFSearchTool, DOCXSearchTool, CSVSearchTool, JSONSearchTool, XMLSearchTool, TXTSearchTool, MDXSearchTool, DirectoryReadTool, DirectorySearchTool, PGSearchTool, BrowserbaseLoadTool, FirecrawlScrapeWebsiteTool, WebsiteSearchTool, EXASearchTool
            - Agent names should reflect doers, not planners
            - Backstories should emphasize hands-on execution experience
            """,
            agent=self.analyzer_agent,
            expected_output="Complete JSON specification for crew creation with all agent details and configuration"
        )
        
        # Execute the analysis with a single efficient task
        try:
            result = analysis_task.execute_sync()
            # Parse the result and convert to CrewSpec
            return self._parse_analysis_result(str(result), task_description, normalized_task)
        except Exception as e:
            # Fallback to basic crew execution if direct execution fails
            analysis_crew = Crew(
                agents=[self.analyzer_agent],
                tasks=[analysis_task],
                process="sequential",
                verbose=False  # Disable verbose for speed
            )
            result = analysis_crew.kickoff(inputs={"task": normalized_task})
            return self._parse_analysis_result(str(result), task_description, normalized_task)
    
    def _parse_analysis_result(self, analysis_result: str, original_task: str, normalized_task: str = None) -> CrewSpec:
        """
        Parse the AI analysis result and convert it to a CrewSpec object.
        
        This method intelligently parses the AI output and converts it 
        to the structured CrewSpec format.
        """
        # Use normalized task if provided, otherwise use original
        actual_task = normalized_task if normalized_task else original_task
        # Try to extract structured information from AI analysis
        complexity = self._determine_complexity_from_analysis(analysis_result)
        agent_roles = self._extract_roles_from_analysis(analysis_result)
        
        # Parse expected output from analysis
        expected_output = self._extract_expected_output(analysis_result, actual_task)
        
        # Parse estimated time from analysis  
        estimated_time = self._estimate_time_from_analysis(analysis_result)
        
        # Generate agent specifications with AI-informed details
        agents = []
        ai_agents = self._extract_agent_specs_from_analysis(analysis_result)
        
        for i, role in enumerate(agent_roles):
            # Use AI-generated specs if available, otherwise create intelligent defaults
            if i < len(ai_agents) and ai_agents[i]:
                ai_agent = ai_agents[i]
                specific_role = self._extract_agent_role(analysis_result, role, original_task)
                agent_spec = AgentSpec(
                    role=specific_role,
                    name=ai_agent.get('name', f"{specific_role}_agent_{i+1}"),
                    goal=ai_agent.get('goal', self._extract_agent_goal(analysis_result, role, original_task)),
                    backstory=ai_agent.get('backstory', self._extract_agent_backstory(analysis_result, role)),
                    required_tools=ai_agent.get('tools', self._extract_agent_tools(analysis_result, role)),
                    memory_type="short_term",
                    max_iter=5,
                    allow_delegation=len(agent_roles) > 1
                )
            else:
                # Fallback to extracted information
                specific_role = self._extract_agent_role(analysis_result, role, actual_task)
                agent_goal = self._extract_agent_goal(analysis_result, role, actual_task)
                agent_backstory = self._extract_agent_backstory(analysis_result, role)
                agent_tools = self._extract_agent_tools(analysis_result, role)
                
                agent_spec = AgentSpec(
                    role=specific_role,
                    name=f"{specific_role}_agent_{i+1}",
                    goal=agent_goal,
                    backstory=agent_backstory,
                    required_tools=agent_tools,
                    memory_type="short_term",
                    max_iter=5,
                    allow_delegation=len(agent_roles) > 1
                )
            agents.append(agent_spec)
        
        return CrewSpec(
            name=self._generate_crew_name(original_task, analysis_result),
            task=actual_task,  # Use normalized task for actual execution
            description=f"AI-analyzed crew for: {actual_task}",
            agents=agents,
            expected_output=expected_output,
            complexity=complexity,
            estimated_time=estimated_time,
            process_type="sequential"
        )
    
    def _determine_complexity_from_analysis(self, analysis: str) -> TaskComplexity:
        """Determine complexity from AI analysis."""
        analysis_lower = analysis.lower()
        if "complex" in analysis_lower or "comprehensive" in analysis_lower:
            return TaskComplexity.COMPLEX
        elif "moderate" in analysis_lower or "medium" in analysis_lower:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def _parse_json_from_text(self, text: str) -> dict:
        """Robust JSON parsing from AI-generated text."""
        import re
        import json
        
        # Try multiple extraction methods
        json_data = None
        
        # Method 1: Look for JSON wrapped in ```json blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                json_data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Method 2: Look for any JSON block with agentSpecifications
        if not json_data:
            # Use a more flexible pattern that can handle nested braces
            brace_count = 0
            start_pos = text.find('{')
            if start_pos != -1:
                for i, char in enumerate(text[start_pos:], start_pos):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            potential_json = text[start_pos:i+1]
                            if 'agentSpecifications' in potential_json:
                                try:
                                    json_data = json.loads(potential_json)
                                    break
                                except json.JSONDecodeError:
                                    # Try to fix common issues
                                    fixed_json = self._fix_common_json_issues(potential_json)
                                    try:
                                        json_data = json.loads(fixed_json)
                                        break
                                    except json.JSONDecodeError:
                                        continue
        
        # Method 3: Try parsing the entire text as JSON
        if not json_data:
            try:
                text_stripped = text.strip()
                if text_stripped.startswith('{') and text_stripped.endswith('}'):
                    json_data = json.loads(text_stripped)
            except json.JSONDecodeError:
                pass
        
        return json_data
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues in AI responses."""
        import re
        
        # Remove trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix unquoted property names (but be careful not to break quoted strings)
        # This is a simple fix - a full solution would need proper parsing
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # Fix single quotes to double quotes (but be careful with contractions)
        json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
        
        return json_str

    def _extract_roles_from_analysis(self, analysis: str) -> List[str]:
        """Extract agent roles from AI analysis."""
        import re
        import json
        
        # Try to parse JSON from AI analysis first
        try:
            json_data = self._parse_json_from_text(analysis)
            
            if json_data and 'agentSpecifications' in json_data:
                roles = []
                print(f"🔧 DEBUG: Found {len(json_data['agentSpecifications'])} agent specifications in JSON")
                for i, agent in enumerate(json_data['agentSpecifications']):
                    if 'role' in agent:
                        # Extract role and clean it up
                        role = agent['role'].lower()
                        # Convert "Data Collection Agent" -> "data_collector"
                        clean_role = re.sub(r'\s+agent$', '', role)  # Remove " agent" suffix
                        clean_role = re.sub(r'[^a-z0-9]+', '_', clean_role)  # Replace spaces/special chars
                        clean_role = clean_role.strip('_')  # Remove leading/trailing underscores
                        print(f"🔧 DEBUG: Agent {i+1}: {agent.get('role', 'Unknown')} -> {clean_role}")
                        if clean_role not in roles:
                            roles.append(clean_role)
                if roles:
                    print(f"🔧 DEBUG: Extracted {len(roles)} roles from JSON: {roles}")
                    return roles[:5]  # Limit to 5 agents max
        except (json.JSONDecodeError, KeyError) as e:
            print(f"🔧 DEBUG: JSON parsing failed: {e}")
            pass
        
        # Fallback to pattern-based extraction
        analysis_lower = analysis.lower()
        found_roles = []
        
        # Look for explicit role mentions
        role_patterns = {
            r'\bresearcher?\b': 'researcher',
            r'\bwriter?\b': 'writer', 
            r'\banalyst?\b': 'analyst',
            r'\bdeveloper?\b': 'developer',
            r'\breviewer?\b': 'reviewer',
            r'\bcoordinator?\b': 'coordinator',
            r'\bspecialist?\b': 'specialist',
            r'\bmanager?\b': 'manager',
            r'\bdesigner?\b': 'designer',
            r'\bcollector?\b': 'collector',
            r'\bstorage?\b': 'storage_agent',
            r'\balert?\b': 'alert_agent'
        }
        
        for pattern, role in role_patterns.items():
            if re.search(pattern, analysis_lower) and role not in found_roles:
                found_roles.append(role)
        
        # If no specific roles found, use intelligent defaults based on task indicators
        if not found_roles:
            if any(word in analysis_lower for word in ['research', 'analyze', 'investigate']):
                found_roles.append('researcher')
            if any(word in analysis_lower for word in ['write', 'create', 'document']):
                found_roles.append('writer')
            if any(word in analysis_lower for word in ['analyze', 'data', 'metrics']):
                found_roles.append('analyst')
        
        # Ensure we have at least one role
        if not found_roles:
            found_roles = ['researcher', 'analyst', 'writer']
            
        return found_roles[:5]  # Limit to 5 agents max
    
    def _get_tools_for_role(self, role: str) -> List[str]:
        """Get appropriate tools for an agent role."""
        tool_mapping = {
            "researcher": ["web_search", "document_search", "file_operations"],
            "analyst": ["data_processing", "code_execution", "web_search"],
            "writer": ["file_operations", "web_search"],
            "developer": ["code_execution", "api_calls", "file_operations"],
            "reviewer": ["file_operations", "web_search"]
        }
        return tool_mapping.get(role, ["web_search", "file_operations"])
    
    def _extract_ai_crew_names(self, analysis: str) -> List[str]:
        """Extract AI-suggested crew names from analysis."""
        import re
        
        # Look for crew name suggestions in the analysis
        crew_name_patterns = [
            r'crew names?[:\s]*([^.]*)',
            r'suggested names?[:\s]*([^.]*)',
            r'name suggestions?[:\s]*([^.]*)',
            r'recommended names?[:\s]*([^.]*)'
        ]
        
        suggested_names = []
        
        for pattern in crew_name_patterns:
            matches = re.finditer(pattern, analysis.lower())
            for match in matches:
                name_text = match.group(1)
                # Extract individual names from the text
                # Look for quoted names or capitalized words
                quoted_names = re.findall(r'"([^"]+)"', name_text)
                capitalized_names = re.findall(r'\b([A-Z][a-zA-Z]*(?:[A-Z][a-zA-Z]*)*)\b', name_text)
                
                suggested_names.extend(quoted_names)
                suggested_names.extend(capitalized_names)
        
        # Clean and validate names
        clean_names = []
        for name in suggested_names:
            # Clean the name
            clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', name).strip()
            # Convert to snake_case
            snake_case = re.sub(r'\s+', '_', clean_name.lower())
            if len(snake_case) > 2 and snake_case not in clean_names:
                clean_names.append(snake_case)
        
        return clean_names[:3]  # Return top 3 suggestions

    def _generate_crew_name(self, task_description: str, analysis: str = "") -> str:
        """Generate a meaningful crew name using AI suggestions first, fallback to algorithm."""
        # First, try to extract AI-suggested names
        if analysis:
            ai_names = self._extract_ai_crew_names(analysis)
            if ai_names:
                # Use the first AI suggestion
                return ai_names[0]
        
        # Fallback to algorithm-based naming
        import re
        
        # Remove common task action words that don't add meaning
        skip_words = {
            'create', 'build', 'make', 'develop', 'design', 'write', 'generate', 
            'find', 'search', 'analyze', 'help', 'me', 'us', 'a', 'an', 'the', 
            'for', 'about', 'on', 'with', 'that', 'which', 'this', 'to', 'and',
            'crew', 'team', 'group', 'agent', 'agents'  # Added these to avoid duplication
        }
        
        # Clean and split the task description
        words = re.sub(r'[^\w\s]', ' ', task_description.lower()).split()
        
        # Filter out skip words and keep meaningful words
        meaningful_words = [word for word in words if word not in skip_words and len(word) > 2]
        
        # Take up to 3 most meaningful words
        if meaningful_words:
            crew_words = meaningful_words[:3]
        else:
            # Fallback: take first 2 non-skip words
            fallback_words = [word for word in words[:5] if word not in skip_words]
            crew_words = fallback_words[:2] if fallback_words else ['task']
        
        # Create a clean crew name
        crew_name = "_".join(crew_words)
        
        # Ensure name is not too long (max 30 chars)
        if len(crew_name) > 30:
            crew_name = crew_name[:30]
        
        return crew_name
    
    def _estimate_time_from_analysis(self, analysis: str) -> int:
        """Estimate execution time from analysis."""
        import re
        
        # Try to find explicit time mentions in minutes
        time_patterns = [
            r'(\d+)\s*minutes?',
            r'(\d+)\s*mins?',
            r'(\d+)\s*hours?\s*(\d+)?\s*minutes?',
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, analysis.lower())
            if match:
                if 'hour' in pattern:
                    hours = int(match.group(1))
                    minutes = int(match.group(2)) if match.group(2) else 0
                    return hours * 60 + minutes
                else:
                    return int(match.group(1))
        
        # Fallback to complexity-based estimation
        if "complex" in analysis.lower():
            return 30
        elif "moderate" in analysis.lower():
            return 15
        else:
            return 5
    
    def _extract_expected_output(self, analysis: str, task: str) -> str:
        """Extract expected output from AI analysis with execution focus."""
        # Look for output-related keywords in analysis
        analysis_lower = analysis.lower()
        task_lower = task.lower()
        
        # For monitoring/tracking tasks - expect actual data/results
        if any(word in task_lower for word in ['monitor', 'track', 'price', 'watch']):
            return f"Real-time monitoring results and data collected from executing: {task}"
        
        # For research tasks - expect findings/results
        if any(word in task_lower for word in ['research', 'find', 'paper', 'study']):
            return f"Research findings and collected results from executing: {task}"
        
        # For analysis tasks - expect insights/conclusions
        if any(word in task_lower for word in ['analysis', 'analyze', 'data']):
            return f"Analysis results and actionable insights from executing: {task}"
        
        # For development tasks - expect working solutions
        if any(word in task_lower for word in ['develop', 'build', 'create', 'implement']):
            return f"Working solution and implementation results for: {task}"
        
        # Fallback based on analysis content
        if "report" in analysis_lower:
            return f"Comprehensive report with execution results for: {task}"
        elif "summary" in analysis_lower:
            return f"Executive summary of execution results for: {task}"
        elif "document" in analysis_lower:
            return f"Documentation of execution results for: {task}"
        else:
            return f"Complete execution results and deliverables for: {task}"
    
    def _extract_agent_goal(self, analysis: str, role: str, task: str) -> str:
        """Extract agent-specific goal from analysis with action-oriented execution focus."""
        
        # Make task more action-oriented by converting to active execution
        action_task = self._make_task_actionable(task)
        
        role_goals = {
            "researcher": f"Research and gather live data while actively executing: {action_task}",
            "writer": f"Create and deliver content while executing: {action_task}",
            "analyst": f"Analyze data and provide real-time insights while executing: {action_task}",
            "developer": f"Develop and deploy working solutions that actively execute: {action_task}",
            "reviewer": f"Review results and validate execution of: {action_task}",
            "coordinator": f"Coordinate and manage active execution of: {action_task}",
            "specialist": f"Apply specialized expertise to directly execute: {action_task}",
            "manager": f"Manage and oversee active execution of: {action_task}",
            "designer": f"Design and implement solutions that execute: {action_task}"
        }
        return role_goals.get(role, f"Actively execute {role} work for: {action_task}")
    
    def _make_task_actionable(self, task: str) -> str:
        """Convert task description to emphasize active execution."""
        task_lower = task.lower()
        
        # If task already starts with action verbs, keep it
        if task_lower.startswith(('monitor', 'track', 'analyze', 'create', 'build', 'develop', 'execute', 'process', 'generate', 'find', 'search', 'collect')):
            return task
        
        # For monitoring/tracking tasks
        if any(word in task_lower for word in ['price', 'prices', 'product', 'monitor', 'track', 'watch']):
            return task.replace('monitors', 'actively monitor').replace('tracking', 'actively track')
        
        # For research tasks  
        if any(word in task_lower for word in ['research', 'find', 'paper', 'study']):
            return task.replace('research', 'conduct research and deliver results for')
        
        # For analysis tasks
        if any(word in task_lower for word in ['analysis', 'analyze', 'data']):
            return task.replace('analysis', 'live analysis and reporting for')
            
        # Default: add execution emphasis
        return f"actively execute and deliver results for: {task}"
    
    def _extract_agent_role(self, analysis: str, base_role: str, task: str) -> str:
        """Generate a specific, contextual role using AI analysis."""
        
        # Create a prompt to generate a specific role based on the task and analysis
        role_prompt = f"""
        Based on this task: "{task}"
        And this analysis: "{analysis}"
        
        Generate a specific, descriptive role for a {base_role} agent that would be perfect for this task.
        
        The role should be:
        - Specific and descriptive (not just generic like "writer" or "analyst")
        - Contextually relevant to the task domain
        - Professional and clear
        - Follow the pattern: "[Domain/Context] [Specific Function]"
        
        Examples of good roles:
        - "LinkedIn Content Strategy Specialist"
        - "Technical Documentation Expert" 
        - "Social Media Analytics Researcher"
        - "Brand Storytelling Creator"
        - "Data Visualization Analyst"
        
        Generate ONLY the role title, nothing else.
        """
        
        try:
            # Use the LLM to generate a contextual role
            from crewai import LLM
            import os
            
            # Use available LLM
            if os.getenv('OPENAI_API_KEY'):
                llm = LLM(model='gpt-4o-mini', temperature=0.3)
            elif os.getenv('ANTHROPIC_API_KEY'):
                llm = LLM(model='claude-3-haiku-20240307', temperature=0.3)
            else:
                # Fallback to a descriptive role if no LLM available
                return self._generate_fallback_role(base_role, task)
            
            # Generate the role using LLM
            response = llm.call([{"role": "system", "content": "You are a professional role generator."}, 
                               {"role": "user", "content": role_prompt}])
            
            generated_role = response.strip().strip('"').strip()
            
            # Validate and clean the generated role
            if len(generated_role) > 5 and len(generated_role) < 100:
                return generated_role
            else:
                return self._generate_fallback_role(base_role, task)
                
        except Exception as e:
            print(f"Warning: Could not generate AI role, using fallback: {e}")
            return self._generate_fallback_role(base_role, task)
    
    def _generate_fallback_role(self, base_role: str, task: str) -> str:
        """Generate a descriptive fallback role when AI generation fails."""
        task_lower = task.lower()
        
        # Context-based role generation
        if any(word in task_lower for word in ['blog', 'article', 'content', 'post', 'social media', 'linkedin']):
            context = "Social Media Content"
        elif any(word in task_lower for word in ['data', 'analysis', 'report', 'research']):
            context = "Data & Analytics"
        elif any(word in task_lower for word in ['code', 'software', 'app', 'develop']):
            context = "Software Development"
        elif any(word in task_lower for word in ['marketing', 'campaign', 'brand']):
            context = "Marketing & Brand"
        else:
            context = "Professional"
        
        role_titles = {
            "researcher": f"{context} Research Specialist",
            "writer": f"{context} Content Creator",
            "analyst": f"{context} Strategy Analyst", 
            "developer": f"{context} Solution Developer",
            "reviewer": f"{context} Quality Specialist",
            "manager": f"{context} Project Manager",
            "coordinator": f"{context} Operations Coordinator",
            "specialist": f"{context} Domain Expert",
            "designer": f"{context} Experience Designer"
        }
        
        return role_titles.get(base_role, f"{context} {base_role.title()} Specialist")

    def _extract_agent_backstory(self, analysis: str, role: str) -> str:
        """Extract agent-specific backstory from analysis."""
        role_backstories = {
            "researcher": "You are an expert researcher with extensive experience in gathering, analyzing, and synthesizing information from diverse sources. You excel at finding reliable data and presenting it clearly.",
            "writer": "You are a skilled writer with expertise in creating compelling content across various formats. You excel at translating complex information into accessible, engaging narratives.",
            "analyst": "You are a data analyst with strong analytical skills and experience in interpreting complex datasets. You excel at identifying patterns, trends, and deriving actionable insights.",
            "developer": "You are a skilled developer with extensive experience in building robust, scalable solutions. You excel at writing clean, efficient code and solving complex technical problems.",
            "reviewer": "You are a meticulous reviewer with a keen eye for detail and quality. You excel at identifying issues, ensuring accuracy, and providing constructive feedback.",
            "coordinator": "You are an experienced project coordinator with expertise in managing teams and workflows. You excel at ensuring smooth collaboration and timely delivery.",
            "specialist": "You are a domain specialist with deep expertise in your field. You excel at providing specialized knowledge and solving complex domain-specific problems.",
            "manager": "You are an experienced project manager with strong leadership skills. You excel at planning, organizing, and directing team efforts to achieve objectives.",
            "designer": "You are a creative designer with expertise in visual communication and user experience. You excel at creating compelling and effective visual solutions."
        }
        return role_backstories.get(role, f"You are an experienced {role} with specialized skills and expertise in your domain.")
    
    def _extract_agent_tools(self, analysis: str, role: str) -> List[str]:
        """Extract agent-specific tools from analysis."""
        # Start with role-based defaults
        base_tools = self._get_tools_for_role(role)
        
        # Try to identify additional tools mentioned in analysis
        analysis_lower = analysis.lower()
        additional_tools = []
        
        tool_indicators = {
            "web_search": ["web search", "internet", "search online", "google"],
            "web_scraping": ["scrape", "crawl", "extract from websites"],
            "document_search": ["documents", "pdf", "files", "papers"],
            "github_search": ["github", "code repository", "source code"],
            "vision": ["images", "visual", "dall-e", "pictures"],
            "database_search": ["database", "sql", "postgres"],
            "code_execution": ["code", "programming", "python", "script"],
            "data_processing": ["data processing", "analytics", "statistics"],
            "browser_automation": ["browser", "automation", "web automation"]
        }
        
        for tool, indicators in tool_indicators.items():
            if any(indicator in analysis_lower for indicator in indicators):
                if tool not in base_tools and tool not in additional_tools:
                    additional_tools.append(tool)
        
        # Combine base tools with additional tools, limit to 4 total
        all_tools = base_tools + additional_tools
        return all_tools[:4]
    
    def _extract_agent_specs_from_analysis(self, analysis: str) -> List[Dict[str, Any]]:
        """Extract complete agent specifications from AI analysis JSON."""
        import re
        import json
        
        try:
            # Use the robust JSON parser
            json_data = self._parse_json_from_text(analysis)
            
            if json_data and 'agentSpecifications' in json_data:
                agent_specs = []
                print(f"🔧 DEBUG: Extracting {len(json_data['agentSpecifications'])} agent specs from JSON")
                for i, agent in enumerate(json_data['agentSpecifications']):
                    spec = {
                        'name': agent.get('agentName', ''),
                        'goal': agent.get('goal', ''),
                        'backstory': agent.get('backstory', ''),
                        'tools': self._convert_ai_tools_to_crewaimaster_tools(agent.get('tools', []))
                    }
                    print(f"🔧 DEBUG: Agent spec {i+1}: {spec['name']} ({agent.get('role', 'Unknown role')})")
                    agent_specs.append(spec)
                return agent_specs
        except (json.JSONDecodeError, KeyError) as e:
            print(f"🔧 DEBUG: Agent spec extraction failed: {e}")
            pass
        
        return []
    
    def _convert_ai_tools_to_crewaimaster_tools(self, ai_tools: List[str]) -> List[str]:
        """Convert AI-suggested tools to actual CrewAI tool names."""
        # Valid CrewAI tools (exact names from CrewAI documentation)
        valid_crewai_tools = {
            'SerperDevTool', 'FileReadTool', 'ScrapeWebsiteTool', 'GithubSearchTool', 
            'YoutubeVideoSearchTool', 'YoutubeChannelSearchTool', 'CodeInterpreterTool',
            'PDFSearchTool', 'DOCXSearchTool', 'CSVSearchTool', 'JSONSearchTool', 
            'XMLSearchTool', 'TXTSearchTool', 'MDXSearchTool', 'DirectoryReadTool', 
            'DirectorySearchTool', 'PGSearchTool', 'BrowserbaseLoadTool', 
            'FirecrawlScrapeWebsiteTool', 'WebsiteSearchTool', 'EXASearchTool',
            'ApifyActorsTool', 'ComposioTool', 'CodeDocsSearchTool', 'RagTool'
        }
        
        # Tool mapping for common aliases to actual CrewAI tools
        tool_mapping = {
            'web_search': 'SerperDevTool',
            'file_operations': 'FileReadTool', 
            'web_scraping': 'ScrapeWebsiteTool',
            'github_search': 'GithubSearchTool',
            'youtube_search': 'YoutubeVideoSearchTool',
            'code_execution': 'CodeInterpreterTool',
            'document_search': 'ScrapeWebsiteTool',  # Changed: For research tasks, prioritize web scraping over file search
            'csv': 'CSVSearchTool',
            'json': 'JSONSearchTool',
            'pdf': 'PDFSearchTool',
            'docx': 'DOCXSearchTool',
            'txt': 'TXTSearchTool',
            'xml': 'XMLSearchTool',
            'markdown': 'MDXSearchTool',
            'database': 'PGSearchTool',
            'postgres': 'PGSearchTool',
            'browser': 'BrowserbaseLoadTool',
            'website_search': 'WebsiteSearchTool',
            'firecrawl': 'FirecrawlScrapeWebsiteTool',
            'beautifulsoup': 'ScrapeWebsiteTool',
            'scrapy': 'ScrapeWebsiteTool', 
            'python': 'CodeInterpreterTool',
            'sqlite': 'PGSearchTool',
            'data_processing': 'CodeInterpreterTool',
            'browser_automation': 'BrowserbaseLoadTool',
            'api_calls': 'SerperDevTool',
            'vision': 'SerperDevTool'
        }
        
        converted_tools = []
        for tool in ai_tools:
            tool_lower = tool.lower()
            
            # If it's already a valid CrewAI tool name, use it directly
            if tool in valid_crewai_tools:
                if tool not in converted_tools:
                    converted_tools.append(tool)
            # Otherwise, try to map it to a CrewAI tool
            else:
                mapped_tool = tool_mapping.get(tool_lower, 'SerperDevTool')
                if mapped_tool not in converted_tools:
                    converted_tools.append(mapped_tool)
        
        # Ensure we have at least SerperDevTool for web search
        if not converted_tools:
            converted_tools = ['SerperDevTool']
            
        return converted_tools[:4]  # Limit to 4 tools
    
    def get_analysis_metrics(self) -> Dict[str, Any]:
        """Get metrics about analysis performance."""
        return {
            "total_analyses": 0,  # Would track actual usage
            "average_analysis_time": 0.0,
            "complexity_distribution": {
                "simple": 0,
                "moderate": 0, 
                "complex": 0
            },
            "most_common_roles": [],
            "tool_usage_stats": {}
        }