"""
Task analysis module for CrewAIMaster.

This module analyzes natural language task descriptions and determines:
- Required agent roles and capabilities
- Appropriate tools for each agent
- Task complexity and crew structure
- Memory and knowledge requirements
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"      # Single agent, basic tools
    MODERATE = "moderate"  # 2-3 agents, multiple tools
    COMPLEX = "complex"    # 3+ agents, advanced coordination

class AgentRole(Enum):
    """Predefined agent roles."""
    RESEARCHER = "researcher"
    WRITER = "writer"
    ANALYST = "analyst"
    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"

@dataclass
class TaskRequirement:
    """Represents a specific requirement extracted from task analysis."""
    requirement_type: str  # "data_source", "output_format", "skill", etc.
    value: str
    priority: int  # 1-5, higher is more important
    description: str

@dataclass
class AgentSpec:
    """Specification for creating an agent."""
    role: str
    name: str
    goal: str
    backstory: str
    required_tools: List[str]
    memory_type: str = "short_term"
    max_iter: int = 5
    allow_delegation: bool = False

@dataclass
class CrewSpec:
    """Specification for creating a crew."""
    name: str
    task: str
    description: str
    agents: List[AgentSpec]
    expected_output: str
    complexity: TaskComplexity
    estimated_time: int  # minutes
    process_type: str = "sequential"

class TaskAnalyzer:
    """Analyzes task descriptions and generates crew specifications."""
    
    def __init__(self):
        """Initialize the task analyzer."""
        self.role_patterns = {
            AgentRole.RESEARCHER: [
                r'\b(research|investigate|find|discover|explore|analyze data)\b',
                r'\b(gather information|collect data|study|examine)\b',
                r'\b(look up|search for|identify sources)\b'
            ],
            AgentRole.WRITER: [
                r'\b(write|create|compose|draft|generate)\b',
                r'\b(article|blog|report|documentation|content)\b',
                r'\b(summarize|explain|describe)\b'
            ],
            AgentRole.ANALYST: [
                r'\b(analyze|evaluate|assess|compare|review)\b',
                r'\b(data analysis|statistics|metrics|trends)\b',
                r'\b(performance|insights|patterns)\b'
            ],
            AgentRole.DEVELOPER: [
                r'\b(develop|build|create|implement|code)\b',
                r'\b(software|application|website|api|system)\b',
                r'\b(programming|debugging|testing)\b'
            ],
            AgentRole.REVIEWER: [
                r'\b(review|check|validate|verify|audit)\b',
                r'\b(quality|accuracy|compliance|standards)\b',
                r'\b(feedback|critique|evaluation)\b'
            ]
        }
        
        self.tool_patterns = {
            "web_search": [
                r'\b(search|google|find online|web search|internet|research)\b',
                r'\b(latest|current|recent|news|trends|urls|links|websites)\b',
                r'\b(fetch|retrieve|get|obtain|gather)\b'
            ],
            "file_operations": [
                r'\b(file|document|read|write|save|export)\b',
                r'\b(upload|download|process file|fetch file|store)\b'
            ],
            "code_execution": [
                r'\b(python|javascript|code|script|programming)\b',
                r'\b(execute|run|compile|debug|interpret)\b'
            ],
            "api_calls": [
                r'\b(api|service|endpoint|integration|webhook)\b',
                r'\b(rest|graphql|http|request)\b'
            ],
            "data_processing": [
                r'\b(data|dataset|database|sql|analytics)\b',
                r'\b(process|transform|clean|parse)\b'
            ],
            "email": [
                r'\b(email|send|notify|alert|message)\b',
                r'\b(gmail|outlook|mail|communication)\b'
            ],
            "scheduling": [
                r'\b(schedule|calendar|appointment|meeting|time)\b',
                r'\b(remind|recurring|daily|weekly)\b'
            ],
            
            # New CrewAI tools
            "web_scraping": [
                r'\b(scrape|crawl|extract|harvest)\b',
                r'\b(website|webpage|web page|html)\b',
                r'\b(firecrawl|scraping|crawling)\b'
            ],
            "document_search": [
                r'\b(pdf|docx|txt|csv|json|xml|md|markdown)\b',
                r'\b(document|paper|report|file search)\b',
                r'\b(search in|find in|extract from)\b'
            ],
            "github_search": [
                r'\b(github|git|repository|repo|code search)\b',
                r'\b(source code|codebase|programming)\b',
                r'\b(commit|branch|pull request)\b'
            ],
            "youtube_search": [
                r'\b(youtube|video|channel|content)\b',
                r'\b(watch|view|analyze video)\b',
                r'\b(video content|video analysis)\b'
            ],
            "vision": [
                r'\b(image|photo|picture|visual)\b',
                r'\b(dall-e|generate image|create image)\b',
                r'\b(vision|analyze image|describe image)\b'
            ],
            "database_search": [
                r'\b(postgresql|postgres|database query)\b',
                r'\b(sql|query|database search)\b',
                r'\b(db|database|table)\b'
            ],
            "browser_automation": [
                r'\b(browser|automation|browserbase)\b',
                r'\b(automate|interact|control browser)\b',
                r'\b(selenium|playwright|browser control)\b'
            ]
        }
        
        self.complexity_indicators = {
            TaskComplexity.SIMPLE: [
                r'\b(simple|basic|quick|easy|straightforward)\b',
                r'\b(single|one|just)\b'
            ],
            TaskComplexity.COMPLEX: [
                r'\b(complex|comprehensive|detailed|thorough|complete)\b',
                r'\b(multiple|several|various|many|different)\b',
                r'\b(integrate|coordinate|manage|orchestrate)\b'
            ]
        }
    
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
        """Analyze a task description and return a crew specification."""
        # Normalize the task description first
        normalized_task = self._normalize_task_description(task_description)
        task_lower = normalized_task.lower()
        
        # Extract requirements using normalized task
        requirements = self._extract_requirements(normalized_task)
        
        # Determine complexity
        complexity = self._determine_complexity(task_lower)
        
        # Identify required roles
        required_roles = self._identify_roles(task_lower)
        
        # Generate agent specifications using normalized task
        agents = self._generate_agent_specs(normalized_task, required_roles, requirements)
        
        # Determine process type and crew structure
        process_type = self._determine_process_type(complexity, len(agents))
        
        # Generate crew name and expected output using normalized task
        crew_name = self._generate_crew_name(normalized_task)
        expected_output = self._generate_expected_output(normalized_task, agents)
        
        # Estimate execution time
        estimated_time = self._estimate_execution_time(complexity, len(agents))
        
        return CrewSpec(
            name=crew_name,
            task=normalized_task,  # Use normalized task for execution
            description=f"Crew designed to {normalized_task.lower()}",
            agents=agents,
            process_type=process_type,
            expected_output=expected_output,
            complexity=complexity,
            estimated_time=estimated_time
        )
    
    def _extract_requirements(self, task_description: str) -> List[TaskRequirement]:
        """Extract specific requirements from task description."""
        requirements = []
        task_lower = task_description.lower()
        
        # Output format requirements
        output_formats = {
            r'\b(report|document)\b': "report",
            r'\b(summary|summarize)\b': "summary",
            r'\b(list|listing)\b': "list", 
            r'\b(analysis|analyze)\b': "analysis",
            r'\b(comparison|compare)\b': "comparison",
            r'\b(presentation|slides)\b': "presentation",
            r'\b(email|message)\b': "email",
            r'\b(code|script|program)\b': "code"
        }
        
        for pattern, format_type in output_formats.items():
            if re.search(pattern, task_lower):
                requirements.append(TaskRequirement(
                    requirement_type="output_format",
                    value=format_type,
                    priority=4,
                    description=f"Output should be in {format_type} format"
                ))
        
        # Data source requirements
        data_sources = {
            r'\b(web|internet|online)\b': "web_search",
            r'\b(file|document|pdf)\b': "file_processing",
            r'\b(database|sql|data)\b': "database",
            r'\b(api|service|endpoint)\b': "api_integration"
        }
        
        for pattern, source_type in data_sources.items():
            if re.search(pattern, task_lower):
                requirements.append(TaskRequirement(
                    requirement_type="data_source",
                    value=source_type,
                    priority=5,
                    description=f"Requires access to {source_type}"
                ))
        
        # Time sensitivity
        urgency_patterns = {
            r'\b(urgent|asap|immediately|quick|fast)\b': "high",
            r'\b(daily|regular|scheduled|routine)\b': "recurring",
            r'\b(when|if|trigger)\b': "conditional"
        }
        
        for pattern, urgency in urgency_patterns.items():
            if re.search(pattern, task_lower):
                requirements.append(TaskRequirement(
                    requirement_type="urgency",
                    value=urgency,
                    priority=3,
                    description=f"Task has {urgency} urgency"
                ))
        
        return requirements
    
    def _determine_complexity(self, task_description: str) -> TaskComplexity:
        """Determine task complexity based on description."""
        # Count complexity indicators
        simple_score = sum(1 for pattern in self.complexity_indicators[TaskComplexity.SIMPLE] 
                          if re.search(pattern, task_description))
        complex_score = sum(1 for pattern in self.complexity_indicators[TaskComplexity.COMPLEX] 
                           if re.search(pattern, task_description))
        
        # Check for multiple operations
        operation_count = len(re.findall(r'\b(and|then|after|also|plus|additionally)\b', task_description))
        
        # Determine complexity
        if complex_score > simple_score or operation_count >= 3:
            return TaskComplexity.COMPLEX
        elif operation_count >= 1 or len(task_description.split()) > 20:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def _identify_roles(self, task_description: str) -> List[AgentRole]:
        """Identify required agent roles based on task description."""
        identified_roles = []
        
        for role, patterns in self.role_patterns.items():
            if any(re.search(pattern, task_description) for pattern in patterns):
                identified_roles.append(role)
        
        # Ensure we have at least one role
        if not identified_roles:
            # Default to researcher for analysis/information tasks
            if any(word in task_description for word in ['find', 'get', 'information', 'about']):
                identified_roles.append(AgentRole.RESEARCHER)
            else:
                identified_roles.append(AgentRole.SPECIALIST)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(identified_roles))
    
    def _generate_agent_specs(self, task_description: str, roles: List[AgentRole], 
                            requirements: List[TaskRequirement]) -> List[AgentSpec]:
        """Generate agent specifications for the identified roles."""
        agents = []
        task_lower = task_description.lower()
        
        # Extract required tools based on task content
        required_tools = []
        for tool, patterns in self.tool_patterns.items():
            if any(re.search(pattern, task_lower) for pattern in patterns):
                required_tools.append(tool)
        
        for role in roles:
            agent_spec = self._create_agent_spec(role, task_description, required_tools, requirements)
            agents.append(agent_spec)
        
        return agents
    
    def _create_agent_spec(self, role: AgentRole, task_description: str, 
                          available_tools: List[str], requirements: List[TaskRequirement]) -> AgentSpec:
        """Create a specific agent specification for a role."""
        role_templates = {
            AgentRole.RESEARCHER: {
                "name_template": "{topic}_researcher",
                "goal_template": "Actively execute research and deliver real data about {topic}",
                "backstory_template": "You are an expert researcher with extensive hands-on experience in gathering, analyzing, and delivering actionable information. You excel at finding reliable data and immediately acting on research tasks to produce concrete results.",
                "preferred_tools": ["web_search", "file_operations", "data_processing"]
            },
            AgentRole.WRITER: {
                "name_template": "{topic}_writer", 
                "goal_template": "Execute content creation and deliver completed written materials about {topic}",
                "backstory_template": "You are a skilled writer with expertise in creating and delivering compelling content. You excel at translating requirements into finished written products and completing content creation tasks efficiently.",
                "preferred_tools": ["file_operations", "web_search"]
            },
            AgentRole.ANALYST: {
                "name_template": "{topic}_analyst",
                "goal_template": "Execute data analysis and deliver actionable insights for {topic}",
                "backstory_template": "You are a data analyst with strong hands-on experience in processing datasets and delivering analysis results. You excel at identifying patterns, trends, and producing actionable analytical outputs.",
                "preferred_tools": ["data_processing", "file_operations", "code_execution"]
            },
            AgentRole.DEVELOPER: {
                "name_template": "{topic}_developer",
                "goal_template": "Execute development work and deliver working technical solutions for {topic}",
                "backstory_template": "You are a skilled developer with extensive experience in building and deploying working solutions. You excel at writing functional code, implementing features, and delivering completed technical products.",
                "preferred_tools": ["code_execution", "api_calls", "file_operations"]
            },
            AgentRole.REVIEWER: {
                "name_template": "{topic}_reviewer",
                "goal_template": "Review and validate the quality and accuracy of work related to {topic}",
                "backstory_template": "You are a meticulous reviewer with a keen eye for detail and quality. You excel at identifying issues, ensuring accuracy, and providing constructive feedback.",
                "preferred_tools": ["file_operations", "web_search"]
            },
            AgentRole.SPECIALIST: {
                "name_template": "{topic}_specialist",
                "goal_template": "Execute specialized tasks and deliver expert results for {topic}",
                "backstory_template": "You are a domain specialist with deep hands-on expertise in executing specialized tasks. You excel at applying domain knowledge to complete complex work and deliver expert-level results efficiently.",
                "preferred_tools": ["web_search", "file_operations", "data_processing"]
            }
        }
        
        # Extract topic from task description
        topic = self._extract_topic(task_description)
        
        template = role_templates.get(role, role_templates[AgentRole.SPECIALIST])
        
        # Select appropriate tools
        agent_tools = []
        preferred_tools = template["preferred_tools"]
        
        # Add preferred tools that are available
        for tool in preferred_tools:
            if tool in available_tools:
                agent_tools.append(tool)
        
        # Add any other tools that might be needed
        for tool in available_tools:
            if tool not in agent_tools and len(agent_tools) < 3:  # Limit to 3 tools per agent
                agent_tools.append(tool)
        
        # Determine memory type based on requirements
        memory_type = "short_term"
        for req in requirements:
            if req.requirement_type == "urgency" and req.value == "recurring":
                memory_type = "long_term"
                break
        
        return AgentSpec(
            role=role.value,
            name=template["name_template"].format(topic=topic.lower().replace(" ", "_")),
            goal=template["goal_template"].format(topic=topic),
            backstory=template["backstory_template"],
            required_tools=agent_tools,
            memory_type=memory_type,
            max_iter=5,
            allow_delegation=len([r for r in role_templates if r != role]) > 0
        )
    
    def _extract_topic(self, task_description: str) -> str:
        """Extract the main topic from task description."""
        import re
        
        # Enhanced skip words list
        skip_words = {
            'create', 'build', 'make', 'develop', 'design', 'write', 'generate',
            'find', 'search', 'analyze', 'help', 'me', 'us', 'a', 'an', 'the', 
            'for', 'about', 'on', 'with', 'that', 'which', 'this', 'to', 'and',
            'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has'
        }
        
        # Clean the description and extract words
        clean_desc = re.sub(r'[^\w\s]', ' ', task_description.lower())
        words = clean_desc.split()
        
        # Filter meaningful words (longer than 2 chars, not in skip list)
        meaningful_words = [
            word for word in words 
            if word not in skip_words and len(word) > 2 and word.isalpha()
        ]
        
        if meaningful_words:
            # Take up to 3 most meaningful words
            return " ".join(meaningful_words[:3])
        else:
            # Fallback: take any non-skip words
            fallback_words = [word for word in words[:5] if word not in skip_words]
            if fallback_words:
                return " ".join(fallback_words[:2])
            else:
                return "general_task"
    
    def _determine_process_type(self, complexity: TaskComplexity, agent_count: int) -> str:
        """Determine the appropriate process type for the crew."""
        if agent_count == 1:
            return "sequential"
        elif complexity == TaskComplexity.SIMPLE and agent_count <= 2:
            return "sequential"
        elif complexity == TaskComplexity.COMPLEX and agent_count > 3:
            return "hierarchical"
        else:
            return "sequential"
    
    def _generate_crew_name(self, task_description: str) -> str:
        """Generate a descriptive name for the crew."""
        topic = self._extract_topic(task_description)
        clean_topic = topic.replace(' ', '_').lower()
        
        # Avoid double "crew" suffix
        if clean_topic.endswith('_crew'):
            return clean_topic
        else:
            return f"{clean_topic}_crew"
    
    def _generate_expected_output(self, task_description: str, agents: List[AgentSpec]) -> str:
        """Generate execution-focused expected output description based on task and agents."""
        task_lower = task_description.lower()
        
        # For monitoring/tracking tasks - expect actual data/results
        if any(word in task_lower for word in ['monitor', 'track', 'price', 'watch']):
            return f"Real-time monitoring results and collected price data from executing: {task_description}"
        
        # For research tasks - expect findings/results
        if any(word in task_lower for word in ['research', 'find', 'paper', 'study']):
            return f"Research results and found materials from executing: {task_description}"
        
        # For analysis tasks - expect insights/conclusions
        if any(word in task_lower for word in ['analysis', 'analyze', 'data']):
            return f"Analysis results and actionable insights from executing: {task_description}"
        
        # For development tasks - expect working solutions
        if any(word in task_lower for word in ['develop', 'build', 'create', 'implement']):
            return f"Working solution and completed implementation for: {task_description}"
        
        # Extract output format from requirements if specified
        output_indicators = {
            'report': 'completed report with execution results',
            'summary': 'comprehensive summary of execution results',
            'list': 'organized list of collected results',
            'code': 'functional working code solution',
            'presentation': 'completed presentation materials'
        }
        
        for indicator, output_type in output_indicators.items():
            if indicator in task_lower:
                return f"A {output_type} for: {task_description}"
        
        # Default based on agent roles - focus on execution results
        if any(agent.role == 'writer' for agent in agents):
            return f"Completed written content and deliverables for: {task_description}"
        elif any(agent.role == 'analyst' for agent in agents):
            return f"Analysis results and actionable insights for: {task_description}"
        elif any(agent.role == 'developer' for agent in agents):
            return f"Working technical solution and implementation for: {task_description}"
        else:
            return f"Complete execution results and deliverables for: {task_description}"
    
    def _estimate_execution_time(self, complexity: TaskComplexity, agent_count: int) -> int:
        """Estimate execution time in minutes."""
        base_time = {
            TaskComplexity.SIMPLE: 5,
            TaskComplexity.MODERATE: 15,
            TaskComplexity.COMPLEX: 30
        }
        
        # Add time per additional agent
        additional_time = (agent_count - 1) * 5
        
        return base_time[complexity] + additional_time