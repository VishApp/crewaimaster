"""
Master Agent Tools for CrewMaster.

These tools are used by the MasterAgent crew to analyze tasks, design agents,
and orchestrate crew creation intelligently.
"""

import json
from typing import Dict, List, Any, Optional
from crewai.tools import tool
from ..tools.registry import ToolRegistry


@tool("Task Analysis Tool")
def analyze_task_requirements(task_description: str) -> str:
    """
    Analyze a task description to identify required agent roles, tools, and complexity.
    
    Args:
        task_description: The user's task description to analyze
        
    Returns:
        JSON string with analysis results including roles, tools, and complexity
    """
    analysis = {
        "task": task_description,
        "suggested_roles": [],
        "required_tools": [],
        "complexity": "moderate",
        "estimated_agents": 2,
        "process_type": "sequential",
        "reasoning": ""
    }
    
    task_lower = task_description.lower()
    
    # Analyze for research needs
    if any(word in task_lower for word in ['research', 'find', 'investigate', 'analyze', 'competitor']):
        analysis["suggested_roles"].append({
            "role": "researcher",
            "specialization": "market_research" if "competitor" in task_lower else "general_research",
            "tools": ["web_search", "web_scraping", "document_search"]
        })
    
    # Analyze for data processing needs
    if any(word in task_lower for word in ['data', 'analyze', 'process', 'metrics', 'pricing']):
        analysis["suggested_roles"].append({
            "role": "analyst", 
            "specialization": "data_analysis",
            "tools": ["data_processing", "code_execution", "file_operations"]
        })
    
    # Analyze for content creation needs
    if any(word in task_lower for word in ['write', 'create', 'report', 'document', 'summary']):
        analysis["suggested_roles"].append({
            "role": "writer",
            "specialization": "technical_writing",
            "tools": ["file_operations", "document_search"]
        })
    
    # Analyze for development needs
    if any(word in task_lower for word in ['build', 'develop', 'code', 'api', 'website']):
        analysis["suggested_roles"].append({
            "role": "developer",
            "specialization": "software_development", 
            "tools": ["code_execution", "github_search", "api_calls"]
        })
    
    # Determine complexity
    operation_count = len([w for w in ['and', 'then', 'also', 'plus'] if w in task_lower])
    if operation_count >= 2 or len(task_description.split()) > 25:
        analysis["complexity"] = "complex"
        analysis["estimated_agents"] = min(len(analysis["suggested_roles"]) + 1, 4)
    elif operation_count >= 1 or len(analysis["suggested_roles"]) > 1:
        analysis["complexity"] = "moderate"
        analysis["estimated_agents"] = len(analysis["suggested_roles"])
    else:
        analysis["complexity"] = "simple"
        analysis["estimated_agents"] = 1
    
    # Ensure we have at least one role
    if not analysis["suggested_roles"]:
        analysis["suggested_roles"].append({
            "role": "specialist",
            "specialization": "general_purpose",
            "tools": ["web_search", "file_operations"]
        })
    
    # Collect all required tools
    for role_info in analysis["suggested_roles"]:
        analysis["required_tools"].extend(role_info["tools"])
    analysis["required_tools"] = list(set(analysis["required_tools"]))  # Remove duplicates
    
    analysis["reasoning"] = f"Identified {len(analysis['suggested_roles'])} roles based on task keywords and complexity analysis."
    
    return json.dumps(analysis, indent=2)


@tool("Agent Design Tool")
def design_agent_specification(role_info: str, task_context: str) -> str:
    """
    Design a detailed agent specification based on role requirements and task context.
    
    Args:
        role_info: JSON string with role information from task analysis
        task_context: The original task description for context
        
    Returns:
        JSON string with complete agent specification
    """
    try:
        role_data = json.loads(role_info)
    except:
        role_data = {"role": "specialist", "specialization": "general", "tools": ["web_search"]}
    
    role = role_data.get("role", "specialist")
    specialization = role_data.get("specialization", "general")
    tools = role_data.get("tools", ["web_search"])
    
    # Extract topic from task context for naming
    task_words = task_context.lower().split()
    topic_words = [w for w in task_words[:5] if w not in {'create', 'build', 'make', 'write', 'find', 'analyze', 'help', 'me', 'a', 'an', 'the'}]
    topic = "_".join(topic_words[:2]) if topic_words else "general"
    
    # Role-specific templates
    templates = {
        "researcher": {
            "goal": f"Research and gather comprehensive information about {topic} to support the team's objectives",
            "backstory": "You are an expert researcher with extensive experience in market analysis, data gathering, and information synthesis. You excel at finding reliable sources, extracting key insights, and presenting findings in a clear, actionable format."
        },
        "analyst": {
            "goal": f"Analyze data and information related to {topic} to provide actionable insights and recommendations", 
            "backstory": "You are a skilled data analyst with expertise in processing complex information, identifying patterns and trends, and translating data into strategic insights. You have strong analytical thinking and attention to detail."
        },
        "writer": {
            "goal": f"Create clear, engaging, and well-structured content about {topic} based on research and analysis",
            "backstory": "You are a professional writer with expertise in technical and business communication. You excel at transforming complex information into accessible, compelling narratives that engage readers and drive action."
        },
        "developer": {
            "goal": f"Develop and implement technical solutions related to {topic} with clean, efficient code",
            "backstory": "You are a skilled software developer with expertise in building robust, scalable solutions. You write clean code, follow best practices, and create well-documented technical implementations."
        },
        "specialist": {
            "goal": f"Provide specialized expertise and solutions for {topic} using domain knowledge",
            "backstory": "You are a domain specialist with deep expertise in your field. You provide expert guidance, solve complex problems, and deliver high-quality specialized solutions."
        }
    }
    
    template = templates.get(role, templates["specialist"])
    
    agent_spec = {
        "name": f"{topic}_{role}",
        "role": role,
        "goal": template["goal"],
        "backstory": template["backstory"],
        "tools": tools,
        "specialization": specialization,
        "memory_type": "short_term",
        "max_iter": 5,
        "allow_delegation": len(tools) > 2,
        "verbose": True
    }
    
    return json.dumps(agent_spec, indent=2)


@tool("Crew Orchestration Tool") 
def design_crew_structure(agents_info: str, task_description: str) -> str:
    """
    Design the optimal crew structure and task workflow for the given agents and task.
    
    Args:
        agents_info: JSON string with information about all agents
        task_description: The original task description
        
    Returns:
        JSON string with crew structure and task assignments
    """
    try:
        agents_data = json.loads(agents_info) if isinstance(agents_info, str) else agents_info
        if not isinstance(agents_data, list):
            agents_data = [agents_data]
    except:
        agents_data = []
    
    agent_count = len(agents_data)
    
    # Determine process type
    if agent_count == 1:
        process_type = "sequential"
    elif agent_count <= 3:
        process_type = "sequential"
    else:
        process_type = "hierarchical"
    
    # Create task assignments
    tasks = []
    task_lower = task_description.lower()
    
    for i, agent in enumerate(agents_data):
        agent_role = agent.get("role", "specialist")
        agent_name = agent.get("name", f"agent_{i}")
        
        if agent_role == "researcher":
            task_desc = f"Research and gather information needed for: {task_description}"
            expected_output = "Comprehensive research findings with sources and key insights"
        elif agent_role == "analyst":
            task_desc = f"Analyze the research data and identify patterns, trends, and insights for: {task_description}"
            expected_output = "Detailed analysis with actionable insights and recommendations"
        elif agent_role == "writer":
            task_desc = f"Create well-structured content based on research and analysis for: {task_description}"
            expected_output = "Professional, well-written content that addresses all requirements"
        elif agent_role == "developer":
            task_desc = f"Implement technical solutions based on requirements for: {task_description}"
            expected_output = "Working technical implementation with documentation"
        else:
            task_desc = f"Provide specialized expertise and solutions for: {task_description}"
            expected_output = "Expert recommendations and solutions"
        
        tasks.append({
            "description": task_desc,
            "expected_output": expected_output,
            "agent": agent_name,
            "dependencies": [] if i == 0 else [tasks[i-1]["agent"]]
        })
    
    crew_structure = {
        "name": f"{task_description.split()[:3]}"[0].lower().replace(' ', '_') + "_crew",
        "process_type": process_type,
        "agents": agents_data,
        "tasks": tasks,
        "expected_output": f"Complete solution for: {task_description}",
        "estimated_time": min(5 + (agent_count * 10), 60),
        "memory_enabled": False,
        "verbose": True
    }
    
    return json.dumps(crew_structure, indent=2)


@tool("Available Tools Registry")
def get_available_tools() -> str:
    """
    Get the list of all available tools in the CrewMaster system.
    
    Returns:
        JSON string with available tools and their descriptions
    """
    tool_registry = ToolRegistry()
    
    available_tools = {}
    for tool_name, tool_instance in tool_registry.tools.items():
        available_tools[tool_name] = {
            "name": tool_name,
            "category": tool_instance.category,
            "description": tool_instance.description,
            "capabilities": getattr(tool_instance, 'capabilities', [])
        }
    
    # Add core tools that are always available
    core_tools = {
        "web_search": {
            "name": "web_search",
            "category": "research",
            "description": "Search the web for information",
            "capabilities": ["internet_search", "information_gathering"]
        },
        "file_operations": {
            "name": "file_operations", 
            "category": "utility",
            "description": "Read, write, and manipulate files",
            "capabilities": ["file_io", "document_processing"]
        },
        "data_processing": {
            "name": "data_processing",
            "category": "analysis", 
            "description": "Process and analyze data",
            "capabilities": ["data_analysis", "statistics"]
        },
        "code_execution": {
            "name": "code_execution",
            "category": "development",
            "description": "Execute Python code and scripts",
            "capabilities": ["programming", "automation"]
        }
    }
    
    available_tools.update(core_tools)
    
    return json.dumps(available_tools, indent=2)


@tool("Crew Name Generator")
def generate_crew_name(task_description: str) -> str:
    """
    Generate a meaningful name for a crew based on the task description.
    
    Args:
        task_description: The task the crew will perform
        
    Returns:
        A suitable crew name
    """
    # Extract key words from task description
    words = task_description.lower().split()
    
    # Remove common words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'create', 'make', 'build', 'help', 'me'}
    meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Take first 2-3 meaningful words
    if len(meaningful_words) >= 2:
        crew_name = "_".join(meaningful_words[:2]) + "_crew"
    elif meaningful_words:
        crew_name = meaningful_words[0] + "_crew"
    else:
        crew_name = "general_purpose_crew"
    
    return crew_name