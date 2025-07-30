"""
Python Code Generator for CrewAIMaster
Generates standalone Python code from crew configurations that can be executed anywhere.
"""

import os
import json
import zipfile
import tempfile
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

class CrewCodeGenerator:
    """Generates Python code for crews, agents, and tools."""
    
    def __init__(self):
        self.generated_files = {}
        
    def generate_crew_package(self, crew_data: Dict[str, Any], output_path: str) -> bool:
        """
        Generate a complete Python package for a crew and save as zip file.
        
        Args:
            crew_data: Crew configuration data
            output_path: Path where to save the zip file
            
        Returns:
            bool: Success status
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                package_dir = Path(temp_dir) / f"{crew_data['name']}_crew"
                package_dir.mkdir()
                
                # Generate all files
                self._generate_main_file(crew_data, package_dir)
                self._generate_agents_file(crew_data, package_dir)
                self._generate_tools_file(crew_data, package_dir)
                self._generate_config_file(crew_data, package_dir)
                self._generate_requirements_file(package_dir)
                self._generate_readme_file(crew_data, package_dir)
                self._generate_run_script(crew_data, package_dir)
                self._generate_init_file(package_dir)
                
                # Create zip file
                zip_path = output_path if output_path.endswith('.zip') else f"{output_path}.zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in package_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(temp_dir)
                            zipf.write(file_path, arcname)
                
                return True
                
        except Exception as e:
            print(f"Error generating crew package: {e}")
            return False
    
    def _generate_main_file(self, crew_data: Dict[str, Any], package_dir: Path):
        """Generate the main crew execution file."""
        agents_imports = []
        agent_instances = []
        
        for agent in crew_data['agents']:
            class_name = self._to_class_name(agent['name'])
            agents_imports.append(f"from agents import {class_name}")
            agent_instances.append(f"            '{agent['name']}': {class_name}().agent,")
        
        main_content = f'''"""
{crew_data['name']} - Generated CrewAI Implementation
Generated on: {datetime.now().isoformat()}

This is a standalone implementation that can run anywhere with CrewAI installed.
"""

import os
from crewai import Crew, Process
from crewai.agent import Agent
from crewai.task import Task

# Import agents
{chr(10).join(agents_imports)}

# Import tools
from tools import get_tools_for_agent

class {self._to_class_name(crew_data['name'])}Crew:
    """
    {crew_data['description']}
    
    Original Task: {crew_data['task']}
    """
    
    def __init__(self):
        self.crew_name = "{crew_data['name']}"
        self.agents = self._create_agents()
        self.tasks = self._create_tasks()
        self.crew = self._create_crew()
    
    def _create_agents(self):
        """Create and return all agents for this crew."""
        return {{
{chr(10).join(agent_instances)}
        }}
    
    def _create_tasks(self):
        """Create tasks for the crew."""
        # Main task based on original crew configuration
        main_task = Task(
            description="{crew_data['task']}",
            agent=list(self.agents.values())[0],  # Assign to first agent
            expected_output="Comprehensive results based on the task requirements"
        )
        
        return [main_task]
    
    def _create_crew(self):
        """Create the crew with agents and tasks."""
        return Crew(
            agents=list(self.agents.values()),
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
    
    def run(self, inputs: str = ""):
        """
        Execute the crew with optional inputs.
        
        Args:
            inputs (str): Additional context or inputs for the crew
            
        Returns:
            CrewOutput: Results from crew execution
        """
        # Add inputs to task if provided
        if inputs and self.tasks:
            original_description = self.tasks[0].description
            self.tasks[0].description = f"{{original_description}}\\n\\nAdditional Context: {{inputs}}"
        
        result = self.crew.kickoff()
        return result

def main():
    """Main entry point for running the crew."""
    import sys
    
    # Get inputs from command line or prompt
    inputs = ""
    if len(sys.argv) > 1:
        inputs = " ".join(sys.argv[1:])
    else:
        inputs = input("Enter additional context for the crew (optional): ")
    
    # Create and run crew
    crew = {self._to_class_name(crew_data['name'])}Crew()
    print(f"\\n🚀 Starting {{crew.crew_name}} crew...")
    print(f"📋 Task: {{crew.tasks[0].description if crew.tasks else 'No tasks defined'}}")
    print("-" * 60)
    
    try:
        result = crew.run(inputs)
        print("\\n" + "="*60)
        print("🎉 Crew execution completed!")
        print("📄 Results:")
        print("-" * 60)
        print(result)
        
    except Exception as e:
        print(f"\\n❌ Error during crew execution: {{e}}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
'''
        
        with open(package_dir / "main.py", 'w') as f:
            f.write(main_content)
    
    def _generate_agents_file(self, crew_data: Dict[str, Any], package_dir: Path):
        """Generate the agents module with all agent definitions."""
        agents_content = '''"""
Agent definitions for the crew.
Each agent is implemented as a class with specific tools and capabilities.
"""

from crewai.agent import Agent
from tools import get_tools_for_agent

'''
        
        for agent in crew_data['agents']:
            class_name = self._to_class_name(agent['name'])
            tools_list = agent.get('required_tools', [])
            
            agent_class = f'''
class {class_name}:
    """
    {agent['role']} Agent
    
    Goal: {agent['goal']}
    Background: {agent['backstory']}
    """
    
    def __init__(self):
        self.agent = Agent(
            role="{agent['role']}",
            goal="{agent['goal']}",
            backstory="{agent['backstory']}",
            tools=get_tools_for_agent({tools_list}),
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
    
    def __call__(self):
        return self.agent
'''
            agents_content += agent_class
        
        with open(package_dir / "agents.py", 'w') as f:
            f.write(agents_content)
    
    def _generate_tools_file(self, crew_data: Dict[str, Any], package_dir: Path):
        """Generate the tools module with tool implementations."""
        # Collect all unique tools used by agents
        all_tools = set()
        for agent in crew_data['agents']:
            all_tools.update(agent.get('required_tools', []))
        
        tools_content = f'''"""
Tools implementation for the crew.
Provides all tools needed by the agents.
"""

from typing import List, Any
import os

# Import CrewAI tools
try:
    from crewai_tools import (
        WebsiteSearchTool,SerperDevTool,
        FileReadTool, CodeInterpreterTool,
        PDFSearchTool, CSVSearchTool, TXTSearchTool, DOCXSearchTool,
        YoutubeVideoSearchTool, YoutubeChannelSearchTool,
        VisionTool, GithubSearchTool, BrowserbaseLoadTool,
        ScrapeWebsiteTool, FirecrawlScrapeWebsiteTool
    )
except ImportError as e:
    print(f"Warning: Some CrewAI tools not available: {{e}}")
    print("Install with: pip install crewai-tools")

def get_tools_for_agent(tool_names: List[str]) -> List[Any]:
    """
    Get tool instances for the given tool names.
    
    Args:
        tool_names: List of tool names to instantiate
        
    Returns:
        List of tool instances
    """
    tools = []
    
    for tool_name in tool_names:
        try:
            tool = _create_tool(tool_name)
            if tool:
                tools.append(tool)
        except Exception as e:
            print(f"Warning: Could not create tool '{{tool_name}}': {{e}}")
    
    return tools

def _create_tool(tool_name: str) -> Any:
    """Create a single tool instance by name."""
    tool_mapping = {{
        'web_search': _create_web_search_tool,
        'file_operations': _create_file_operations_tool,
        'code_execution': _create_code_execution_tool,
        'document_search': _create_document_search_tool,
        'youtube_search': _create_youtube_search_tool,
        'vision': _create_vision_tool,
        'github_search': _create_github_search_tool,
        'web_scraping': _create_web_scraping_tool,
        'browser_automation': _create_browser_automation_tool,
    }}
    
    creator_func = tool_mapping.get(tool_name)
    if creator_func:
        return creator_func()
    else:
        print(f"Unknown tool: {{tool_name}}")
        return None

def _create_web_search_tool():
    """Create web search tool."""
    if os.getenv('SERPER_API_KEY'):
        return SerperDevTool()
    else:
        print("Warning: SERPER_API_KEY not set, web search may not work optimally")
        return SerperDevTool()  # CrewAI handles fallbacks

def _create_file_operations_tool():
    """Create file operations tool."""
    return FileReadTool()

def _create_code_execution_tool():
    """Create code execution tool."""
    return CodeInterpreterTool()

def _create_document_search_tool():
    """Create document search tool."""
    # Return PDF search tool as default
    return PDFSearchTool()

def _create_youtube_search_tool():
    """Create YouTube search tool."""
    return YoutubeVideoSearchTool()

def _create_vision_tool():
    """Create vision tool."""
    if os.getenv('OPENAI_API_KEY'):
        return VisionTool()
    else:
        print("Warning: OPENAI_API_KEY not set, vision tool may not work")
        return VisionTool()

def _create_github_search_tool():
    """Create GitHub search tool."""
    if os.getenv('GITHUB_TOKEN'):
        return GithubSearchTool()
    else:
        print("Warning: GITHUB_TOKEN not set, GitHub search may not work")
        return GithubSearchTool()

def _create_web_scraping_tool():
    """Create web scraping tool."""
    if os.getenv('FIRECRAWL_API_KEY'):
        return FirecrawlScrapeWebsiteTool()
    else:
        return ScrapeWebsiteTool()

def _create_browser_automation_tool():
    """Create browser automation tool."""
    if os.getenv('BROWSERBASE_API_KEY'):
        return BrowserbaseLoadTool()
    else:
        print("Warning: BROWSERBASE_API_KEY not set, browser automation may not work")
        return BrowserbaseLoadTool()

# Tool validation
REQUIRED_TOOLS = {list(all_tools)}
AVAILABLE_TOOLS = [
    'web_search', 'file_operations', 'code_execution', 'document_search',
    'youtube_search', 'vision', 'github_search', 'web_scraping', 'browser_automation'
]

def validate_tools():
    """Validate that all required tools are available."""
    missing_tools = []
    for tool in REQUIRED_TOOLS:
        if tool not in AVAILABLE_TOOLS:
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"Warning: These tools are not implemented: {{missing_tools}}")
    
    return len(missing_tools) == 0

# Environment check
def check_environment():
    """Check if required environment variables are set."""
    env_warnings = []
    
    if 'web_search' in REQUIRED_TOOLS and not os.getenv('SERPER_API_KEY'):
        env_warnings.append("SERPER_API_KEY not set - web search may be limited")
    
    if 'vision' in REQUIRED_TOOLS and not os.getenv('OPENAI_API_KEY'):
        env_warnings.append("OPENAI_API_KEY not set - vision tools may not work")
    
    if 'github_search' in REQUIRED_TOOLS and not os.getenv('GITHUB_TOKEN'):
        env_warnings.append("GITHUB_TOKEN not set - GitHub search may not work")
    
    if env_warnings:
        print("Environment Warnings:")
        for warning in env_warnings:
            print(f"  - {{warning}}")
        print()
'''
        
        with open(package_dir / "tools.py", 'w') as f:
            f.write(tools_content)
    
    def _generate_config_file(self, crew_data: Dict[str, Any], package_dir: Path):
        """Generate configuration file."""
        config_content = f'''"""
Configuration for {crew_data['name']} crew.
"""

# Crew metadata
CREW_NAME = "{crew_data['name']}"
CREW_DESCRIPTION = "{crew_data['description']}"
CREW_TASK = "{crew_data['task']}"

# Generation info
GENERATED_ON = "{datetime.now().isoformat()}"
crewaimaster_VERSION = "1.0"

# Agent configuration
AGENTS = {json.dumps([{
    'name': agent['name'],
    'role': agent['role'],
    'goal': agent['goal'],
    'backstory': agent['backstory'],
    'tools': agent.get('required_tools', [])
} for agent in crew_data['agents']], indent=4)}

# Environment variables required
REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",  # For LLM and some tools
    "SERPER_API_KEY",  # For web search (optional)
]

# Optional environment variables
OPTIONAL_ENV_VARS = [
    "GITHUB_TOKEN",      # For GitHub search
    "FIRECRAWL_API_KEY", # For web scraping
    "BROWSERBASE_API_KEY", # For browser automation
    "YOUTUBE_API_KEY",   # For YouTube search
]
'''
        
        with open(package_dir / "config.py", 'w') as f:
            f.write(config_content)
    
    def _generate_requirements_file(self, package_dir: Path):
        """Generate requirements.txt file."""
        requirements_content = '''# Core CrewAI dependencies
crewai>=0.28.0
crewai-tools>=0.1.0

# LLM providers
openai>=1.0.0
anthropic>=0.8.0

# Optional tools dependencies
firecrawl-py>=0.0.8
browserbase>=0.1.0

# Utility dependencies
python-dotenv>=1.0.0
pydantic>=2.0.0
'''
        
        with open(package_dir / "requirements.txt", 'w') as f:
            f.write(requirements_content)
    
    def _generate_readme_file(self, crew_data: Dict[str, Any], package_dir: Path):
        """Generate README.md file."""
        agents_list = '\n'.join([
            f"- **{agent['name']}** ({agent['role']}): {agent['goal']}"
            for agent in crew_data['agents']
        ])
        
        tools_set = set()
        for agent in crew_data['agents']:
            tools_set.update(agent.get('required_tools', []))
        tools_list = '\n'.join([f"- {tool}" for tool in sorted(tools_set)])
        
        readme_content = f'''# {crew_data['name']} Crew

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Description
{crew_data['description']}

## Original Task
{crew_data['task']}

## Agents
{agents_list}

## Tools Used
{tools_list}

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
export SERPER_API_KEY="your-serper-api-key"  # Optional
```

### 3. Run the Crew
```bash
python main.py
```

Or with additional context:
```bash
python main.py "Your additional instructions here"
```

### 4. Use as a Module
```python
from main import {self._to_class_name(crew_data['name'])}Crew

crew = {self._to_class_name(crew_data['name'])}Crew()
result = crew.run("Your inputs here")
print(result)
```

## Environment Variables

### Required
- `OPENAI_API_KEY`: Your OpenAI API key for LLM functionality

### Optional (depending on tools used)
- `SERPER_API_KEY`: For web search functionality
- `GITHUB_TOKEN`: For GitHub repository search
- `FIRECRAWL_API_KEY`: For advanced web scraping
- `BROWSERBASE_API_KEY`: For browser automation
- `YOUTUBE_API_KEY`: For YouTube search

## File Structure
```
{crew_data['name']}_crew/
├── main.py           # Main crew execution
├── agents.py         # Agent definitions
├── tools.py          # Tool implementations
├── config.py         # Configuration
├── requirements.txt  # Dependencies
├── run.sh           # Execution script
└── README.md        # This file
```

## Generated by CrewAIMaster
This crew was automatically generated from CrewAIMaster configuration.
You can modify the code as needed for your specific requirements.

For more information about CrewAI, visit: https://github.com/joaomdmoura/crewAI
'''
        
        with open(package_dir / "README.md", 'w') as f:
            f.write(readme_content)
    
    def _generate_run_script(self, crew_data: Dict[str, Any], package_dir: Path):
        """Generate executable run script."""
        script_content = f'''#!/bin/bash

# {crew_data['name']} Crew Execution Script
# Generated by CrewAIMaster

echo "🚀 Starting {crew_data['name']} crew..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo "Set it with: export OPENAI_API_KEY='your-key'"
fi

# Run the crew
echo "🏃 Running crew..."
python main.py "$@"

echo "✅ Crew execution completed!"
'''
        
        script_path = package_dir / "run.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        script_path.chmod(0o755)
    
    def _generate_init_file(self, package_dir: Path):
        """Generate __init__.py file."""
        init_content = '''"""
Generated CrewAI implementation package.
"""

from .main import *

__version__ = "1.0.0"
__generated_by__ = "CrewAIMaster"
'''
        
        with open(package_dir / "__init__.py", 'w') as f:
            f.write(init_content)
    
    def _to_class_name(self, name: str) -> str:
        """Convert a name to a valid Python class name."""
        # Remove special characters and convert to PascalCase
        clean_name = ''.join(word.capitalize() for word in name.replace('-', '_').replace(' ', '_').split('_') if word.isalnum())
        return clean_name if clean_name else "GeneratedClass"

def generate_crew_code_package(crew_data: Dict[str, Any], output_path: str) -> bool:
    """
    Public function to generate a crew code package.
    
    Args:
        crew_data: Dictionary containing crew configuration
        output_path: Path where to save the zip file
        
    Returns:
        bool: True if successful, False otherwise
    """
    generator = CrewCodeGenerator()
    return generator.generate_crew_package(crew_data, output_path)