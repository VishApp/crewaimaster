"""
File-based crew generation system for CrewMaster.

This module generates proper CrewAI project structures with YAML configurations
and Python modules, making crews version-controllable and easily shareable.
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..core.task_analyzer import CrewSpec, AgentSpec


class CrewFileGenerator:
    """Generates file-based CrewAI projects from specifications."""
    
    def __init__(self, crews_base_path: str = "crews"):
        """Initialize the file generator."""
        self.crews_base_path = Path(crews_base_path)
        self.crews_base_path.mkdir(exist_ok=True)
    
    def generate_crew_project(self, spec: CrewSpec) -> str:
        """Generate a complete CrewAI project structure."""
        crew_path = self.crews_base_path / spec.name
        
        # Check if crew already exists
        if crew_path.exists():
            raise ValueError(f"Crew '{spec.name}' already exists at {crew_path}")
        
        # Create directory structure
        self._create_directory_structure(crew_path, spec.name)
        
        # Generate configuration files
        self._generate_agents_yaml(crew_path, spec.agents)
        self._generate_tasks_yaml(crew_path, spec)
        
        # Generate Python modules
        self._generate_main_py(crew_path, spec)
        self._generate_crew_py(crew_path, spec)
        self._generate_tools_py(crew_path, spec)
        
        # Generate supporting files
        self._generate_requirements_txt(crew_path, spec)
        self._generate_run_script(crew_path, spec)
        self._generate_readme(crew_path, spec)
        self._generate_init_files(crew_path, spec.name)
        
        return str(crew_path)
    
    def _create_directory_structure(self, crew_path: Path, crew_name: str):
        """Create the basic directory structure."""
        directories = [
            crew_path,
            crew_path / "src" / crew_name,
            crew_path / "src" / crew_name / "tools",
            crew_path / "config"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _generate_agents_yaml(self, crew_path: Path, agents: List[AgentSpec]):
        """Generate agents.yaml configuration file."""
        agents_config = {}
        
        for agent in agents:
            agent_config = {
                'role': agent.role,
                'goal': agent.goal,
                'backstory': agent.backstory,
                'verbose': True,
                'allow_delegation': agent.allow_delegation,
                'max_iter': agent.max_iter,
                'tools': agent.required_tools,
                'llm': {
                    'provider': 'openai',  # Can be customized
                    'model': 'gpt-4o-mini',
                    'temperature': 0.7,
                    'max_tokens': 2000,
                    # Additional parameters available:
                    # 'top_p': 0.9,
                    # 'frequency_penalty': 0.1,
                    # 'presence_penalty': 0.1,
                    # 'stop': ['###', '---'],
                    # 'timeout': 30,
                    # 'max_retries': 3,
                    # 'api_key': 'agent-specific-key',
                    # 'base_url': 'https://api.openai.com/v1',
                    # 'api_version': '2023-05-15',
                    # 'organization': 'your-org-id'
                }
            }
            
            agents_config[agent.name] = agent_config
        
        config_path = crew_path / "config" / "agents.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(agents_config, f, default_flow_style=False, indent=2)
    
    def _generate_tasks_yaml(self, crew_path: Path, spec: CrewSpec):
        """Generate tasks.yaml configuration file."""
        tasks_config = {
            'main_task': {
                'description': spec.task,
                'expected_output': spec.expected_output or f"Complete results for: {spec.task}",
                'agent': spec.agents[0].name if spec.agents else 'researcher'
            }
        }
        
        # Add collaborative tasks for additional agents
        if len(spec.agents) > 1:
            for i, agent in enumerate(spec.agents[1:], 1):
                task_name = f"{agent.role.lower()}_task"
                
                # Create action-oriented task descriptions based on role
                if agent.role.lower() == 'data_analyst':
                    description = f"As a {agent.role}, analyze the research data and create CSV/Excel files with processed findings, charts, and data visualizations."
                    expected_output = "Structured data files (CSV/Excel) with analysis, charts, and actionable insights"
                elif agent.role.lower() == 'report_writer':
                    description = f"As a {agent.role}, compile the analysis into a comprehensive report and create PDF/DOCX documents with professional formatting."
                    expected_output = "Final report documents (PDF/DOCX) with proper formatting, conclusions, and recommendations"
                else:
                    description = f"As a {agent.role}, process the research findings and create deliverable outputs based on your expertise."
                    expected_output = f"Actionable deliverables and outputs from a {agent.role} perspective"
                
                tasks_config[task_name] = {
                    'description': description,
                    'expected_output': expected_output,
                    'agent': agent.name,
                    'context': ['main_task'] if i == 1 else [f"{spec.agents[i-1].role.lower()}_task"]
                }
        
        config_path = crew_path / "config" / "tasks.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(tasks_config, f, default_flow_style=False, indent=2)
    
    def _generate_main_py(self, crew_path: Path, spec: CrewSpec):
        """Generate main.py entry point."""
        content = f'''#!/usr/bin/env python3
"""
{spec.name} - CrewAI Project

{spec.description or f"A CrewAI project for {spec.task}"}

Generated by CrewMaster on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import sys
from .crew import {self._to_class_name(spec.name)}


def main():
    """Main entry point for the crew."""
    # Get task input from command line arguments
    task_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "{spec.task}"
    
    # Initialize and run the crew
    crew = {self._to_class_name(spec.name)}()
    result = crew.run(task_input)
    
    print("\\n" + "="*50)
    print("CREW EXECUTION COMPLETED")
    print("="*50)
    print(result)
    
    return result


if __name__ == "__main__":
    main()
'''
        
        main_path = crew_path / "src" / spec.name / "main.py"
        with open(main_path, 'w') as f:
            f.write(content)
    
    def _generate_crew_py(self, crew_path: Path, spec: CrewSpec):
        """Generate crew.py orchestration logic."""
        content = f'''"""
{spec.name} Crew Implementation

This module contains the main crew logic and orchestration.
"""

import yaml
import os
from pathlib import Path
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from .tools.custom_tools import get_tools_for_agent


class {self._to_class_name(spec.name)}:
    """Main crew class for {spec.name}."""
    
    def __init__(self):
        """Initialize the crew."""
        self.config_path = Path(__file__).parent.parent.parent / "config"
        self.agents_config = self._load_config("agents.yaml")
        self.tasks_config = self._load_config("tasks.yaml")
        
        # Setup LLM configuration
        self.llm = self._setup_llm()
        
        # Initialize agents and tasks
        self.agents = self._create_agents()
        self.tasks = self._create_tasks()
        
        # Create the crew
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=list(self.tasks.values()),
            process=Process.{spec.process_type},
            verbose=True,
            memory=False  # Can be enabled as needed
        )
    
    def _load_config(self, filename: str) -> dict:
        """Load configuration from YAML file."""
        config_file = self.config_path / filename
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_llm(self) -> LLM:
        """Setup LLM configuration for CrewAI."""
        # Check for CrewMaster environment variables first
        provider = os.getenv('CREWMASTER_LLM_PROVIDER', 'openai')
        model = os.getenv('CREWMASTER_LLM_MODEL', 'gpt-4')
        api_key = os.getenv('CREWMASTER_LLM_API_KEY')
        base_url = os.getenv('CREWMASTER_LLM_BASE_URL')
        
        # If custom provider configuration exists, use it
        if provider == 'custom' and api_key and base_url:
            return LLM(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.7,
                max_tokens=2000
            )
        
        # Otherwise, use standard providers (OpenAI, Anthropic, etc.)
        # CrewAI will auto-detect based on environment variables
        if os.getenv('OPENAI_API_KEY'):
            return LLM(model='gpt-4', temperature=0.7)
        elif os.getenv('ANTHROPIC_API_KEY'):
            return LLM(model='claude-3-sonnet-20240229', temperature=0.7)
        elif os.getenv('GOOGLE_API_KEY'):
            return LLM(model='gemini-pro', temperature=0.7)
        else:
            # Default to OpenAI (will fail if no API key, but that's expected)
            return LLM(model='gpt-3.5-turbo', temperature=0.7)
    
    def _create_agent_llm(self, llm_config: dict) -> LLM:
        """Create LLM instance for a specific agent."""
        provider = llm_config.get('provider', 'openai')
        model = llm_config.get('model', 'gpt-4')
        
        # Extract all possible LLM parameters from config
        llm_params = {{
            'model': model,
            'temperature': llm_config.get('temperature', 0.7),
            'max_tokens': llm_config.get('max_tokens'),
            'top_p': llm_config.get('top_p'),
            'frequency_penalty': llm_config.get('frequency_penalty'),
            'presence_penalty': llm_config.get('presence_penalty'),
            'stop': llm_config.get('stop'),
            'timeout': llm_config.get('timeout'),
            'max_retries': llm_config.get('max_retries'),
            'api_key': llm_config.get('api_key'),
            'base_url': llm_config.get('base_url'),
            'api_version': llm_config.get('api_version'),
            'organization': llm_config.get('organization')
        }}
        
        # Remove None values to avoid passing them to LLM constructor
        llm_params = {{k: v for k, v in llm_params.items() if v is not None}}
        
        # Check if environment variables override the config
        env_provider = os.getenv('CREWMASTER_LLM_PROVIDER')
        env_model = os.getenv('CREWMASTER_LLM_MODEL')
        env_api_key = os.getenv('CREWMASTER_LLM_API_KEY')
        env_base_url = os.getenv('CREWMASTER_LLM_BASE_URL')
        
        # If environment variables are set, use them (highest priority)
        if env_provider and env_model:
            llm_params['model'] = env_model
            if env_provider == 'custom' and env_api_key and env_base_url:
                llm_params['api_key'] = env_api_key
                llm_params['base_url'] = env_base_url
            return LLM(**llm_params)
        
        # Otherwise use agent-specific config
        return LLM(**llm_params)
    
    def _create_agents(self) -> dict:
        """Create agents from configuration."""
        agents = {{}}
        
        for agent_name, agent_config in self.agents_config.items():
            # Get tools for this agent
            tools = get_tools_for_agent(agent_config.get('tools', []))
            
            # Use agent-specific LLM config if available, otherwise use default
            agent_llm_config = agent_config.get('llm', {{}})
            if agent_llm_config:
                agent_llm = self._create_agent_llm(agent_llm_config)
            else:
                agent_llm = self.llm
            
            agent = Agent(
                role=agent_config['role'],
                goal=agent_config['goal'],
                backstory=agent_config['backstory'],
                verbose=agent_config.get('verbose', True),
                allow_delegation=agent_config.get('allow_delegation', False),
                max_iter=agent_config.get('max_iter', 5),
                tools=tools,
                llm=agent_llm  # Use agent-specific LLM
            )
            
            agents[agent_name] = agent
        
        return agents
    
    def _create_tasks(self) -> dict:
        """Create tasks from configuration."""
        tasks = {{}}
        
        for task_name, task_config in self.tasks_config.items():
            # Get the agent for this task
            agent_name = task_config['agent']
            agent = self.agents[agent_name]
            
            # Handle task context (dependencies)
            context_tasks = []
            if 'context' in task_config:
                for context_task_name in task_config['context']:
                    if context_task_name in tasks:
                        context_tasks.append(tasks[context_task_name])
            
            task = Task(
                description=task_config['description'],
                expected_output=task_config['expected_output'],
                agent=agent,
                context=context_tasks if context_tasks else None
            )
            
            tasks[task_name] = task
        
        return tasks
    
    def run(self, task_input: str = None) -> str:
        """Run the crew with optional task input."""
        try:
            # If task input is provided, update the main task description
            if task_input and 'main_task' in self.tasks:
                original_desc = self.tasks_config['main_task']['description']
                enhanced_desc = f"{{original_desc}}\\n\\nSpecific task input: {{task_input}}"
                self.tasks['main_task'].description = enhanced_desc
            
            # Execute the crew
            result = self.crew.kickoff()
            return str(result)
            
        except Exception as e:
            return f"Crew execution failed: {{str(e)}}"
    
    def get_crew_info(self) -> dict:
        """Get information about the crew configuration."""
        return {{
            'name': '{spec.name}',
            'description': '{spec.description or "No description provided"}',
            'agents': list(self.agents.keys()),
            'tasks': list(self.tasks.keys()),
            'process_type': '{spec.process_type}'
        }}
'''
        
        crew_path_file = crew_path / "src" / spec.name / "crew.py"
        with open(crew_path_file, 'w') as f:
            f.write(content)
    
    def _generate_tools_py(self, crew_path: Path, spec: CrewSpec):
        """Generate tools/custom_tools.py."""
        # Collect all unique tools from agents
        all_tools = set()
        for agent in spec.agents:
            all_tools.update(agent.required_tools)
        
        content = f'''"""
Custom tools for {spec.name} crew.

This module handles tool initialization and provides tools to agents.
"""

from typing import List, Any


def get_tools_for_agent(tool_names: List[str]) -> List[Any]:
    """Get actual CrewAI tools for an agent based on tool names."""
    try:
        from crewai_tools import (
            SerperDevTool, FileReadTool, ScrapeWebsiteTool, GithubSearchTool,
            YoutubeVideoSearchTool, YoutubeChannelSearchTool, CodeInterpreterTool,
            PDFSearchTool, DOCXSearchTool, CSVSearchTool, JSONSearchTool,
            XMLSearchTool, TXTSearchTool, MDXSearchTool, DirectoryReadTool,
            DirectorySearchTool, WebsiteSearchTool
        )
    except ImportError:
        print("Warning: crewai-tools not installed, using mock tools")
        return []
    
    tools = []
    
    for tool_name in tool_names:
        try:
            if tool_name == 'SerperDevTool':
                tools.append(SerperDevTool())
            elif tool_name == 'FileReadTool':
                tools.append(FileReadTool())
            elif tool_name == 'ScrapeWebsiteTool':
                tools.append(ScrapeWebsiteTool())
            elif tool_name == 'GithubSearchTool':
                tools.append(GithubSearchTool())
            elif tool_name == 'YoutubeVideoSearchTool':
                tools.append(YoutubeVideoSearchTool())
            elif tool_name == 'YoutubeChannelSearchTool':
                tools.append(YoutubeChannelSearchTool())
            elif tool_name == 'CodeInterpreterTool':
                tools.append(CodeInterpreterTool())
            elif tool_name == 'PDFSearchTool':
                tools.append(PDFSearchTool())
            elif tool_name == 'DOCXSearchTool':
                tools.append(DOCXSearchTool())
            elif tool_name == 'CSVSearchTool':
                tools.append(CSVSearchTool())
            elif tool_name == 'JSONSearchTool':
                tools.append(JSONSearchTool())
            elif tool_name == 'XMLSearchTool':
                tools.append(XMLSearchTool())
            elif tool_name == 'TXTSearchTool':
                tools.append(TXTSearchTool())
            elif tool_name == 'MDXSearchTool':
                tools.append(MDXSearchTool())
            elif tool_name == 'DirectoryReadTool':
                tools.append(DirectoryReadTool())
            elif tool_name == 'DirectorySearchTool':
                tools.append(DirectorySearchTool())
            elif tool_name == 'WebsiteSearchTool':
                tools.append(WebsiteSearchTool())
            # Add fallback for any unrecognized tools
            else:
                print(f"Warning: Unknown tool '{{tool_name}}', using SerperDevTool as fallback")
                if SerperDevTool not in [type(t) for t in tools]:
                    tools.append(SerperDevTool())
        except ImportError as e:
            print(f"Warning: Could not import {{tool_name}}, using SerperDevTool as fallback: {{e}}")
            if SerperDevTool not in [type(t) for t in tools]:
                tools.append(SerperDevTool())
        except Exception as e:
            print(f"Warning: Could not instantiate {{tool_name}}: {{e}}")
    
    # Ensure we have at least one tool
    if not tools:
        try:
            tools.append(SerperDevTool())
        except:
            pass
    
    return tools


# Note: Using actual CrewAI tools instead of custom implementations
# Tools are imported and instantiated directly from crewai_tools package
'''
        
        tools_path = crew_path / "src" / spec.name / "tools" / "custom_tools.py"
        with open(tools_path, 'w') as f:
            f.write(content)
    
    def _generate_tool_implementations(self, tools: set) -> str:
        """Generate implementations for specific tools."""
        # No custom tool implementations needed - using actual CrewAI tools
        return ""
    
    def _generate_requirements_txt(self, crew_path: Path, spec: CrewSpec):
        """Generate requirements.txt file."""
        requirements = [
            "crewai>=0.80.0",
            "crewai-tools>=0.12.0",
            "pydantic>=2.0.0",
            "pyyaml>=6.0.0",
        ]
        
        # Add tool-specific requirements
        all_tools = set()
        for agent in spec.agents:
            all_tools.update(agent.required_tools)
        
        # Check for specific CrewAI tools that might need additional dependencies
        if 'SerperDevTool' in all_tools:
            requirements.extend([
                # SerperDevTool requires SERPER_API_KEY environment variable
                # No additional package needed - included in crewai-tools
            ])
        
        if 'CodeInterpreterTool' in all_tools:
            requirements.extend([
                "matplotlib>=3.5.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0"
            ])
        
        req_path = crew_path / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write('\n'.join(requirements))
    
    def _generate_run_script(self, crew_path: Path, spec: CrewSpec):
        """Generate run.sh script."""
        content = f'''#!/bin/bash
# Run script for {spec.name} crew

set -e

echo "Starting {spec.name} crew..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the crew
echo "Running crew with arguments: $@"
cd src
python -m {spec.name}.main "$@"
'''
        
        script_path = crew_path / "run.sh"
        with open(script_path, 'w') as f:
            f.write(content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
    
    def _generate_readme(self, crew_path: Path, spec: CrewSpec):
        """Generate README.md file."""
        agents_list = '\n'.join([f"- **{agent.name}**: {agent.role}" for agent in spec.agents])
        tools_list = '\n'.join([f"- {tool}" for tool in set().union(*[agent.required_tools for agent in spec.agents])])
        
        content = f'''# {spec.name}

{spec.description or f"A CrewAI project for {spec.task}"}

## Overview

This crew consists of multiple AI agents working together to accomplish complex tasks.

**Task**: {spec.task}

**Expected Output**: {spec.expected_output}

## Agents

{agents_list}

## Tools Used

{tools_list}

## Usage

### Quick Start

```bash
# Make the run script executable (if not already)
chmod +x run.sh

# Run the crew
./run.sh "Your task description here"
```

### Manual Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the crew
cd src
python -m {spec.name}.main "Your task description here"
```

## Configuration

- **Agents**: Configure in `config/agents.yaml`
- **Tasks**: Configure in `config/tasks.yaml`
- **Tools**: Customize in `src/{spec.name}/tools/custom_tools.py`

## Generated by CrewMaster

This project was generated by CrewMaster on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.

For more information about CrewAI, visit: https://docs.crewai.com/
'''
        
        readme_path = crew_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(content)
    
    def _generate_init_files(self, crew_path: Path, crew_name: str):
        """Generate __init__.py files."""
        # Main package __init__.py
        main_init = crew_path / "src" / crew_name / "__init__.py"
        with open(main_init, 'w') as f:
            f.write(f'''"""
{crew_name} - CrewAI Project

Generated by CrewMaster.
"""

from .crew import {self._to_class_name(crew_name)}

__version__ = "1.0.0"
__all__ = ["{self._to_class_name(crew_name)}"]
''')
        
        # Tools package __init__.py
        tools_init = crew_path / "src" / crew_name / "tools" / "__init__.py"
        with open(tools_init, 'w') as f:
            f.write('"""Custom tools package."""\n')
    
    def _to_class_name(self, name: str) -> str:
        """Convert crew name to Python class name."""
        # Remove special characters and convert to PascalCase
        clean_name = ''.join(c for c in name if c.isalnum() or c == '_')
        words = clean_name.replace('_', ' ').split()
        return ''.join(word.capitalize() for word in words) + 'Crew'
    
    def export_crew_as_zip(self, crew_name: str, output_path: str = None) -> str:
        """Export a crew as a ZIP file."""
        import zipfile
        from datetime import datetime
        
        crew_path = self.crews_base_path / crew_name
        if not crew_path.exists():
            raise ValueError(f"Crew '{crew_name}' does not exist")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{crew_name}_{timestamp}.zip"
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in crew_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(crew_path.parent)
                    zipf.write(file_path, arcname)
        
        return output_path
    
    def list_generated_crews(self) -> List[Dict[str, Any]]:
        """List all generated crews."""
        crews = []
        
        for crew_dir in self.crews_base_path.iterdir():
            if crew_dir.is_dir():
                # Try to read crew info from config files
                config_path = crew_dir / "config"
                info = {
                    'name': crew_dir.name,
                    'path': str(crew_dir),
                    'created_at': datetime.fromtimestamp(crew_dir.stat().st_ctime),
                    'agents': [],
                    'tasks': []
                }
                
                try:
                    # Read agents config
                    agents_file = config_path / "agents.yaml"
                    if agents_file.exists():
                        with open(agents_file, 'r') as f:
                            agents_config = yaml.safe_load(f)
                            info['agents'] = list(agents_config.keys())
                    
                    # Read tasks config
                    tasks_file = config_path / "tasks.yaml"
                    if tasks_file.exists():
                        with open(tasks_file, 'r') as f:
                            tasks_config = yaml.safe_load(f)
                            info['tasks'] = list(tasks_config.keys())
                
                except Exception:
                    pass  # Skip if config files are corrupted
                
                crews.append(info)
        
        return crews