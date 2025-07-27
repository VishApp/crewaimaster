"""
CrewMaster CLI - Command line interface for managing multi-agent crews.
"""

import warnings
import os
# Suppress common deprecation warnings from dependencies
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*langchain.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Pydantic.*deprecated.*") 
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince20.*")
warnings.filterwarnings("ignore", message=".*event loop.*")
warnings.filterwarnings("ignore", message=".*extra keyword arguments.*")
warnings.filterwarnings("ignore", message=".*Field.*deprecated.*")
# Set environment variable to suppress additional warnings
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

import typer
import json
import yaml
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .core.master_agent import MasterAgent
from .core.master_agent_crew import MasterAgentCrew
from .core.config import Config
from .core.code_generator import generate_crew_code_package

app = typer.Typer(
    name="crewmaster",
    help="""[bold cyan]CrewMaster: Build intelligent multi-agent systems using CrewAI[/bold cyan]

[green]🎯 Quick Examples:[/green]
  [cyan]crewmaster create[/cyan] "research AI trends and write a report"          # CREATE crew with AI orchestration
  [cyan]crewmaster run[/cyan] my_research_crew --input "focus on 2024 data"       # EXECUTE with context
  [cyan]crewmaster list crews[/cyan]                                              # VIEW all created crews

[green]📚 For comprehensive help:[/green]
  [cyan]crewmaster help-extended[/cyan]   # Complete documentation with examples

[green]🔧 Common Workflows:[/green]
  • Create → List → Run → Performance
  • Create → Inspect → Edit → Run  
  • Export → Share → Import

[dim]💡 Tools are automatically assigned based on task analysis[/dim]
[dim]💡 All crews and agents use memorable names instead of complex IDs[/dim]
""",
    rich_markup_mode="rich"
)

console = Console()

@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    """Main callback that shows banner when no command is provided."""
    if ctx.invoked_subcommand is None:
        display_banner()
        
        console.print(f"\n[bold yellow]💡 Quick Help[/bold yellow]")
        console.print("=" * 60)
        console.print("  [cyan]crewmaster stats[/cyan]          - Show System Statistics")
        console.print("  [cyan]crewmaster --help[/cyan]          - Full command list")
        console.print("  [cyan]crewmaster help-extended[/cyan]   - Complete guide")
        
        console.print(f"\n[bold yellow]🎯 Getting Started Guide[/bold yellow]")
        console.print("=" * 60)
        
        console.print("\n[bold green]Step 1:[/bold green] Create your first crew with AI orchestration")
        console.print("  [cyan]crewmaster create \"analyze competitor pricing and create a report\"[/cyan]")
        
        console.print("\n[bold green]Step 2:[/bold green] List and inspect your crews")
        console.print("  [cyan]crewmaster list crews[/cyan]")
        console.print("  [cyan]crewmaster inspect my_analysis_crew[/cyan]")
        
        console.print("\n[bold green]Step 3:[/bold green] Run your crew (requires OpenAI API key)")
        console.print("  [cyan]export OPENAI_API_KEY=\"your-key\"[/cyan]")
        console.print("  [cyan]crewmaster run my_analysis_crew[/cyan]")
        
        console.print("\n[bold green]Step 4:[/bold green] Monitor and optimize")
        console.print("  [cyan]crewmaster performance my_analysis_crew --history[/cyan]")
        console.print("  [cyan]crewmaster stats --detailed[/cyan]")
        
        console.print("\n[dim]💡 Tips:[/dim]")
        console.print("[dim]  • AI orchestration is now the default - smarter task analysis and agent design[/dim]")
        console.print("[dim]  • Tools are automatically assigned based on your task description[/dim]")
        console.print("[dim]  • Use memorable names - no complex IDs required[/dim]")
        console.print("[dim]  • Export/import crews for backup and sharing[/dim]")
        console.print("[dim]  • Try 'crewmaster help-extended' for comprehensive documentation[/dim]")

def _analyze_modification_request(modification: str, target_type: str) -> Optional[Dict[str, Any]]:
    """Analyze natural language modification request and create a plan."""
    import re
    
    modification_lower = modification.lower()
    plan = {"steps": [], "actions": []}
    
    # Tool addition patterns
    if re.search(r'\b(add|include|give|assign).*(tool|capability)', modification_lower):
        # Extract tool names
        tool_patterns = [
            r'\b(web[_\s]?search|search)\b',
            r'\b(web[_\s]?scraping|scraping|scrape)\b', 
            r'\b(document[_\s]?search|pdf|docx|csv)\b',
            r'\b(github[_\s]?search|git)\b',
            r'\b(youtube[_\s]?search|video)\b',
            r'\b(vision|dall[-_]?e|image)\b',
            r'\b(database[_\s]?search|sql|postgres)\b',
            r'\b(browser[_\s]?automation|browser)\b',
            r'\b(code[_\s]?execution|python|execute)\b',
            r'\b(file[_\s]?operations|file)\b',
            r'\b(data[_\s]?processing|data)\b'
        ]
        
        tool_mapping = {
            r'\b(web[_\s]?search|search)\b': 'web_search',
            r'\b(web[_\s]?scraping|scraping|scrape)\b': 'web_scraping',
            r'\b(document[_\s]?search|pdf|docx|csv)\b': 'document_search',
            r'\b(github[_\s]?search|git)\b': 'github_search', 
            r'\b(youtube[_\s]?search|video)\b': 'youtube_search',
            r'\b(vision|dall[-_]?e|image)\b': 'vision',
            r'\b(database[_\s]?search|sql|postgres)\b': 'database_search',
            r'\b(browser[_\s]?automation|browser)\b': 'browser_automation',
            r'\b(code[_\s]?execution|python|execute)\b': 'code_execution',
            r'\b(file[_\s]?operations|file)\b': 'file_operations',
            r'\b(data[_\s]?processing|data)\b': 'data_processing'
        }
        
        tools_to_add = []
        for pattern, tool_name in tool_mapping.items():
            if re.search(pattern, modification_lower):
                tools_to_add.append(tool_name)
        
        if tools_to_add:
            plan["steps"].append(f"Add tools: {', '.join(tools_to_add)}")
            plan["actions"].append({"type": "add_tools", "tools": tools_to_add})
    
    # Property update patterns
    if re.search(r'\b(change|update|modify|set).*(goal|role|task|description)', modification_lower):
        # Extract what property to change
        if 'goal' in modification_lower:
            # Try to extract new goal
            goal_match = re.search(r'goal.*(to|is|should be)\s*["\']?([^"\']+)["\']?', modification_lower)
            if goal_match:
                new_goal = goal_match.group(2).strip()
                plan["steps"].append(f"Update goal to: {new_goal}")
                plan["actions"].append({"type": "update_property", "property": "goal", "value": new_goal})
        
        if 'task' in modification_lower:
            task_match = re.search(r'task.*(to|is|should be)\s*["\']?([^"\']+)["\']?', modification_lower)
            if task_match:
                new_task = task_match.group(2).strip()
                plan["steps"].append(f"Update task to: {new_task}")
                plan["actions"].append({"type": "update_property", "property": "task", "value": new_task})
        
        if 'description' in modification_lower:
            desc_match = re.search(r'description.*(to|is|should be)\s*["\']?([^"\']+)["\']?', modification_lower)
            if desc_match:
                new_desc = desc_match.group(2).strip()
                plan["steps"].append(f"Update description to: {new_desc}")
                plan["actions"].append({"type": "update_property", "property": "description", "value": new_desc})
    
    # Agent addition patterns (for crews)
    if target_type == "crew" and re.search(r'\b(add|create|include).*(agent)', modification_lower):
        # Try to extract agent role and capabilities
        role_patterns = {
            r'\b(analyst|data\s+analyst|analyzer)\b': 'analyst',
            r'\b(researcher|research)\b': 'researcher', 
            r'\b(writer|content\s+creator)\b': 'writer',
            r'\b(developer|programmer|coder)\b': 'developer',
            r'\b(reviewer|quality\s+assurance|qa)\b': 'reviewer'
        }
        
        for pattern, role in role_patterns.items():
            if re.search(pattern, modification_lower):
                plan["steps"].append(f"Add new {role} agent")
                plan["actions"].append({"type": "add_agent", "role": role, "description": modification})
                break
    
    # Fallback: If no specific patterns matched, treat it as a goal/task update
    if not plan["steps"]:
        # Check if it looks like a task description (contains action words)
        action_words = ['fetch', 'get', 'find', 'search', 'analyze', 'create', 'build', 'write', 'research', 'collect', 'gather', 'retrieve', 'process', 'generate']
        if any(word in modification_lower for word in action_words):
            if target_type == "agent":
                plan["steps"].append(f"Update agent goal to: {modification}")
                plan["actions"].append({"type": "update_property", "property": "goal", "value": modification})
            elif target_type == "crew":
                plan["steps"].append(f"Update crew task to: {modification}")
                plan["actions"].append({"type": "update_property", "property": "task", "value": modification})
    
    return plan if plan["steps"] else None

def _apply_modifications(master_agent, target_type: str, target_name: str, plan: Dict[str, Any]) -> bool:
    """Apply the modification plan to the target crew or agent."""
    try:
        success = True
        
        for action in plan["actions"]:
            if action["type"] == "add_tools":
                if target_type == "agent":
                    # Find and update agent
                    for crew_name, crew_model in master_agent.crew_designer._crews_cache.items():
                        for agent in crew_model.agents:
                            if agent.name == target_name:
                                current_tools = getattr(agent, 'required_tools', [])
                                new_tools = list(set(current_tools + action["tools"]))
                                agent.required_tools = new_tools
                                master_agent.crew_designer._crews_cache[crew_name] = crew_model
                                master_agent.crew_designer._save_cache()
                                break
            
            elif action["type"] == "update_property":
                if target_type == "crew":
                    crew_model = master_agent.crew_designer.get_crew_from_cache(target_name)
                    if crew_model and hasattr(crew_model, action["property"]):
                        setattr(crew_model, action["property"], action["value"])
                        master_agent.crew_designer._crews_cache[target_name] = crew_model
                        master_agent.crew_designer._save_cache()
                
                elif target_type == "agent":
                    for crew_name, crew_model in master_agent.crew_designer._crews_cache.items():
                        for agent in crew_model.agents:
                            if agent.name == target_name and hasattr(agent, action["property"]):
                                setattr(agent, action["property"], action["value"])
                                master_agent.crew_designer._crews_cache[crew_name] = crew_model
                                master_agent.crew_designer._save_cache()
                                break
            
            elif action["type"] == "add_agent":
                if target_type == "crew":
                    crew_model = master_agent.crew_designer.get_crew_from_cache(target_name)
                    if crew_model:
                        # Create new agent
                        from ..database.models import AgentModel
                        new_agent = AgentModel()
                        new_agent.id = f"{action['role']}_agent_{len(crew_model.agents) + 1}"
                        new_agent.name = f"{action['role']}_agent_{len(crew_model.agents) + 1}"
                        new_agent.role = action["role"]
                        new_agent.goal = f"Provide {action['role']} expertise for the crew"
                        new_agent.backstory = f"You are a skilled {action['role']} with relevant experience."
                        new_agent.required_tools = ["web_search", "file_operations"]  # Default tools
                        
                        crew_model.agents.append(new_agent)
                        master_agent.crew_designer._crews_cache[target_name] = crew_model
                        master_agent.crew_designer._save_cache()
            
            elif action["type"] == "recreate_agents_for_task":
                if target_type == "crew":
                    crew_model = master_agent.crew_designer.get_crew_from_cache(target_name)
                    if crew_model:
                        # Recreate agents appropriate for the new task
                        new_task = action["value"]
                        old_agent_count = len(crew_model.agents)
                        console.print(f"[dim]🔄 Recreating {old_agent_count} agents for task: {new_task}[/dim]")
                        
                        new_agents = _create_agents_for_task(new_task, old_agent_count)
                        console.print(f"[dim]✅ Created {len(new_agents)} new agents: {[a.name for a in new_agents]}[/dim]")
                        
                        # Replace old agents with new ones
                        crew_model.agents = new_agents
                        
                        master_agent.crew_designer._crews_cache[target_name] = crew_model
                        master_agent.crew_designer._save_cache()
                        console.print(f"[dim]💾 Saved updated crew to cache[/dim]")
        
        return success
    
    except Exception as e:
        console.print(f"Error applying modifications: {e}")
        return False

def _generate_contextual_backstory(agent_name: str, agent_role: str, new_task: str) -> str:
    """Generate a contextually relevant backstory for an agent based on the new task."""
    task_lower = new_task.lower()
    
    # Determine the domain/context of the task
    if any(word in task_lower for word in ['news', 'article', 'journalism', 'media']):
        domain = "journalism and media analysis"
        experience = "has extensive experience in news gathering, fact-checking, and media analysis"
        specialization = "specializes in identifying credible sources and extracting key information from news articles"
    
    elif any(word in task_lower for word in ['weather', 'climate', 'meteorology', 'forecast']):
        domain = "meteorology and weather analysis"
        experience = "has a strong background in weather data analysis and meteorological research"
        specialization = "specializes in interpreting weather patterns and climate data"
    
    elif any(word in task_lower for word in ['github', 'repository', 'code', 'software', 'programming']):
        domain = "software development and open-source projects"
        experience = "has over 5 years of experience in software development and data science"
        specialization = "specializes in analyzing code repositories and open-source projects"
    
    elif any(word in task_lower for word in ['cryptocurrency', 'crypto', 'blockchain', 'bitcoin']):
        domain = "cryptocurrency and blockchain analysis"
        experience = "has deep expertise in cryptocurrency markets and blockchain technology"
        specialization = "specializes in financial data analysis and market trend identification"
    
    elif any(word in task_lower for word in ['stock', 'finance', 'market', 'trading', 'investment']):
        domain = "financial markets and investment analysis"
        experience = "has extensive experience in financial analysis and market research"
        specialization = "specializes in market trend analysis and investment insights"
    
    elif any(word in task_lower for word in ['football', 'soccer', 'sports', 'athletic', 'games']):
        domain = "sports analysis and journalism"
        experience = "has extensive experience in sports journalism and athletic performance analysis"
        specialization = "specializes in sports statistics, game analysis, and athletic news coverage"
    
    elif any(word in task_lower for word in ['research', 'study', 'analysis', 'investigate']):
        domain = "research and data analysis"
        experience = "has a strong background in research methodology and data analysis"
        specialization = "specializes in comprehensive research and analytical thinking"
    
    else:
        # Generic fallback
        domain = "data analysis and research"
        experience = "has extensive experience in data analysis and research"
        specialization = "specializes in thorough analysis and information gathering"
    
    # Generate the backstory
    backstory = f"{agent_name}, a seasoned {agent_role}, {experience}. Working in {domain}, they {specialization}. With a collaborative approach and attention to detail, they excel at delivering comprehensive and accurate results."
    
    return backstory

def _create_agents_for_task(task: str, agent_count: int) -> list:
    """Create appropriate agents for a given task."""
    from .database.models import AgentModel
    task_lower = task.lower()
    agents = []
    print(f"🔧 DEBUG: Creating {agent_count} agents for task: '{task}'")
    
    # Define agent templates based on task type
    agent_templates = []
    
    if any(word in task_lower for word in ['news', 'article', 'journalism', 'media']):
        agent_templates = [
            {
                "name": "NewsGatherer",
                "role": "news_researcher", 
                "goal": f"Research and gather comprehensive news information for: {task}",
                "backstory": "An experienced news researcher with expertise in identifying credible sources, fact-checking information, and gathering comprehensive news coverage from multiple reliable outlets.",
                "tools": ["web_search", "web_scraping"]
            },
            {
                "name": "ContentAnalyzer", 
                "role": "content_analyst",
                "goal": f"Analyze and organize news content for: {task}",
                "backstory": "A skilled content analyst specialized in processing news articles, extracting key information, and organizing content in a clear and structured manner.",
                "tools": ["data_processing", "file_operations"]
            }
        ]
    
    elif any(word in task_lower for word in ['github', 'repository', 'code', 'project']):
        agent_templates = [
            {
                "name": "CodeExplorer",
                "role": "repository_analyst",
                "goal": f"Explore and analyze GitHub repositories for: {task}",
                "backstory": "A seasoned developer and repository analyst with deep expertise in code analysis, project evaluation, and open-source software assessment.",
                "tools": ["github_search", "web_search"]
            },
            {
                "name": "ProjectEvaluator",
                "role": "project_evaluator", 
                "goal": f"Evaluate and rank projects based on criteria for: {task}",
                "backstory": "An expert project evaluator specialized in assessing software projects, analyzing code quality, community engagement, and project viability.",
                "tools": ["data_processing", "web_scraping"]
            }
        ]
    
    elif any(word in task_lower for word in ['cryptocurrency', 'crypto', 'blockchain', 'bitcoin']):
        agent_templates = [
            {
                "name": "CryptoAnalyst",
                "role": "crypto_market_analyst",
                "goal": f"Analyze cryptocurrency markets and trends for: {task}",
                "backstory": "A specialized cryptocurrency analyst with deep knowledge of blockchain technology, market dynamics, and digital asset evaluation.",
                "tools": ["web_search", "data_processing"]
            },
            {
                "name": "MarketResearcher",
                "role": "market_researcher",
                "goal": f"Research market data and trends for: {task}", 
                "backstory": "An experienced market researcher focused on cryptocurrency and blockchain markets, skilled at identifying trends and analyzing market sentiment.",
                "tools": ["web_scraping", "data_processing"]
            }
        ]
    
    elif any(word in task_lower for word in ['weather', 'climate', 'meteorology']):
        agent_templates = [
            {
                "name": "WeatherAnalyst",
                "role": "meteorological_analyst",
                "goal": f"Analyze weather data and patterns for: {task}",
                "backstory": "A meteorological analyst with expertise in weather data interpretation, climate analysis, and atmospheric science.",
                "tools": ["web_search", "data_processing"]
            }
        ]
    
    elif any(word in task_lower for word in ['football', 'soccer', 'sports', 'athletic', 'games']):
        print(f"🔧 DEBUG: Detected sports/football task, creating sports agents")
        agent_templates = [
            {
                "name": "SportsReporter",
                "role": "sports_journalist",
                "goal": f"Research and report on sports news for: {task}",
                "backstory": "An experienced sports journalist with deep knowledge of athletic competitions, player statistics, and sports industry trends. Specialized in delivering timely and accurate sports coverage.",
                "tools": ["web_search", "web_scraping"]
            },
            {
                "name": "StatsAnalyzer",
                "role": "sports_analyst",
                "goal": f"Analyze sports statistics and performance data for: {task}",
                "backstory": "A sports data analyst with expertise in performance metrics, game analysis, and statistical trends in athletics. Skilled at interpreting complex sports data.",
                "tools": ["data_processing", "web_search"]
            }
        ]
    
    else:
        # Generic research agents
        agent_templates = [
            {
                "name": "ResearchSpecialist",
                "role": "research_analyst",
                "goal": f"Conduct comprehensive research for: {task}",
                "backstory": "A versatile research analyst with broad expertise in information gathering, data analysis, and comprehensive research across multiple domains.",
                "tools": ["web_search", "web_scraping"]
            },
            {
                "name": "DataProcessor",
                "role": "data_analyst", 
                "goal": f"Process and analyze data for: {task}",
                "backstory": "A skilled data analyst specialized in processing, organizing, and analyzing information to deliver clear and actionable insights.",
                "tools": ["data_processing", "file_operations"]
            }
        ]
    
    # Create the requested number of agents
    for i in range(min(agent_count, len(agent_templates))):
        template = agent_templates[i]
        agent = AgentModel()
        agent.id = f"{template['name'].lower()}_{i+1}"
        agent.name = f"{template['name']}_{i+1}"
        agent.role = template["role"]
        agent.goal = template["goal"]
        agent.backstory = template["backstory"]
        agent.required_tools = template["tools"]
        agents.append(agent)
    
    # If we need more agents than templates, create generic ones
    while len(agents) < agent_count:
        i = len(agents)
        agent = AgentModel()
        agent.id = f"specialist_{i+1}"
        agent.name = f"TaskSpecialist_{i+1}"
        agent.role = "task_specialist"
        agent.goal = f"Provide specialized assistance for: {task}"
        agent.backstory = "A versatile specialist capable of adapting to various task requirements and providing focused expertise."
        agent.required_tools = ["web_search", "file_operations"]
        agents.append(agent)
    
    print(f"🔧 DEBUG: Returning {len(agents)} agents: {[(a.name, a.role) for a in agents]}")
    return agents

def display_banner():
    """Display CrewMaster banner with animated-style elements."""
    import time
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.align import Align
    from rich.text import Text
    # Clean, aligned banner
    banner = """[bold cyan]
                                                                        
[blink]╔═╗╦═╗╔═╗╦ ╦[/blink]  [bold blue]╔╦╗╔═╗╔═╗╔╦╗╔═╗╦═╗[/bold blue]                
[blink]║  ╠╦╝║╣ ║║║[/blink]  [bold blue]║║║╠═╣╚═╗ ║ ║╣ ╠╦╝[/bold blue]                
[blink]╚═╝╩╚═╚═╝╚╩╝[/blink]  [bold blue]╩ ╩╩ ╩╚═╝ ╩ ╚═╝╩╚═[/bold blue]                
                                                                        
[bright_green blink]🤖[/bright_green blink] [bright_green]Build intelligent multi-agent systems[/bright_green]           
[bright_blue blink]⚡[/bright_blue blink] [bright_blue]Powered by CrewAI & Rich terminal experience[/bright_blue]   
                                                                                                                         
"""

    console.print(banner)
    
    # Quick loading animation
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold green]Initializing CrewMaster..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("", total=50)
        for i in range(50):
            progress.update(task, advance=1)
            time.sleep(0.02)
    
    # Create compact status panels
    stats_panel = Panel(
        "[bold green]✨ System Status[/bold green]\n"
        "[bright_green]●[/bright_green] [cyan]Ready to create agents[/cyan]\n"
        "[bright_green]●[/bright_green] [cyan]Auto-tool assignment active[/cyan]\n"
        "[bright_green]●[/bright_green] [cyan]Rich terminal enabled[/cyan]",
        border_style="bright_green",
        width=32,
        title="[bold green]STATUS",
        title_align="center"
    )
    
    features_panel = Panel(
        "[bold blue]⚡ Key Features[/bold blue]\n"
        "[bright_blue]▸[/bright_blue] [yellow]Multi-agent crews[/yellow]\n"
        "[bright_blue]▸[/bright_blue] [yellow]Smart tool assignment[/yellow]\n"
        "[bright_blue]▸[/bright_blue] [yellow]Performance monitoring[/yellow]",
        border_style="bright_blue",
        width=32,
        title="[bold blue]FEATURES",
        title_align="center"
    )
    
    # Display side-by-side panels
    console.print(Columns([stats_panel, features_panel], padding=(0, 1)))
    
    # Simple centered footer
    footer = Text("CrewMaster - Where AI Agents Collaborate", style="bold dim cyan")
    console.print(Align.left(footer))
    
# @app.command()
# def analyze_task(
#     task: str = typer.Argument(..., help="Task description to analyze"),
#     verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
# ):
#     """ANALYZE ONLY: Use AI to analyze a task without actually creating a crew.
#     
#     This gives you a preview of what the AI orchestration would recommend:
#     - Task complexity assessment
#     - Required agent roles and tools  
#     - Estimated execution time
#     - Expected output format
#     
#     💡 Perfect for planning before creating expensive crews or understanding task requirements.
#     
#     Example: crewmaster analyze-task "research AI trends and create presentation"
#     """
#     console.print(f"\\n[bold blue]🔍 Analyzing task with AI:[/bold blue] {task}")
#     
#     try:
#         config = Config()
#         master_agent = MasterAgentCrew(config)
#         
#         analysis = master_agent.analyze_task_with_ai(task, verbose=verbose)
#         
#         if "error" in analysis:
#             console.print(f"[red]❌ Analysis failed: {analysis['error']}[/red]")
#             return
#         
#         # Display analysis results
#         console.print(f"\\n[bold green]📊 Task Analysis Results:[/bold green]")
#         
#         # Basic info
#         table = Table(title="Task Analysis")
#         table.add_column("Aspect", style="cyan")
#         table.add_column("Assessment", style="green")
#         
#         table.add_row("Complexity", analysis['complexity'].title())
#         table.add_row("Estimated Time", f"{analysis['estimated_time']} minutes")
#         table.add_row("Required Agents", str(analysis['agent_count']))
#         table.add_row("Process Type", analysis['process_type'].title())
#         
#         console.print(table)
#         
#         # Agents breakdown
#         if analysis['agents']:
#             console.print(f"\\n[bold cyan]👥 Recommended Agents:[/bold cyan]")
#             agent_table = Table()
#             agent_table.add_column("Agent", style="green")
#             agent_table.add_column("Role", style="blue")
#             agent_table.add_column("Tools", style="yellow")
#             
#             for agent in analysis['agents']:
#                 tools_str = ", ".join(agent['tools']) if agent['tools'] else "None"
#                 agent_table.add_row(agent['name'], agent['role'], tools_str)
#             
#             console.print(agent_table)
#         
#         # Expected output
#         console.print(f"\\n[bold magenta]🎯 Expected Output:[/bold magenta]")
#         console.print(f"   {analysis['expected_output']}")
#         
#         console.print(f"\\n[dim]💡 Next steps:[/dim]")
#         console.print(f"[dim]  • crewmaster create \\\"{task}\\\" (create this crew)[/dim]")
#         console.print(f"[dim]  • crewmaster run <crew_name> --input \\\"specific context\\\" (execute with context)[/dim]")
#         
#     except Exception as e:
#         console.print(f"\\n[bold red]❌ Error analyzing task:[/bold red] {str(e)}")
#         raise typer.Exit(1)

@app.command()
def create(
    task: str = typer.Argument(..., help="Description of the task to accomplish"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Optional name for the crew"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    reuse: bool = typer.Option(True, "--reuse/--no-reuse", help="Reuse existing agents when possible"),
    use_legacy: bool = typer.Option(False, "--legacy", help="Use legacy regex-based analysis instead of AI orchestration")
):
    """Create a new crew and agents for a given task.
    
    🤖 AI ORCHESTRATION (Default): Uses intelligent AI agents to analyze tasks, design agents, and orchestrate crew creation
    ⚙️  LEGACY MODE (--legacy): Uses simple regex patterns for task analysis (faster, no LLM required)
    
    Examples:
      crewmaster create "research competitors and write analysis report"  # AI orchestration (default)
      crewmaster create "build web app" --legacy                         # Legacy regex-based analysis
    """
    if verbose:
        display_banner()
    
    console.print(f"\n[bold green]🚀 Creating crew for task:[/bold green] {task}")
    
    try:
        config = Config()
        
        # Use AI orchestration by default, legacy only when explicitly requested
        if use_legacy:
            console.print("[bold yellow]⚙️  Using legacy regex-based crew creation[/bold yellow]")
            master_agent = MasterAgent(config)
            crew_result = master_agent.create_crew(
                task_description=task,
                crew_name=name,
                reuse_agents=reuse,
                verbose=verbose
            )
        else:
            console.print("[bold cyan]🤖 Using AI orchestration for intelligent crew creation[/bold cyan]")
            master_agent = MasterAgentCrew(config)
            crew_result = master_agent.create_crew(
                task_description=task,
                crew_name=name,
                reuse_agents=reuse,
                verbose=verbose,
                use_ai_orchestration=True
            )
        
        console.print(f"\n[bold green]✅ Created Crew:[/bold green] {crew_result.name}")
        console.print(f"[bold blue]🆔 Crew ID:[/bold blue] {crew_result.id}")
        
        # Display agents table
        table = Table(title="👥 Agents")
        table.add_column("Name", style="cyan")
        table.add_column("Role", style="green")
        table.add_column("Tools", style="yellow")
        table.add_column("Memory", style="magenta")
        
        for agent in crew_result.agents:
            # Get tools from the agent model
            agent_tools = getattr(agent, 'required_tools', []) if hasattr(agent, 'required_tools') else []
            tools_str = ", ".join(agent_tools) if agent_tools else ""
            memory_type = getattr(agent, 'memory_type', 'None')
            table.add_row(agent.name, agent.role, tools_str, memory_type)
        
        console.print(table)
        
        console.print(f"\n[dim]💡 Next steps:[/dim]")
        console.print(f"[dim]  • crewmaster run {crew_result.name}[/dim]")
        console.print(f"[dim]  • crewmaster run {crew_result.name} --input \"additional context\"[/dim]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error creating crew:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command("list")
def list_items(
    item_type: str = typer.Argument("crews", help="What to list: 'crews', 'agents', or 'tools'"),
    show_details: bool = typer.Option(False, "--details", "-d", help="Show detailed information")
):
    """List existing crews, agents, or available tools.
    
    Examples:
      crewmaster list crews             - Show all created crews
      crewmaster list agents            - Show all agents with their tools
      crewmaster list tools             - Show available tools and triggers
    """
    try:
        config = Config()
        master_agent = MasterAgent(config)
        
        if item_type not in ["crews", "agents", "crew", "agent", "tools", "tool"]:
            console.print("[red]❌ Item type must be 'crews', 'agents', or 'tools'[/red]")
            raise typer.Exit(1)
        
        # Get crews from cache since database isn't working
        try:
            crews = list(master_agent.crew_designer._crews_cache.values())
        except:
            crews = []
        
        if not crews and item_type in ["crews", "crew"]:
            console.print("[yellow]No crews found. Create one with 'crewmaster create'[/yellow]")
            return
        
        if item_type in ["crews", "crew"]:
            # List crews
            table = Table(title="📋 Crews")
            table.add_column("Name", style="green") 
            table.add_column("Task", style="blue")
            table.add_column("Agents", style="yellow")
            
            for crew in crews:
                agents_count = f"{len(crew.agents)}" if hasattr(crew, 'agents') else "0"
                
                table.add_row(
                    crew.name,
                    crew.task[:50] + "..." if len(crew.task) > 50 else crew.task,
                    agents_count
                )
            
            console.print(table)
            
        elif item_type in ["agents", "agent"]:
            # List all agents from all crews
            console.print("[bold blue]👥 All Agents:[/bold blue]")
            agent_table = Table()
            agent_table.add_column("Name", style="green")
            agent_table.add_column("Role", style="blue") 
            agent_table.add_column("Tools", style="yellow")
            agent_table.add_column("Crew", style="magenta")
            
            agent_count = 0
            for crew in crews:
                if hasattr(crew, 'agents') and crew.agents:
                    for agent in crew.agents:
                        tools_str = ", ".join(getattr(agent, 'required_tools', [])) if hasattr(agent, 'required_tools') else ""
                        
                        agent_table.add_row(
                            agent.name,
                            agent.role,
                            tools_str,
                            crew.name
                        )
                        agent_count += 1
            
            if agent_count == 0:
                console.print("[yellow]No agents found.[/yellow]")
            else:
                console.print(agent_table)
                
        elif item_type in ["tools", "tool"]:
            # List available tools from registry
            console.print("[bold blue]🛠️  Available Tools:[/bold blue]")
            
            # Get tools from the registry
            from .tools.registry import ToolRegistry
            tool_registry = ToolRegistry()
            
            tools_table = Table()
            tools_table.add_column("Tool Name", style="cyan")
            tools_table.add_column("Category", style="green")
            tools_table.add_column("Description", style="blue")
            
            # Group tools by category
            tools_by_category = {}
            for tool_name, tool_instance in tool_registry.tools.items():
                category = tool_instance.category
                if category not in tools_by_category:
                    tools_by_category[category] = []
                tools_by_category[category].append({
                    'name': tool_name,
                    'description': tool_instance.description
                })
            
            # Display tools grouped by category
            for category in sorted(tools_by_category.keys()):
                for tool in tools_by_category[category]:
                    tools_table.add_row(
                        tool['name'],
                        category.title(),
                        tool['description']
                    )
            
            console.print(tools_table)
            
            console.print(f"\n[dim]💡 Tools are automatically assigned to agents based on task analysis.[/dim]")
            console.print(f"[dim]📝 Use 'crewmaster create \"your task\"' and tools will be auto-selected.[/dim]")
            console.print(f"[dim]🔧 Total tools available: {len(tool_registry.tools)}[/dim]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error listing crews:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command()
def run(
    crew_name: str = typer.Argument(..., help="Name of the crew to run"),
    input_data: Optional[str] = typer.Option(None, "--input", "-i", help="Additional input data for the task"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """EXECUTE: Run an existing crew to actually perform the task.
    
    This executes the agents in the crew to complete the actual work.
    The crew must already exist (created with 'crewmaster create').
    
    Requirements:
    - OpenAI API key: export OPENAI_API_KEY="your-key"
    - Existing crew (use 'crewmaster list crews' to see available crews)
    
    Example: crewmaster run my_research_crew --input "focus on recent data"
    """
    console.print(f"\n[bold green]🏃 Running crew:[/bold green] {crew_name}")
    if input_data:
        console.print(f"[bold blue]📝 With additional context:[/bold blue] {input_data}")
    
    try:
        config = Config()
        master_agent = MasterAgent(config)
        
        result = master_agent.execute_crew(crew_name, input_data, verbose)
        
        console.print(f"\n[bold green]✅ Crew execution completed![/bold green]")
        console.print(Panel(result.output, title="📄 Result", border_style="green"))
        
        if verbose and result.logs:
            console.print("\n[bold cyan]📝 Execution Logs:[/bold cyan]")
            for log_entry in result.logs:
                if isinstance(log_entry, dict):
                    timestamp = log_entry.get('timestamp', 'Unknown')
                    message = log_entry.get('message', str(log_entry))
                    console.print(f"[dim]{timestamp}[/dim] {message}")
                else:
                    # Handle object with attributes
                    timestamp = getattr(log_entry, 'timestamp', 'Unknown')
                    message = getattr(log_entry, 'message', str(log_entry))
                    console.print(f"[dim]{timestamp}[/dim] {message}")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error running crew:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def inspect(
    crew_name: str = typer.Argument(..., help="Name of the crew to inspect")
):
    """Inspect detailed information about a crew."""
    try:
        config = Config()
        master_agent = MasterAgent(config)
        
        # Try to get crew from cache first (using name as key)
        crew = master_agent.crew_designer.get_crew_from_cache(crew_name)
        if not crew:
            crew = master_agent.get_crew(crew_name)
        
        if not crew:
            console.print(f"[red]Crew with name '{crew_name}' not found[/red]")
            return
        
        console.print(f"\n[bold green]🔍 Crew Details[/bold green]")
        console.print(f"[bold]Name:[/bold] {crew.name}")
        console.print(f"[bold]Task:[/bold] {crew.task}")
        console.print(f"[bold]Description:[/bold] {crew.description}")
        
        for i, agent in enumerate(crew.agents, 1):
            console.print(f"\n[bold cyan]👤 Agent {i}: {agent.name}[/bold cyan]")
            console.print(f"  [bold]Role:[/bold] {agent.role}")
            console.print(f"  [bold]Goal:[/bold] {agent.goal}")
            console.print(f"  [bold]Backstory:[/bold] {agent.backstory}")
            # Show required tools instead of agent.tools (which may not exist)
            if hasattr(agent, 'required_tools') and agent.required_tools:
                tools = ", ".join(agent.required_tools)
                console.print(f"  [bold]Tools:[/bold] {tools}")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error inspecting crew:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command()
def delete(
    crew_name: str = typer.Argument(..., help="Name of the crew to delete"),
    confirm: bool = typer.Option(False, "--confirm", "-y", help="Skip confirmation prompt")
):
    """Delete a crew and all its agents.
    
    Examples:
      crewmaster delete my_research_crew           (with confirmation prompt)
      crewmaster delete my_research_crew --confirm (skip confirmation)
    """
    if not confirm:
        if not typer.confirm(f"Are you sure you want to delete crew '{crew_name}'?"):
            console.print("Operation cancelled.")
            return
    
    try:
        config = Config()
        master_agent = MasterAgent(config)
        
        # Check if crew exists in cache (using name as key)
        if crew_name in master_agent.crew_designer._crews_cache:
            # Delete from cache
            del master_agent.crew_designer._crews_cache[crew_name]
            # Delete CrewAI instance if exists
            if crew_name in master_agent.crew_designer._crewai_instances:
                del master_agent.crew_designer._crewai_instances[crew_name]
            # Save cache
            master_agent.crew_designer._save_cache()
            console.print(f"[green]✅ Crew '{crew_name}' deleted successfully[/green]")
        else:
            console.print(f"[red]❌ Crew '{crew_name}' not found[/red]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error deleting crew:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command()
def edit(
    target_type: str = typer.Argument(..., help="Type to edit: 'crew' or 'agent'"),
    target_name: str = typer.Argument(..., help="Name of the crew or agent to edit"),
    field: str = typer.Option(None, "--field", "-f", help="Field to edit (name, goal, backstory, tools)"),
    value: str = typer.Option(None, "--value", "-v", help="New value for the field"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", "-i", help="Interactive editing mode")
):
    """Edit crew or agent properties.
    
    Examples:
      crewmaster edit crew my_research_crew --field name --value "Updated Research Crew"
      crewmaster edit agent researcher --field goal --value "New goal"  
      crewmaster edit crew my_research_crew     (interactive mode)
    """
    try:
        config = Config()
        master_agent = MasterAgent(config)
        
        if target_type not in ["crew", "agent"]:
            console.print("[red]❌ Target type must be 'crew' or 'agent'[/red]")
            raise typer.Exit(1)
        
        if target_type == "crew":
            crew_model = master_agent.crew_designer.get_crew_from_cache(target_name)
            if not crew_model:
                console.print(f"[red]❌ Crew with name '{target_name}' not found[/red]")
                raise typer.Exit(1)
            
            console.print(f"\n[bold blue]🔧 Editing Crew:[/bold blue] {crew_model.name}")
            
            if interactive and not (field and value):
                console.print("\n[dim]Available fields: name, task, description[/dim]")
                field = typer.prompt("Field to edit")
                current_value = getattr(crew_model, field, "Not set")
                console.print(f"Current {field}: {current_value}")
                value = typer.prompt(f"New {field}")
            
            if field and value:
                if hasattr(crew_model, field):
                    # Special handling for name changes (need to update cache key)
                    if field == "name":
                        old_name = crew_model.name
                        crew_model.name = value
                        crew_model.id = value  # Keep ID in sync with name
                        # Remove from old key and add to new key
                        del master_agent.crew_designer._crews_cache[old_name]
                        master_agent.crew_designer._crews_cache[value] = crew_model
                        # Update CrewAI instance key if exists
                        if old_name in master_agent.crew_designer._crewai_instances:
                            master_agent.crew_designer._crewai_instances[value] = master_agent.crew_designer._crewai_instances[old_name]
                            del master_agent.crew_designer._crewai_instances[old_name]
                    else:
                        setattr(crew_model, field, value)
                        # Update in cache
                        master_agent.crew_designer._crews_cache[target_name] = crew_model
                    
                    master_agent.crew_designer._save_cache()
                    console.print(f"[green]✅ Updated {field} to: {value}[/green]")
                else:
                    console.print(f"[red]❌ Field '{field}' not found[/red]")
            
        elif target_type == "agent":
            # Find agent in all crews by name
            agent_found = False
            for crew_name, crew_model in master_agent.crew_designer._crews_cache.items():
                for agent in crew_model.agents:
                    if agent.name == target_name:
                        agent_found = True
                        console.print(f"\n[bold blue]🔧 Editing Agent:[/bold blue] {agent.name}")
                        console.print(f"[bold blue]👥 Part of Crew:[/bold blue] {crew_model.name}")
                        
                        if interactive and not (field and value):
                            console.print("\n[dim]Available fields: name, role, goal, backstory, required_tools[/dim]")
                            field = typer.prompt("Field to edit")
                            current_value = getattr(agent, field, "Not set")
                            console.print(f"Current {field}: {current_value}")
                            value = typer.prompt(f"New {field}")
                        
                        if field and value:
                            if hasattr(agent, field):
                                if field == "required_tools":
                                    # Parse comma-separated tools
                                    value = [tool.strip() for tool in value.split(",")]
                                elif field == "name":
                                    # Update agent ID to match name
                                    agent.id = value
                                setattr(agent, field, value)
                                # Update in cache
                                master_agent.crew_designer._crews_cache[crew_name] = crew_model
                                master_agent.crew_designer._save_cache()
                                console.print(f"[green]✅ Updated {field} to: {value}[/green]")
                            else:
                                console.print(f"[red]❌ Field '{field}' not found[/red]")
                        break
                if agent_found:
                    break
            
            if not agent_found:
                console.print(f"[red]❌ Agent with name '{target_name}' not found[/red]")
                raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error editing {target_type}:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command()
def modify(
    target_type: str = typer.Argument(..., help="Type to modify: 'crew' or 'agent'"),
    target_name: str = typer.Argument(..., help="Name of the crew or agent to modify"),
    modification: str = typer.Argument(..., help="Natural language description of the modification"),
    confirm: bool = typer.Option(True, "--confirm/--no-confirm", help="Confirm changes before applying")
):
    """Modify a crew or agent using natural language instructions.
    
    🤖 INTELLIGENT MODIFICATION: Uses AI agents to understand complex modification requests
    🔧 FALLBACK: Uses pattern-based analysis if AI is unavailable
    
    Examples:
      crewmaster modify crew my_crew "add a data analyst agent with SQL skills"
      crewmaster modify agent researcher "add web scraping and vision tools"
      crewmaster modify crew analytics_crew "change the task to focus on Q4 data only"
      crewmaster modify agent writer "update goal to create technical documentation"
    """
    console.print(f"\n[bold blue]🔧 Analyzing modification request...[/bold blue]")
    console.print(f"Target: {target_type} '{target_name}'")
    console.print(f"Request: {modification}")
    
    try:
        config = Config()
        
        # Try AI-powered modification first
        try:
            master_agent_crew = MasterAgentCrew(config)
            if master_agent_crew.get_ai_mode():
                console.print(f"\n[bold cyan]🤖 Using AI-powered modification analysis...[/bold cyan]")
                
                ai_result = master_agent_crew.modify_with_ai(target_type, target_name, modification, verbose=False)
                
                if ai_result["success"]:
                    console.print(f"\n[bold green]📋 AI Analysis Complete:[/bold green]")
                    console.print(f"[dim]{ai_result['ai_analysis'][:200]}...[/dim]")
                    
                    modification_plan = ai_result["modification_plan"]
                    
                    if modification_plan["steps"]:
                        console.print(f"\n[bold green]📋 AI Modification Plan:[/bold green]")
                        for step in modification_plan['steps']:
                            console.print(f"  • {step}")
                        
                        if confirm:
                            if not typer.confirm("\nProceed with AI-recommended modifications?"):
                                console.print("Modification cancelled.")
                                return
                        
                        # Apply the AI-generated modifications
                        master_agent = MasterAgent(config)
                        success = _apply_modifications(master_agent, target_type, target_name, modification_plan)
                        
                        if success:
                            console.print(f"\n[bold green]✅ Successfully modified {target_type} '{target_name}' using AI analysis[/bold green]")
                        else:
                            console.print(f"\n[bold red]❌ Failed to apply some AI modifications[/bold red]")
                        return
                    else:
                        console.print(f"\n[bold yellow]⚠️ AI couldn't generate a modification plan, falling back to pattern matching[/bold yellow]")
                else:
                    console.print(f"\n[bold yellow]⚠️ AI modification failed: {ai_result.get('error', 'Unknown error')}[/bold yellow]")
                    console.print(f"[bold yellow]Falling back to pattern-based analysis...[/bold yellow]")
        except Exception as e:
            console.print(f"\n[bold yellow]⚠️ AI modification unavailable: {str(e)}[/bold yellow]")
            console.print(f"[bold yellow]Using pattern-based analysis...[/bold yellow]")
        
        # Fallback to pattern-based modification
        console.print(f"\n[bold blue]🔧 Using pattern-based modification analysis...[/bold blue]")
        master_agent = MasterAgent(config)
        
        # Analyze the modification request
        modification_plan = _analyze_modification_request(modification, target_type)
        
        if not modification_plan:
            console.print("[red]❌ Could not understand the modification request[/red]")
            raise typer.Exit(1)
        
        # Display the plan
        console.print(f"\n[bold green]📋 Pattern-Based Modification Plan:[/bold green]")
        for step in modification_plan['steps']:
            console.print(f"  • {step}")
        
        if confirm:
            if not typer.confirm("\nProceed with these modifications?"):
                console.print("Modification cancelled.")
                return
        
        # Apply the modifications
        success = _apply_modifications(master_agent, target_type, target_name, modification_plan)
        
        if success:
            console.print(f"\n[bold green]✅ Successfully modified {target_type} '{target_name}'[/bold green]")
        else:
            console.print(f"\n[bold red]❌ Failed to apply some modifications[/bold red]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error modifying {target_type}:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command()
def assign_tools(
    agent_name: str = typer.Argument(..., help="Name of the agent"),
    tools: str = typer.Argument(..., help="Comma-separated list of tool names"),
    action: str = typer.Option("add", help="Action: 'add', 'remove', or 'set'")
):
    """Manually assign tools to an agent.
    
    Examples:
      crewmaster assign-tools researcher "web_search,document_search" --action add
      crewmaster assign-tools analyst "vision,database_search" --action set
      crewmaster assign-tools writer "code_execution" --action remove
    """
    console.print(f"\n[bold blue]🔧 Managing tools for agent:[/bold blue] {agent_name}")
    
    try:
        config = Config()
        master_agent = MasterAgent(config)
        
        # Parse tools
        tool_list = [tool.strip() for tool in tools.split(",")]
        console.print(f"Tools to {action}: {', '.join(tool_list)}")
        
        # Find the agent
        agent_found = False
        for crew_name, crew_model in master_agent.crew_designer._crews_cache.items():
            for agent in crew_model.agents:
                if agent.name == agent_name:
                    agent_found = True
                    
                    # Get current tools
                    current_tools = getattr(agent, 'required_tools', [])
                    console.print(f"Current tools: {', '.join(current_tools) if current_tools else 'None'}")
                    
                    # Apply action
                    if action == "add":
                        new_tools = list(set(current_tools + tool_list))
                    elif action == "remove":
                        new_tools = [t for t in current_tools if t not in tool_list]
                    elif action == "set":
                        new_tools = tool_list
                    else:
                        console.print(f"[red]❌ Invalid action '{action}'. Use 'add', 'remove', or 'set'[/red]")
                        raise typer.Exit(1)
                    
                    # Validate tools exist
                    from .tools.registry import ToolRegistry
                    tool_registry = ToolRegistry()
                    invalid_tools = [t for t in new_tools if t not in tool_registry.tools]
                    
                    if invalid_tools:
                        console.print(f"[red]❌ Invalid tools: {', '.join(invalid_tools)}[/red]")
                        console.print("Use 'crewmaster list tools' to see available tools")
                        raise typer.Exit(1)
                    
                    # Update agent
                    agent.required_tools = new_tools
                    master_agent.crew_designer._crews_cache[crew_name] = crew_model
                    master_agent.crew_designer._save_cache()
                    
                    console.print(f"[bold green]✅ Updated tools for {agent_name}:[/bold green]")
                    console.print(f"New tools: {', '.join(new_tools) if new_tools else 'None'}")
                    break
            
            if agent_found:
                break
        
        if not agent_found:
            console.print(f"[red]❌ Agent '{agent_name}' not found[/red]")
            raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"\n[bold red]❌ Error assigning tools:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command()
def create_tool(
    description: str = typer.Argument(..., help="Natural language description of what the tool should do"),
    show_code: bool = typer.Option(True, "--show-code/--no-show-code", help="Show generated code for review"),
    auto_confirm: bool = typer.Option(False, "--auto-confirm", help="Skip confirmation prompts"),
    legacy: bool = typer.Option(False, "--legacy", help="Use legacy simple tool creation"),
    use_agents: bool = typer.Option(True, "--agents/--no-agents", help="Use CrewAI agents for intelligent generation")
):
    """Create a custom CrewAI tool using AI-powered agent generation.
    
    This command uses intelligent CrewAI agents to analyze your requirements,
    generate proper CrewAI BaseTool implementations with complete functionality,
    validate the code, and integrate with CrewMaster.
    
    🎯 Two Generation Modes:
      --agents (default): Uses CrewAI agents for smart analysis and generation
      --no-agents: Uses pattern-matching approach (faster, less intelligent)
    
    Examples:
      crewmaster create-tool "Send Slack notifications to channels"
      crewmaster create-tool "Fetch weather data from OpenWeatherMap API"
      crewmaster create-tool "Process CSV files and generate Excel reports"
      crewmaster create-tool "Integrate with GitHub API for repository management"
    """
    if legacy:
        _create_legacy_tool(description)
        return
        
    # Check if we should use agent-based generation
    if use_agents:
        console.print(f"\n[bold cyan]🤖 AI Agent-Powered Tool Creation[/bold cyan]")
        console.print(f"[cyan]Description:[/cyan] {description}")
        console.print(f"[dim]Using CrewAI agents for intelligent analysis and generation...[/dim]")
        
        try:
            from .core.ai_tool_creator import AIToolCreator
            
            # Initialize the AI tool creator
            config = Config()
            llm_config = {
                "model": config.llm.model,
                "api_key": config.llm.api_key,
                "base_url": getattr(config.llm, 'base_url', None)
            }
            
            creator = AIToolCreator(llm_config)
            
            # Create the tool using AI agents
            result = creator.create_custom_tool(
                user_description=description,
                show_code=show_code,
                auto_confirm=auto_confirm
            )
            
            if result["success"]:
                console.print(f"\n[bold green]🎉 AI Agent Tool Created Successfully![/bold green]")
                console.print(f"[green]Tool Name:[/green] {result['tool_name']}")
                console.print(f"[green]Category:[/green] {result['category']}")
                console.print(f"[green]Generation Method:[/green] {result.get('generated_with', 'AI Agents')}")
                console.print(f"[green]File Location:[/green] {result['tool_file']}")
                
                if result.get('dependencies'):
                    console.print(f"[yellow]Dependencies:[/yellow] {', '.join(result['dependencies'])}")
                
                # Show warnings if any (like missing dependencies)
                if result.get('warnings'):
                    console.print(f"\n[bold yellow]⚠️  Validation Warnings:[/bold yellow]")
                    for warning in result['warnings']:
                        console.print(f"[yellow]  • {warning}[/yellow]")
                    console.print(f"[dim]💡 Tool structure is valid, install dependencies when ready to use[/dim]")
                
                console.print(f"\n[dim]💡 You can now use this tool:[/dim]")
                console.print(f"[dim]   crewmaster assign-tools <agent_name> \"{result['tool_name'].lower()}\"[/dim]")
                console.print(f"[dim]   crewmaster list-custom-tools  # View all custom tools[/dim]")
                
                # Show dependency installation instructions if needed
                if result.get('dependencies'):
                    console.print(f"\n[dim]📦 Install dependencies:[/dim]")
                    console.print(f"[dim]   pip install {' '.join(result['dependencies'])}[/dim]")
            else:
                console.print(f"\n[bold red]❌ AI Agent Tool Creation Failed[/bold red]")
                console.print(f"[red]Error:[/red] {result['message']}")
                
                if result.get('generated_code'):
                    console.print(f"\n[yellow]🔍 Generated Code (for debugging):[/yellow]")
                    console.print(f"[dim]{result['generated_code'][:500]}...[/dim]")
                
                console.print(f"\n[dim]💡 Try:[/dim]")
                console.print(f"[dim]   --no-agents  # Use pattern-based generation[/dim]")
                console.print(f"[dim]   --legacy     # Use simple legacy tool creation[/dim]")
                
        except ValueError as e:
            console.print(f"\n[bold yellow]⚠️  AI agents require LLM configuration[/bold yellow]")
            console.print(f"[yellow]Error:[/yellow] {str(e)}")
            console.print(f"\n[dim]💡 Solutions:[/dim]")
            console.print(f"[dim]   export OPENAI_API_KEY=\"your-key\"  # Set OpenAI API key[/dim]")
            console.print(f"[dim]   crewmaster create-tool \"{description}\" --no-agents  # Use pattern-based generation[/dim]")
            raise typer.Exit(1)
            
        except Exception as e:
            console.print(f"\n[bold red]❌ Error during AI agent tool creation:[/bold red] {str(e)}")
            console.print(f"\n[yellow]💡 Fallback options:[/yellow]")
            console.print(f"[yellow]   --no-agents  # Use pattern-based generation[/yellow]")
            console.print(f"[yellow]   --legacy     # Use simple tool creation[/yellow]")
            raise typer.Exit(1)
    else:
        # Use the existing pattern-based approach
        console.print(f"\n[bold blue]🔧 Pattern-Based Tool Creation[/bold blue]")
        console.print(f"[cyan]Description:[/cyan] {description}")
        
        try:
            from .core.intelligent_tool_creator import IntelligentToolCreator
            
            # Initialize the pattern-based tool creator
            config = Config()
            llm_config = {
                "model": config.llm.model,
                "api_key": config.llm.api_key,
                "base_url": getattr(config.llm, 'base_url', None)
            }
            
            creator = IntelligentToolCreator(llm_config)
            
            # Create the tool using pattern-based approach
            result = creator.create_custom_tool(
                user_description=description,
                show_code=show_code,
                auto_confirm=auto_confirm
            )
            
            if result["success"]:
                console.print(f"\n[bold green]✅ Pattern-Based Tool Created![/bold green]")
                console.print(f"[green]Tool Name:[/green] {result['tool_name']}")
                console.print(f"[green]Category:[/green] {result['category']}")
                console.print(f"[green]File Location:[/green] {result['tool_file']}")
                console.print(f"\n[dim]💡 You can now assign this tool to agents:[/dim]")
                console.print(f"[dim]   crewmaster assign-tools <agent_name> \"{result['tool_name'].lower()}\"[/dim]")
            else:
                console.print(f"\n[bold red]❌ Tool Creation Failed[/bold red]")
                console.print(f"[red]Error:[/red] {result['message']}")
                
        except Exception as e:
            console.print(f"\n[bold red]❌ Error during pattern-based tool creation:[/bold red] {str(e)}")
            console.print(f"\n[yellow]💡 Tip:[/yellow] Try using --legacy for simple tool creation")
            raise typer.Exit(1)

def _create_legacy_tool(description: str):
    """Create a simple legacy tool (fallback)."""
    console.print(f"\n[bold yellow]⚙️  Legacy Tool Creation[/bold yellow]")
    
    name = typer.prompt("Tool name")
    category = typer.prompt("Tool category", default="custom")
    
    try:
        from .tools.registry import ToolRegistry
        tool_registry = ToolRegistry()
        
        def tool_function(input_data: str = "") -> str:
            return f"Legacy tool '{name}' executed with input: {input_data}"
        
        success = tool_registry.create_custom_tool(name, description, category, tool_function)
        
        if success:
            console.print(f"[bold green]✅ Created legacy tool '{name}'[/bold green]")
        else:
            console.print(f"[red]❌ Failed to create legacy tool '{name}'[/red]")
            
    except Exception as e:
        console.print(f"[red]❌ Error creating legacy tool: {str(e)}[/red]")

@app.command()
def list_custom_tools(
    show_ai_tools: bool = typer.Option(True, "--ai/--no-ai", help="Include AI-generated tools"),
    show_pattern_tools: bool = typer.Option(True, "--pattern/--no-pattern", help="Include pattern-generated tools"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed information")
):
    """List all custom tools created with different generation methods."""
    console.print(f"\n[bold blue]📋 Custom CrewAI Tools[/bold blue]")
    
    try:
        all_tools = []
        
        # Get AI-generated tools
        if show_ai_tools:
            try:
                from .core.ai_tool_creator import AIToolCreator
                ai_creator = AIToolCreator({}, require_llm=False)  # No LLM needed for listing
                ai_tools = ai_creator.list_ai_generated_tools()
                all_tools.extend(ai_tools)
            except Exception as e:
                console.print(f"[dim]Note: Could not load AI tools: {str(e)}[/dim]")
        
        # Get pattern-generated tools  
        if show_pattern_tools:
            try:
                from .core.intelligent_tool_creator import IntelligentToolCreator
                pattern_creator = IntelligentToolCreator()
                pattern_tools = pattern_creator.list_custom_tools()
                # Add type identifier to distinguish from AI tools
                for tool in pattern_tools:
                    tool['type'] = 'Pattern Generated'
                all_tools.extend(pattern_tools)
            except Exception as e:
                console.print(f"[dim]Note: Could not load pattern tools: {str(e)}[/dim]")
        
        if not all_tools:
            console.print("[dim]No custom tools found.[/dim]")
            console.print("[dim]💡 Create tools with:[/dim]")
            console.print("[dim]   crewmaster create-tool \"description\"  # AI agents (default)[/dim]")
            console.print("[dim]   crewmaster create-tool \"description\" --no-agents  # Pattern-based[/dim]")
            return
        
        # Sort by creation time (newest first)
        all_tools.sort(key=lambda x: x['created'], reverse=True)
        
        # Display tools table
        table = Table(title=f"Custom Tools ({len(all_tools)} total)")
        table.add_column("Tool Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Description", style="blue")
        table.add_column("Created", style="green")
        
        if detailed:
            table.add_column("Size", style="dim")
            table.add_column("File", style="dim")
        
        from datetime import datetime
        for tool in all_tools:
            created_date = datetime.fromtimestamp(tool['created']).strftime("%Y-%m-%d %H:%M")
            
            # Truncate description
            description = tool['description']
            if len(description) > 45:
                description = description[:42] + "..."
            
            # Get tool type
            tool_type = tool.get('type', 'Unknown')
            
            row_data = [
                tool['name'].title(),
                tool_type,
                description,
                created_date
            ]
            
            if detailed:
                # Add size info
                size_kb = tool.get('size', 0) // 1024 if tool.get('size') else 0
                row_data.append(f"{size_kb}KB")
                row_data.append(tool['file'].split('/')[-1])  # Just filename
            
            table.add_row(*row_data)
        
        console.print(table)
        
        # Show statistics
        ai_count = len([t for t in all_tools if t.get('type') == 'AI Generated'])
        pattern_count = len([t for t in all_tools if t.get('type') == 'Pattern Generated'])
        
        console.print(f"\n[dim]📊 Statistics:[/dim]")
        console.print(f"[dim]   AI Generated: {ai_count}[/dim]")
        console.print(f"[dim]   Pattern Generated: {pattern_count}[/dim]")
        console.print(f"[dim]   Tools are stored in: /tmp/crewmaster_custom_tools/[/dim]")
        
        console.print(f"\n[dim]💡 Management commands:[/dim]")
        console.print(f"[dim]   crewmaster list-custom-tools --detailed  # Show more info[/dim]")
        console.print(f"[dim]   crewmaster assign-tools <agent> <tool_name>  # Assign to agent[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Error listing custom tools: {str(e)}[/red]")


@app.command()
def ai_tool_stats():
    """Show statistics about AI-generated tools."""
    console.print(f"\n[bold blue]📊 AI Tool Generation Statistics[/bold blue]")
    
    try:
        from .core.ai_tool_creator import AIToolCreator
        
        creator = AIToolCreator({}, require_llm=False)  # No LLM needed for stats
        
        stats = creator.get_generation_stats()
        
        # Create stats table
        table = Table(title="AI Generation Stats")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total AI Tools", str(stats['total_tools']))
        table.add_row("Total Code Size", f"{stats['total_size']:,} bytes")
        table.add_row("Average Tool Size", f"{stats['avg_size']:,} bytes")
        table.add_row("Tools Directory", stats['tools_directory'])
        
        if stats['newest_tool']:
            from datetime import datetime
            newest_date = datetime.fromtimestamp(stats['newest_tool']['created']).strftime("%Y-%m-%d %H:%M")
            table.add_row("Newest Tool", f"{stats['newest_tool']['name']} ({newest_date})")
        
        if stats['oldest_tool']:
            from datetime import datetime
            oldest_date = datetime.fromtimestamp(stats['oldest_tool']['created']).strftime("%Y-%m-%d %H:%M")
            table.add_row("Oldest Tool", f"{stats['oldest_tool']['name']} ({oldest_date})")
        
        console.print(table)
        
        if stats['total_tools'] == 0:
            console.print("\n[dim]💡 Create your first AI tool with:[/dim]")
            console.print("[dim]   crewmaster create-tool \"your tool description\"[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Error getting AI tool stats: {str(e)}[/red]")

@app.command()
def performance(
    crew_name: str = typer.Argument(..., help="Name of crew to analyze"),
    show_history: bool = typer.Option(False, "--history", help="Show execution history"),
    days: int = typer.Option(7, help="Number of days to analyze")
):
    """Show performance metrics and statistics for crews.
    
    Examples:
      crewmaster performance my_research_crew
      crewmaster performance analytics_crew --history --days 30
    """
    console.print(f"\n[bold blue]📊 Performance Analysis:[/bold blue] {crew_name}")
    
    try:
        config = Config()
        master_agent = MasterAgent(config)
        
        # Check if crew exists
        if crew_name not in master_agent.crew_designer._crews_cache:
            console.print(f"[red]❌ Crew '{crew_name}' not found[/red]")
            raise typer.Exit(1)
        
        _show_crew_performance(master_agent, crew_name, show_history, days)
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error analyzing performance:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command()
def export(
    target_name: str = typer.Argument(..., help="Name of crew to export"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("json", help="Export format: 'json', 'yaml', or 'python'"),
    include_history: bool = typer.Option(False, "--include-history", help="Include execution history")
):
    """Export crew configuration and data.
    
    Examples:
      crewmaster export my_research_crew
      crewmaster export analytics_crew --output crew_backup.json
      crewmaster export data_team --format yaml --include-history
      crewmaster export research_crew --format python --output research_crew.zip
    """
    console.print(f"\n[bold blue]📤 Exporting crew:[/bold blue] {target_name}")
    
    try:
        config = Config()
        master_agent = MasterAgent(config)
        
        # Get crew data
        crew_data = _export_crew_data(master_agent, target_name, include_history)
        
        if not crew_data:
            console.print(f"[red]❌ Crew '{target_name}' not found[/red]")
            raise typer.Exit(1)
        
        # Handle Python code export differently
        if format == "python":
            # Determine output file for Python code export
            if not output_file:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                output_file = f"{target_name}_{timestamp}.zip"
            elif not output_file.endswith('.zip'):
                output_file = f"{output_file}.zip"
            
            # Convert export data to simplified format for code generator
            simplified_crew_data = {
                "name": crew_data["crewmaster_export"]["crew"]["name"],
                "task": crew_data["crewmaster_export"]["crew"]["task"],
                "description": crew_data["crewmaster_export"]["crew"]["description"],
                "agents": crew_data["crewmaster_export"]["crew"]["agents"]
            }
            
            # Generate Python code package
            console.print("[cyan]🐍 Generating Python code package...[/cyan]")
            success = generate_crew_code_package(simplified_crew_data, output_file)
            
            if success:
                console.print(f"[bold green]✅ Python code package exported to:[/bold green] {output_file}")
                console.print(f"[dim]Extract and run with: unzip {output_file} && cd {target_name}_crew && ./run.sh[/dim]")
            else:
                console.print("[red]❌ Python code export failed[/red]")
        else:
            # Traditional JSON/YAML export
            # Determine output file
            if not output_file:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                output_file = f"{target_name}_{timestamp}.{format}"
            
            # Export data
            success = _save_export_data(crew_data, output_file, format)
            
            if success:
                console.print(f"[bold green]✅ Exported crew to:[/bold green] {output_file}")
                console.print(f"[dim]Use 'crewmaster import {output_file}' to restore[/dim]")
            else:
                console.print("[red]❌ Export failed[/red]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error exporting:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command()
def stats(
    scope: str = typer.Option("system", help="Scope: 'system', 'crews', 'agents', or 'tools'"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed statistics"),
    days: int = typer.Option(30, help="Number of days for activity analysis")
):
    """Show system statistics and analytics.
    
    Examples:
      crewmaster stats
      crewmaster stats --scope crews --detailed  
      crewmaster stats --scope agents --days 7
    """
    console.print(f"\n[bold blue]📈 CrewMaster Statistics[/bold blue]")
    
    try:
        config = Config()
        master_agent = MasterAgent(config)
        
        if scope in ["system", "all"]:
            _show_system_stats(master_agent, detailed, days)
        
        if scope in ["crews", "all"]:
            _show_crews_stats(master_agent, detailed, days)
        
        if scope in ["agents", "all"]:
            _show_agents_stats(master_agent, detailed, days)
        
        if scope in ["tools", "all"]:
            _show_tools_stats(master_agent, detailed)
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error generating statistics:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command()
def import_crew(
    file_path: str = typer.Argument(..., help="Path to crew export file"),
    new_name: Optional[str] = typer.Option(None, "--name", help="New name for imported crew"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing crew with same name")
):
    """Import crew from exported file.
    
    Examples:
      crewmaster import crew_backup.json
      crewmaster import analytics_team.json --name new_analytics_crew
      crewmaster import backup.yaml --overwrite
    """
    console.print(f"\n[bold blue]📥 Importing crew from:[/bold blue] {file_path}")
    
    try:
        if not os.path.exists(file_path):
            console.print(f"[red]❌ File not found: {file_path}[/red]")
            raise typer.Exit(1)
        
        config = Config()
        master_agent = MasterAgent(config)
        
        # Load and import crew data
        success = _import_crew_data(master_agent, file_path, new_name, overwrite)
        
        if success:
            console.print("[bold green]✅ Crew imported successfully[/bold green]")
            console.print("[dim]Use 'crewmaster list crews' to see imported crew[/dim]")
        else:
            console.print("[red]❌ Import failed[/red]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error importing:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def welcome():
    """Display the CrewMaster welcome screen with banner and quick start guide."""
    display_banner()
    
    console.print("\n[bold yellow]🎯 Getting Started Guide[/bold yellow]")
    console.print("=" * 60)
    
    console.print("\n[bold green]Step 1:[/bold green] Create your first crew with AI orchestration")
    console.print("  [cyan]crewmaster create \"analyze competitor pricing and create a report\"[/cyan]")
    
    console.print("\n[bold green]Step 2:[/bold green] List and inspect your crews")
    console.print("  [cyan]crewmaster list crews[/cyan]")
    console.print("  [cyan]crewmaster inspect my_analysis_crew[/cyan]")
    
    console.print("\n[bold green]Step 3:[/bold green] Run your crew (requires OpenAI API key)")
    console.print("  [cyan]export OPENAI_API_KEY=\"your-key\"[/cyan]")
    console.print("  [cyan]crewmaster run my_analysis_crew[/cyan]")
    
    console.print("\n[bold green]Step 4:[/bold green] Monitor and optimize")
    console.print("  [cyan]crewmaster performance my_analysis_crew --history[/cyan]")
    console.print("  [cyan]crewmaster stats --detailed[/cyan]")
    
    console.print("\n[dim]💡 Tips:[/dim]")
    console.print("[dim]  • AI orchestration is now the default - smarter task analysis and agent design[/dim]")
    console.print("[dim]  • Tools are automatically assigned based on your task description[/dim]")
    console.print("[dim]  • Use memorable names - no complex IDs required[/dim]")
    console.print("[dim]  • Export/import crews for backup and sharing[/dim]")
    console.print("[dim]  • Try 'crewmaster help-extended' for comprehensive documentation[/dim]")

@app.command()
def help_extended():
    """Show comprehensive help with examples and usage patterns."""
    console.print("\n[bold cyan]🚀 CrewMaster - Complete Command Reference[/bold cyan]")
    console.print("=" * 60)
    
    console.print("\n[bold green]📋 CREW CREATION & ANALYSIS[/bold green]")
    console.print("• [cyan]crewmaster create[/cyan] \"task description\" [--name \"custom_name\"]")
    console.print("  └─ 🤖 [bold]Default[/bold]: AI orchestration - intelligent task analysis, agent design, optimization")
    console.print("  └─ ⚙️  [bold]--legacy[/bold]: Simple regex patterns - faster, no LLM required")
    console.print("  └─ ✅ Creates a crew you can execute with 'run' command")
    console.print("  └─ Example: crewmaster create \"research AI trends and write report\"")
    
    
    console.print("• [cyan]crewmaster run[/cyan] <crew_name> [--input \"context\"]")
    console.print("  └─ 🏃 [bold]Execute Crew[/bold]: Actually runs the crew to perform the task")
    console.print("  └─ ⚡ Requires OpenAI API key and existing crew")
    console.print("  └─ 📊 This is where the real work happens")
    console.print("  └─ Example: crewmaster run my_research_crew --input \"focus on 2024 data\"")
    
    console.print("• [cyan]crewmaster list crews[/cyan]")
    console.print("  └─ Shows all created crews with names and agent counts")
    
    console.print("• [cyan]crewmaster run[/cyan] <crew_name> [--input \"context\"]")
    console.print("  └─ Executes a crew (requires OpenAI API key)")
    console.print("  └─ Example: crewmaster run ai_trends_crew --input \"focus on 2024 data\"")
    
    console.print("• [cyan]crewmaster modify crew[/cyan] <crew_name> \"natural language instruction\"")
    console.print("  └─ Modify crews using natural language")
    console.print("  └─ Example: crewmaster modify crew my_crew \"add a data analyst agent\"")
    
    console.print("• [cyan]crewmaster delete[/cyan] <crew_name> [--confirm]")
    console.print("  └─ Removes a crew and all its agents")
    console.print("  └─ Example: crewmaster delete ai_trends_crew")
    
    console.print("• [cyan]crewmaster inspect[/cyan] <crew_name>")
    console.print("  └─ Detailed view of crew configuration and agents")
    console.print("  └─ Example: crewmaster inspect ai_trends_crew")
    
    console.print("• [cyan]crewmaster edit crew[/cyan] <crew_name>")
    console.print("  └─ Modify crew properties (name, task, description)")
    console.print("  └─ Example: crewmaster edit crew ai_trends_crew")
    
    console.print("\n[bold green]👥 AGENT MANAGEMENT[/bold green]")
    console.print("• [cyan]crewmaster list agents[/cyan]")
    console.print("  └─ Shows all agents across all crews with their tools")
    
    console.print("• [cyan]crewmaster edit agent[/cyan] <agent_name>")
    console.print("  └─ Interactive editing of agent properties")
    console.print("  └─ Fields: name, role, goal, backstory, required_tools")
    
    console.print("• [cyan]crewmaster modify agent[/cyan] <agent_name> \"natural language instruction\"")
    console.print("  └─ Modify agents using natural language")
    console.print("  └─ Example: crewmaster modify agent researcher \"add vision and scraping tools\"")
    
    console.print("• [cyan]crewmaster assign-tools[/cyan] <agent_name> \"tool1,tool2\" [--action add|remove|set]")
    console.print("  └─ Manually assign tools to agents")
    console.print("  └─ Example: crewmaster assign-tools researcher \"web_scraping,vision\" --action add")
    
    console.print("\n[bold green]🛠️  TOOLS & CAPABILITIES[/bold green]")
    console.print("• [cyan]crewmaster list tools[/cyan]")
    console.print("  └─ Shows all available tools with descriptions and categories")
    console.print("  └─ Core Tools: web_search, file_operations, code_execution, data_processing")
    console.print("  └─ CrewAI Tools: web_scraping, document_search, github_search, youtube_search")
    console.print("  └─ AI Tools: vision (DALL-E), browser_automation, database_search")
    
    console.print("• [cyan]crewmaster create-tool[/cyan] \"description\" [--agents/--no-agents] [--show-code]")
    console.print("  └─ 🤖 [bold]AI Agents (default)[/bold]: Uses 3 CrewAI agents for intelligent tool generation")
    console.print("  └─ ⚡ Analyzes requirements → Generates CrewAI BaseTool → Validates code")
    console.print("  └─ 📦 Handles dependencies, shows code preview, validates implementation")
    console.print("  └─ Example: crewmaster create-tool \"Send Slack notifications with emoji support\"")
    
    console.print("• [cyan]crewmaster create-tool[/cyan] \"description\" --no-agents")
    console.print("  └─ 🔧 [bold]Pattern-Based[/bold]: Fast pattern-matching approach (no LLM required)")
    console.print("  └─ Example: crewmaster create-tool \"API integration tool\" --no-agents")
    
    console.print("• [cyan]crewmaster create-tool[/cyan] \"description\" --legacy")
    console.print("  └─ ⚙️  [bold]Legacy[/bold]: Simple tool creation (fallback option)")
    
    console.print("• [cyan]crewmaster list-custom-tools[/cyan] [--ai/--pattern] [--detailed]")
    console.print("  └─ List all custom tools created with different generation methods")
    console.print("  └─ Shows creation time, type (AI/Pattern), and validation status")
    
    console.print("• [cyan]crewmaster ai-tool-stats[/cyan]")
    console.print("  └─ Statistics about AI-generated tools and generation performance")
    
    console.print("\n[bold green]📊 PERFORMANCE & ANALYTICS[/bold green]")
    console.print("• [cyan]crewmaster performance[/cyan] <crew_name> [--history] [--days N]")
    console.print("  └─ Show detailed performance metrics and execution statistics")
    console.print("  └─ Example: crewmaster performance my_research_crew --history")
    console.print("  └─ Example: crewmaster performance analytics_crew --days 30")
    
    console.print("• [cyan]crewmaster stats[/cyan] [--scope system|crews|agents|tools] [--detailed]")
    console.print("  └─ Show comprehensive system analytics and usage statistics")
    console.print("  └─ Example: crewmaster stats --scope crews --detailed")
    console.print("  └─ Example: crewmaster stats --scope tools")
    
    console.print("\n[bold green]💾 BACKUP & SHARING[/bold green]")
    console.print("• [cyan]crewmaster export[/cyan] <crew_name> [--output file.json] [--format json|yaml]")
    console.print("  └─ Export crew configuration for backup or sharing")
    console.print("  └─ Example: crewmaster export analytics_crew --output backup.json")
    console.print("  └─ Example: crewmaster export research_team --format yaml --include-history")
    
    console.print("• [cyan]crewmaster import-crew[/cyan] <file_path> [--name new_crew_name] [--overwrite]")
    console.print("  └─ Import crew from exported backup file")
    console.print("  └─ Example: crewmaster import-crew backup.json --name restored_crew")
    console.print("  └─ Example: crewmaster import-crew team_config.yaml --overwrite")
    
    console.print("\n[bold green]📚 EXAMPLES BY USE CASE[/bold green]")
    console.print("[dim]Research & Analysis:[/dim]")
    console.print("  crewmaster create \"research competitor pricing and create comparison chart\"")
    console.print("  crewmaster performance competitor_research_crew --history --days 30")
    
    console.print("[dim]Development:[/dim]")
    console.print("  crewmaster create \"build REST API with authentication and write tests\"")
    console.print("  crewmaster assign-tools api_developer \"github_search,code_execution\" --action add")
    
    console.print("[dim]Content Creation:[/dim]")
    console.print("  crewmaster create \"write blog post about AI trends with SEO optimization\"")
    console.print("  crewmaster modify agent content_writer \"add web scraping for research\"")
    
    console.print("[dim]Data Processing:[/dim]")
    console.print("  crewmaster create \"analyze sales data and generate executive dashboard\"")
    console.print("  crewmaster export data_analysis_crew --format yaml --include-history")
    
    console.print("\n[bold green]⚙️  AUTOMATIC TOOL ASSIGNMENT[/bold green]")
    console.print("CrewMaster automatically assigns tools based on task keywords:")
    console.print("• [yellow]research, search, urls[/yellow] → web_search tool")
    console.print("• [yellow]scrape, crawl, website[/yellow] → web_scraping tool") 
    console.print("• [yellow]file, document, pdf, csv[/yellow] → file_operations, document_search tools")
    console.print("• [yellow]code, script, python, github[/yellow] → code_execution, github_search tools")
    console.print("• [yellow]data, analyze, database[/yellow] → data_processing, database_search tools")
    console.print("• [yellow]image, photo, dall-e[/yellow] → vision tool")
    console.print("• [yellow]youtube, video, channel[/yellow] → youtube_search tool")
    console.print("• [yellow]browser, automation[/yellow] → browser_automation tool")
    
    console.print("\n[bold green]🎯 COMMAND WORKFLOW EXPLAINED[/bold green]")
    
    console.print("\n[bold blue]📝 Step-by-Step Process:[/bold blue]")
    console.print("[bold yellow]Step 1: Create Crew[/bold yellow]")
    console.print("  [cyan]crewmaster create[/cyan] \"your task\"")  
    console.print("  └─ 🤖 Creates actual crew with agents and tools")
    console.print("  └─ 💾 Crew is saved and ready for execution")
    
    console.print("\n[bold yellow]Step 2: Execute Work[/bold yellow]")
    console.print("  [cyan]crewmaster run[/cyan] <crew_name>")
    console.print("  └─ 🏃 Agents actually perform the task")
    console.print("  └─ 📊 Produces real output/results")
    
    console.print("\n[bold blue]🔄 Complete Example Workflow:[/bold blue]")
    console.print("1. [dim]# Create the crew (saves it for later use)[/dim]")
    console.print("   [cyan]crewmaster create[/cyan] \"research competitor pricing and create report\"")
    console.print("   [dim]→ Creates: competitor_pricing_report_crew[/dim]")
    
    console.print("\n2. [dim]# See what crews you have[/dim]")
    console.print("   [cyan]crewmaster list crews[/cyan]")
    console.print("   [dim]→ Shows: all your created crews[/dim]")
    
    console.print("\n3. [dim]# Execute the crew to do the actual work[/dim]")
    console.print("   [cyan]crewmaster run[/cyan] competitor_pricing_report_crew")
    console.print("   [dim]→ Produces: actual research report[/dim]")
    
    console.print("\n[bold blue]🚀 Quick Workflow:[/bold blue]")
    console.print("1. [cyan]crewmaster create[/cyan] \"your task\" (creates crew)")
    console.print("2. [cyan]crewmaster run[/cyan] <crew_name> (executes crew)")
    console.print("3. [cyan]crewmaster performance[/cyan] <crew_name> (check results)")
    
    console.print("\n[bold green]💡 NAME-BASED SIMPLICITY[/bold green]")
    console.print("• All crews and agents use memorable names instead of complex UUIDs")
    console.print("• Simply use the name you see in [cyan]crewmaster list[/cyan] commands")
    console.print("• Example: [cyan]crewmaster run my_research_crew[/cyan]")
    console.print("• Example: [cyan]crewmaster performance my_research_crew[/cyan]")
    console.print("• Example: [cyan]crewmaster export analytics_team[/cyan]")
    console.print("• No more copying and pasting long IDs!")
    
    console.print("\n[bold green]🔧 CUSTOM TOOLS & EXTENSIBILITY[/bold green]")
    console.print("• 🤖 [bold]AI-Powered Tool Generation[/bold]: Uses 3 CrewAI agents for intelligent analysis")
    console.print("• 📦 Complete CrewAI BaseTool implementations with proper validation")
    console.print("• 🔍 Dependency management and code preview before creation")
    console.print("• ⚡ Multiple generation modes: AI agents, pattern-based, or legacy")
    console.print("• 💾 Tools are automatically registered and available across sessions")
    console.print("• Example workflow:")
    console.print("  1. [cyan]crewmaster create-tool[/cyan] \"Send Slack notifications with emoji support\"")
    console.print("  2. [cyan]crewmaster list-custom-tools[/cyan] (see all AI-generated tools)")
    console.print("  3. [cyan]crewmaster assign-tools[/cyan] project_manager \"slackmessageformattertool\" --action add")
    console.print("  4. [cyan]crewmaster ai-tool-stats[/cyan] (view generation statistics)")
    
    console.print("\n[bold green]📊 MONITORING & OPTIMIZATION[/bold green]")
    console.print("• Track crew and agent performance with detailed metrics")
    console.print("• Analyze tool usage patterns across your system")
    console.print("• Export configurations for version control and sharing")
    console.print("• Monitor execution history and success rates")
    console.print("• Use stats to optimize crew composition and tool assignments")
    
    console.print("\n[bold green]📖 SYSTEM & MAINTENANCE[/bold green]")
    console.print("• [cyan]crewmaster reset[/cyan] [--confirm] [--keep-tools]")
    console.print("  └─ Delete all data for a fresh start (DESTRUCTIVE)")
    console.print("  └─ Example: crewmaster reset --keep-tools --confirm")
    console.print("• [cyan]crewmaster stats[/cyan] [--scope crews|agents|tools] [--detailed]")
    console.print("  └─ System overview and statistics")
    console.print("• [cyan]crewmaster version[/cyan] - Show version information")
    
    # console.print("\n[bold green]📖 MORE HELP[/bold green]")
    # console.print("• [cyan]crewmaster <command> --help[/cyan] - Detailed help for specific commands")
    # console.print("• [cyan]crewmaster --help[/cyan] - Basic command list")
    
    console.print(f"\n[dim]💡 All crews and agents are automatically cached for persistence between sessions.[/dim]")
    console.print(f"[dim]🔄 Use export/import for advanced backup and team collaboration workflows.[/dim]")
    console.print(f"[dim]📈 Performance and stats commands help optimize your multi-agent systems.[/dim]")

@app.command()
def reset(
    confirm: bool = typer.Option(False, "--confirm", "-y", help="Skip confirmation prompt"),
    keep_custom_tools: bool = typer.Option(False, "--keep-tools", help="Keep custom tools (only delete crews/agents)")
):
    """Reset CrewMaster by deleting all data for a fresh start.
    
    This command will delete:
    • All crews and their configurations
    • All agents and their settings
    • All cache files and execution history
    • Optionally custom tools (use --keep-tools to preserve)
    
    Examples:
      crewmaster reset --confirm
      crewmaster reset --keep-tools --confirm
    """
    console.print("\n[bold red]⚠️  WARNING: RESET OPERATION[/bold red]")
    console.print("This will permanently delete:")
    console.print("• All crews and agents")
    console.print("• All cache files and history")
    if not keep_custom_tools:
        console.print("• All custom tools")
    console.print("\n[yellow]This action cannot be undone![/yellow]")
    
    if not confirm:
        response = typer.confirm("\nAre you sure you want to reset CrewMaster?")
        if not response:
            console.print("[green]Reset cancelled[/green]")
            return
    
    try:
        _perform_reset(keep_custom_tools)
        console.print("\n[bold green]✅ CrewMaster has been reset successfully![/bold green]")
        console.print("You can now start fresh with 'crewmaster create-crew'")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error during reset:[/bold red] {str(e)}")
        raise typer.Exit(1)

# Helper functions for commands

def _show_crew_performance(master_agent, crew_name: str, show_history: bool, days: int):
    """Show performance metrics for a crew."""
    crew = master_agent.crew_designer.get_crew_from_cache(crew_name)
    if not crew:
        console.print(f"[red]❌ Crew '{crew_name}' not found[/red]")
        return
    
    table = Table(title=f"Performance Metrics: {crew_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Get real execution statistics
    stats = master_agent._get_crew_execution_stats(crew_name)
    
    # Calculate metrics
    total_executions = stats['total_executions']
    successful_executions = stats['successful_executions']
    total_execution_time = stats['total_execution_time']
    
    success_rate = "N/A"
    if total_executions > 0:
        success_rate = f"{(successful_executions / total_executions) * 100:.1f}%"
    
    avg_execution_time = "N/A"
    if total_executions > 0:
        avg_execution_time = f"{total_execution_time / total_executions:.1f}s"
    
    last_executed = stats.get('last_executed', 'Never')
    if last_executed and last_executed != 'Never':
        from datetime import datetime
        try:
            last_exec_dt = datetime.fromisoformat(last_executed)
            last_executed = last_exec_dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            pass
    
    table.add_row("Crew Name", crew.name)
    table.add_row("Total Agents", str(len(crew.agents)))
    table.add_row("Task", crew.task[:50] + "..." if len(crew.task) > 50 else crew.task)
    table.add_row("Executions", str(total_executions))
    table.add_row("Success Rate", success_rate)
    table.add_row("Avg Execution Time", avg_execution_time)
    table.add_row("Last Executed", last_executed)
    
    console.print(table)
    
    if show_history:
        console.print(f"\n[bold blue]📈 Execution History (Last {days} days):[/bold blue]")
        execution_history = stats.get('execution_history', [])
        
        if not execution_history:
            console.print("[dim]No execution history available yet[/dim]")
        else:
            # Filter by days
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            
            recent_history = []
            for entry in execution_history:
                try:
                    entry_date = datetime.fromisoformat(entry['timestamp'])
                    if entry_date >= cutoff_date:
                        recent_history.append(entry)
                except:
                    continue
            
            if not recent_history:
                console.print(f"[dim]No executions found in the last {days} days[/dim]")
            else:
                history_table = Table()
                history_table.add_column("Timestamp", style="cyan")
                history_table.add_column("Execution Time", style="green")
                history_table.add_column("Status", style="blue")
                
                for entry in recent_history[-10:]:  # Show last 10
                    try:
                        timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%m/%d %H:%M:%S")
                        execution_time = f"{entry['execution_time']}s"
                        status = "✅ Success" if entry['status'] == "completed" else "❌ Failed"
                        history_table.add_row(timestamp, execution_time, status)
                    except:
                        continue
                
                console.print(history_table)
                
                if len(recent_history) > 10:
                    console.print(f"[dim]Showing last 10 of {len(recent_history)} executions[/dim]")


def _export_crew_data(master_agent, crew_name: str, include_history: bool) -> Optional[Dict[str, Any]]:
    """Export crew data for backup/sharing."""
    crew = master_agent.crew_designer.get_crew_from_cache(crew_name)
    if not crew:
        return None
    
    # Build export data structure
    export_data = {
        "crewmaster_export": {
            "version": "1.0",
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "crew": {
                "name": crew.name,
                "task": crew.task,
                "description": crew.description,
                "agents": []
            }
        }
    }
    
    # Add agents data
    for agent in crew.agents:
        agent_data = {
            "name": agent.name,
            "role": agent.role,
            "goal": agent.goal,
            "backstory": agent.backstory,
            "required_tools": getattr(agent, 'required_tools', [])
        }
        export_data["crewmaster_export"]["crew"]["agents"].append(agent_data)
    
    # Add execution history if requested
    if include_history:
        export_data["crewmaster_export"]["execution_history"] = []
        # TODO: Add actual execution history when tracking is implemented
    
    return export_data

def _save_export_data(data: Dict[str, Any], output_file: str, format: str) -> bool:
    """Save export data to file."""
    try:
        with open(output_file, 'w') as f:
            if format.lower() == 'yaml':
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:  # Default to JSON
                json.dump(data, f, indent=2)
        return True
    except Exception as e:
        console.print(f"[red]❌ Error saving export file: {e}[/red]")
        return False

def _import_crew_data(master_agent, file_path: str, new_name: Optional[str], overwrite: bool) -> bool:
    """Import crew data from file."""
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Validate export format
        if "crewmaster_export" not in data:
            console.print("[red]❌ Invalid export format[/red]")
            return False
        
        crew_data = data["crewmaster_export"]["crew"]
        crew_name = new_name or crew_data["name"]
        
        # Check if crew already exists
        if crew_name in master_agent.crew_designer._crews_cache and not overwrite:
            console.print(f"[red]❌ Crew '{crew_name}' already exists. Use --overwrite to replace it[/red]")
            return False
        
        # Create crew spec from imported data
        from .core.task_analyzer import CrewSpec, AgentSpec, TaskComplexity
        
        agents = []
        for agent_data in crew_data["agents"]:
            agent_spec = AgentSpec(
                role=agent_data["role"],
                name=agent_data["name"],
                goal=agent_data["goal"],
                backstory=agent_data["backstory"],
                required_tools=agent_data.get("required_tools", [])
            )
            agents.append(agent_spec)
        
        crew_spec = CrewSpec(
            name=crew_name,
            task=crew_data["task"],
            description=crew_data["description"],
            agents=agents,
            expected_output="Imported crew task completion",
            complexity=TaskComplexity.MODERATE,
            estimated_time=15
        )
        
        # Create the crew
        if overwrite and crew_name in master_agent.crew_designer._crews_cache:
            del master_agent.crew_designer._crews_cache[crew_name]
        
        crew_model = master_agent.crew_designer.create_crew_from_spec(crew_spec, reuse_agents=False)
        console.print(f"[green]✅ Successfully imported crew: {crew_model.name}[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error importing crew: {e}[/red]")
        return False

def _show_system_stats(master_agent, detailed: bool, days: int):
    """Show system-wide statistics."""
    crews = list(master_agent.crew_designer._crews_cache.values())
    
    # Calculate statistics
    total_crews = len(crews)
    total_agents = sum(len(crew.agents) for crew in crews)
    
    table = Table(title="System Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Crews", str(total_crews))
    table.add_row("Total Agents", str(total_agents))
    table.add_row("Avg Agents per Crew", f"{total_agents/total_crews:.1f}" if total_crews > 0 else "0")
    table.add_row("Storage Location", "~/.crewmaster/ (cache)")
    
    if detailed:
        # Tool usage statistics
        from .tools.registry import ToolRegistry
        tool_registry = ToolRegistry()
        table.add_row("Available Tools", str(len(tool_registry.tools)))
        table.add_row("Custom Tools", str(len([t for t in tool_registry.tools.values() if hasattr(t, '_is_custom')])))
    
    console.print(table)

def _show_crews_stats(master_agent, detailed: bool, days: int):
    """Show crew-specific statistics."""
    crews = list(master_agent.crew_designer._crews_cache.values())
    
    if not crews:
        console.print("[yellow]No crews found[/yellow]")
        return
    
    table = Table(title="Crew Statistics")
    table.add_column("Crew Name", style="cyan")
    table.add_column("Agents", style="green")
    table.add_column("Task Type", style="blue")
    
    for crew in crews:
        # Determine task type based on keywords
        task_lower = crew.task.lower()
        if any(word in task_lower for word in ['research', 'analyze', 'find']):
            task_type = "Research"
        elif any(word in task_lower for word in ['write', 'create', 'generate']):
            task_type = "Content"
        elif any(word in task_lower for word in ['build', 'develop', 'code']):
            task_type = "Development"
        else:
            task_type = "General"
        
        table.add_row(crew.name, str(len(crew.agents)), task_type)
    
    console.print(table)

def _show_agents_stats(master_agent, detailed: bool, days: int):
    """Show agent-specific statistics."""
    crews = list(master_agent.crew_designer._crews_cache.values())
    all_agents = []
    
    for crew in crews:
        for agent in crew.agents:
            all_agents.append((agent, crew.name))
    
    if not all_agents:
        console.print("[yellow]No agents found[/yellow]")
        return
    
    # Role distribution
    role_counts = {}
    for agent, _ in all_agents:
        role = agent.role
        role_counts[role] = role_counts.get(role, 0) + 1
    
    table = Table(title="Agent Statistics")
    table.add_column("Role", style="cyan")
    table.add_column("Count", style="green")
    
    for role, count in sorted(role_counts.items()):
        table.add_row(role.title(), str(count))
    
    console.print(table)
    
    if detailed:
        console.print(f"\n[bold blue]All Agents:[/bold blue]")
        agent_table = Table()
        agent_table.add_column("Name", style="cyan")
        agent_table.add_column("Role", style="green")
        agent_table.add_column("Crew", style="blue")
        agent_table.add_column("Tools", style="yellow")
        
        for agent, crew_name in all_agents:
            tools = ", ".join(getattr(agent, 'required_tools', [])[:3])  # Show first 3 tools
            if len(getattr(agent, 'required_tools', [])) > 3:
                tools += "..."
            
            agent_table.add_row(agent.name, agent.role, crew_name, tools)
        
        console.print(agent_table)

def _show_tools_stats(master_agent, detailed: bool):
    """Show tool usage statistics."""
    from .tools.registry import ToolRegistry
    tool_registry = ToolRegistry()
    
    # Count tool usage across all agents
    tool_usage = {}
    crews = list(master_agent.crew_designer._crews_cache.values())
    
    for crew in crews:
        for agent in crew.agents:
            for tool in getattr(agent, 'required_tools', []):
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
    
    table = Table(title="Tool Statistics")
    table.add_column("Tool", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Usage Count", style="blue")
    
    # Show tools sorted by usage
    for tool_name in sorted(tool_usage.keys(), key=lambda x: tool_usage[x], reverse=True):
        tool_instance = tool_registry.tools.get(tool_name)
        category = tool_instance.category if tool_instance else "unknown"
        table.add_row(tool_name, category.title(), str(tool_usage[tool_name]))
    
    console.print(table)
    
    if detailed:
        console.print(f"\n[bold blue]All Available Tools:[/bold blue]")
        all_tools_table = Table()
        all_tools_table.add_column("Tool", style="cyan")
        all_tools_table.add_column("Category", style="green")
        all_tools_table.add_column("Description", style="blue")
        
        for tool_name, tool_instance in sorted(tool_registry.tools.items()):
            all_tools_table.add_row(
                tool_name,
                tool_instance.category.title(),
                tool_instance.description[:50] + "..." if len(tool_instance.description) > 50 else tool_instance.description
            )
        
        console.print(all_tools_table)

def _perform_reset(keep_custom_tools: bool = False):
    """Perform the actual reset operation."""
    import os
    import shutil
    from .database.database import Database
    
    # Reset the database (crews, agents, execution logs)
    console.print("[dim]Resetting database...[/dim]")
    try:
        config = Config()
        database = Database(config.database.url)
        database.reset_database()
        console.print("[green]Database reset successfully[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not reset database: {e}[/yellow]")
    
    removed_count = 0
    
    # Remove the temporary cache file used by crew designer
    cache_file = "/tmp/crewmaster_cache.pkl"
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            console.print(f"[dim]Removed cache file: crewmaster_cache.pkl[/dim]")
            removed_count += 1
        except Exception as e:
            console.print(f"[yellow]Warning: Could not remove cache file: {e}[/yellow]")
    
    # Clean up additional files
    home_dir = os.path.expanduser("~")
    crewmaster_dir = os.path.join(home_dir, ".crewmaster")
    
    if os.path.exists(crewmaster_dir):
        # Always clean these additional files
        additional_files = [
            os.path.join(crewmaster_dir, "execution_history.json"),
            os.path.join(crewmaster_dir, "performance_data.json"),
            os.path.join(crewmaster_dir, "crew_history"),  # directory
        ]
        
        for path in additional_files:
            if os.path.exists(path):
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        console.print(f"[dim]Removed file: {os.path.basename(path)}[/dim]")
                        removed_count += 1
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        console.print(f"[dim]Removed directory: {os.path.basename(path)}[/dim]")
                        removed_count += 1
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not remove {os.path.basename(path)}: {e}[/yellow]")
        
        # Conditionally clean custom tools
        if not keep_custom_tools:
            custom_tools_file = os.path.join(crewmaster_dir, "custom_tools.json")
            tools_config_file = os.path.join(crewmaster_dir, "tools_config.json")
            
            for tools_file in [custom_tools_file, tools_config_file]:
                if os.path.exists(tools_file):
                    try:
                        os.remove(tools_file)
                        console.print(f"[dim]Removed file: {os.path.basename(tools_file)}[/dim]")
                        removed_count += 1
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not remove {os.path.basename(tools_file)}: {e}[/yellow]")
    
    console.print(f"[green]Reset completed - Database reset + {removed_count} additional files cleaned[/green]")

@app.command()
def cache_clear():
    """Clear all cached task analysis results.
    
    This will remove all cached analysis data, forcing fresh analysis
    for all future tasks. Use this to ensure you get the latest analysis
    results or to free up cache storage space.
    
    Example: crewmaster cache-clear
    """
    console.print("\n[bold yellow]🗑️  Clearing analysis cache...[/bold yellow]")
    
    try:
        config = Config()
        from .core.master_agent_crew import MasterAgentCrew
        master_agent = MasterAgentCrew(config)
        
        result = master_agent.clear_analysis_cache()
        
        console.print(f"[green]✅ {result['message']}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error clearing cache: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def cache_stats():
    """Show statistics about the analysis cache.
    
    Displays information about cached task analyses including:
    - Total number of cached entries
    - Number of valid vs expired entries
    - Cache file location
    
    Example: crewmaster cache-stats
    """
    console.print("\n[bold blue]📊 Analysis Cache Statistics[/bold blue]")
    
    try:
        config = Config()
        from .core.master_agent_crew import MasterAgentCrew
        master_agent = MasterAgentCrew(config)
        
        stats = master_agent.get_cache_stats()
        
        # Create stats table
        table = Table(title="Cache Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Entries", str(stats['total_entries']))
        table.add_row("Valid Entries", str(stats['valid_entries']))
        table.add_row("Expired Entries", str(stats['expired_entries']))
        table.add_row("Cache File", stats['cache_file'])
        
        console.print(table)
        
        if stats['expired_entries'] > 0:
            console.print(f"\n[dim]💡 Use 'crewmaster cache-clean' to remove {stats['expired_entries']} expired entries[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Error getting cache stats: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def cache_list():
    """List all cached task analyses with details.
    
    Shows a detailed list of all cached task analyses including:
    - Task descriptions
    - Cache age and expiry status
    - Analysis complexity and agent count
    
    Example: crewmaster cache-list
    """
    console.print("\n[bold blue]📝 Cached Task Analyses[/bold blue]")
    
    try:
        config = Config()
        from .core.master_agent_crew import MasterAgentCrew
        master_agent = MasterAgentCrew(config)
        
        cached_tasks = master_agent.list_cached_tasks()
        
        if not cached_tasks:
            console.print("[dim]No cached analyses found.[/dim]")
            return
        
        # Create cached tasks table
        table = Table(title=f"Cached Tasks ({len(cached_tasks)} entries)")
        table.add_column("Task Description", style="white", max_width=60)
        table.add_column("Age", style="yellow") 
        table.add_column("Complexity", style="blue")
        table.add_column("Agents", style="green")
        table.add_column("Status", style="cyan")
        
        for task in cached_tasks:
            description = task['task_description']
            if len(description) > 57:
                description = description[:54] + "..."
            
            age_str = f"{task['age_hours']}h"
            if task['age_hours'] > 24:
                age_str = f"{task['age_hours']/24:.1f}d"
            
            status = "[red]Expired[/red]" if task['is_expired'] else "[green]Valid[/green]"
            
            table.add_row(
                description,
                age_str,
                task['complexity'].title(),
                str(task['agent_count']),
                status
            )
        
        console.print(table)
        
        expired_count = sum(1 for task in cached_tasks if task['is_expired'])
        if expired_count > 0:
            console.print(f"\n[dim]💡 Use 'crewmaster cache-clean' to remove {expired_count} expired entries[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Error listing cache: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def cache_clean():
    """Remove expired cache entries while keeping valid ones.
    
    This cleans up the cache by removing only expired entries (older than 24 hours)
    while preserving recent analysis results. This is a gentler alternative to
    cache-clear that maintains useful cached data.
    
    Example: crewmaster cache-clean
    """
    console.print("\n[bold yellow]🧹 Cleaning expired cache entries...[/bold yellow]")
    
    try:
        config = Config()
        from .core.master_agent_crew import MasterAgentCrew
        master_agent = MasterAgentCrew(config)
        
        result = master_agent.clear_expired_cache()
        
        if result['cleared_entries'] > 0:
            console.print(f"[green]✅ {result['message']}[/green]")
        else:
            console.print("[dim]No expired cache entries found.[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Error cleaning cache: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def version():
    """Show CrewMaster version."""
    from . import __version__
    console.print(f"[bold green]CrewMaster[/bold green] version [cyan]{__version__}[/cyan]")

def main():
    """Main CLI entry point."""
    app()

if __name__ == "__main__":
    main()