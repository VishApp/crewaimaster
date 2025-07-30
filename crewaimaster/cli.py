"""
CrewAIMaster CLI - Simplified command line interface for file-based crew management.
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
from typing import Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

from .core.config import Config
from .core.llm_provider import LLMProviderFactory
from .core.file_generator import CrewFileGenerator
from .core.master_agent_crew import MasterAgentCrew
from .core.task_analyzer import CrewSpec, AgentSpec
from .core.crew_designer import CrewModel

app = typer.Typer(
    name="crewaimaster",
    help="""[bold cyan]CrewAIMaster: Build intelligent multi-agent systems using CrewAI[/bold cyan]

[green]🎯 Quick Start:[/green]
  [cyan]crewaimaster create[/cyan] "Create a blog writer who can write simple and informative blog posts for beginners." --name blog_writer_01 # CREATE
  [cyan]crewaimaster run[/cyan] blog_writer_01 --input "Write a blog post about the benefits of AI" # EXECUTE
""",
    rich_markup_mode="rich"
)

console = Console()

@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    """Main callback that shows banner when no command is provided."""
    if ctx.invoked_subcommand is None:
        display_banner()

def display_banner():
    """Display CrewAIMaster banner."""
    banner = """[bold cyan]
                                                                        
[blink]╔═╗╦═╗╔═╗╦ ╦[/blink]  [bold blue]╔═╗╦  ╔╦╗╔═╗╔═╗╔╦╗╔═╗╦═╗[/bold blue]                
[blink]║  ╠╦╝║╣ ║║║[/blink]  [bold blue]╠═╣║  ║║║╠═╣╚═╗ ║ ║╣ ╠╦╝[/bold blue]                
[blink]╚═╝╩╚═╚═╝╚╩╝[/blink]  [bold blue]╩ ╩╩  ╩ ╩╩ ╩╚═╝ ╩ ╚═╝╩╚═[/bold blue]                
                                                                        
[bright_green]🤖 Build intelligent multi-agent systems[/bright_green]           
                                                                                                                         
"""

    console.print(banner)
    
    console.print(f"\n[bold yellow]🎯 Getting Started[/bold yellow]")
    console.print("=" * 60)
    
    console.print("\n[bold green]Step 1:[/bold green] Create your first crew")
    console.print("  [cyan]crewaimaster create \"A blog writer who can write simple and informative blog posts for beginners.\" --name blog_writer_01[/cyan]")
    console.print("  [dim]📁 Generates: YAML configs, Python modules, documentation[/dim]")
    
    console.print("\n[bold green]Step 2:[/bold green] Run your crew (requires API key)")
    console.print("  [cyan]export OPENAI_API_KEY=\"your-key\"[/cyan]  # OpenAI")
    console.print("  [cyan]export ANTHROPIC_API_KEY=\"your-key\"[/cyan]  # Anthropic")
    
    console.print("\n[bold green]Step 3:[/bold green] Work with generated files")
    console.print("  [cyan]crewaimaster run blog_writer_01 --input \"Write a blog post about the benefits of AI\"[/cyan]")
    console.print("  [cyan]cd crews/blog_writer_01 && ./run.sh 'your input'[/cyan] # Alternative execution")
    console.print("  [dim]🔄 Version control friendly - track changes in Git[/dim]")
    
    console.print("\n[dim]💡 Essential Commands:[/dim]")
    console.print("[dim]  • crewaimaster create \"task\" - Create new crew[/dim]")
    console.print("[dim]  • crewaimaster run <name> - Execute crew[/dim]")
    console.print("[dim]  • crewaimaster providers - Configure LLM providers[/dim]")
    console.print("[dim]  • crewaimaster version - Show version[/dim]")

@app.command()
def create(
    task: str = typer.Argument(..., help="Description of the task to accomplish"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Optional name for the crew"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Create a new crew for a given task.
    
    Creates a self-contained CrewAI project with YAML configurations and Python modules.
    
    Examples:
      crewaimaster create "research competitors and write analysis report"
      crewaimaster create "analyze cryptocurrency market trends" --name crypto_crew
    """
    if verbose:
        display_banner()
    
    console.print(f"\n[bold green]🚀 Creating crew for task:[/bold green] {task}")
    
    try:
        config = Config()
        
        # Use master agents to analyze the task and generate crew specification
        console.print("[dim]🤖 Using AI master agents to analyze task and design crew...[/dim]")
        
        master_crew = MasterAgentCrew(config)
        crew_model = master_crew.create_crew(task, crew_name=name, verbose=verbose, use_ai_orchestration=True)
        
        # Convert CrewModel to CrewSpec for file generation compatibility
        from .core.task_analyzer import TaskComplexity
        crew_spec = CrewSpec(
            name=crew_model.name,
            task=crew_model.task,
            description=crew_model.description,
            agents=[
                AgentSpec(
                    role=agent.role,
                    name=agent.name,
                    goal=agent.goal,
                    backstory=agent.backstory,
                    required_tools=getattr(agent, 'required_tools', []) or ['WebsiteSearchTool', 'FileReadTool'],
                    memory_type=getattr(agent, 'memory_type', 'short_term'),
                    max_iter=getattr(agent, 'max_iter', 5),
                    allow_delegation=getattr(agent, 'allow_delegation', False)
                )
                for agent in crew_model.agents
            ],
            expected_output=getattr(crew_model, 'expected_output', f"Complete results for: {crew_model.task}"),
            complexity=TaskComplexity.MODERATE,  # Default since master agents determine this
            estimated_time=15,  # Default
            process_type="sequential"
        )
        
        if not crew_spec:
            console.print("[red]❌ Failed to create crew using master agents[/red]")
            raise typer.Exit(1)
        
        # Generate file-based crew
        console.print("[dim]📁 Generating file-based crew structure...[/dim]")
        file_generator = CrewFileGenerator()
        crew_path = file_generator.generate_crew_project(crew_spec)
        
        console.print(f"\n[bold green]✅ Created File-Based Crew:[/bold green] {crew_spec.name}")
        console.print(f"[bold blue]📁 Crew Path:[/bold blue] {crew_path}")
        
        # Display agents summary
        console.print(f"\n[bold blue]👥 Generated Agents ({len(crew_spec.agents)}):[/bold blue]")
        for i, agent in enumerate(crew_spec.agents, 1):
            console.print(f"  {i}. [green]{agent.name}[/green] - {agent.role}")
        
        console.print(f"\n[bold green]📁 Generated Files:[/bold green]")
        console.print(f"  [cyan]•[/cyan] config/agents.yaml - Agent configurations")
        console.print(f"  [cyan]•[/cyan] config/tasks.yaml - Task definitions")
        console.print(f"  [cyan]•[/cyan] src/{crew_spec.name}/crew.py - Main crew logic")
        console.print(f"  [cyan]•[/cyan] src/{crew_spec.name}/main.py - Entry point")
        console.print(f"  [cyan]•[/cyan] requirements.txt - Dependencies")
        console.print(f"  [cyan]•[/cyan] run.sh - Execution script")
        console.print(f"  [cyan]•[/cyan] README.md - Documentation")
        
        console.print(f"\n[dim]💡 Next steps:[/dim]")
        console.print(f"[dim]  • crewaimaster run {crew_spec.name} --input \"your input\"[/dim]")
        console.print(f"[dim]  • cd {crew_path} && ./run.sh \"your input\"[/dim]")

    except Exception as e:
        console.print(f"\n[bold red]❌ Error creating crew:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command()
def run(
    crew_name: str = typer.Argument(..., help="Name of the crew to run"),
    input_data: Optional[str] = typer.Option(None, "--input", "-i", help="Additional input data for the task"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Execute an existing crew to perform the task.
    
    Runs the crew in its generated project directory.
    
    Requirements:
    - OpenAI/Anthropic API key: export OPENAI_API_KEY="your-key"
    - Existing crew (created with 'crewaimaster create')
    
    Example: crewaimaster run my_research_crew --input "focus on recent data"
    """
    console.print(f"\n[bold green]🏃 Running crew:[/bold green] {crew_name}")
    if input_data:
        console.print(f"[bold blue]📝 With additional context:[/bold blue] {input_data}")
    
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        # Find crew directory
        crews_base_path = Path("crews")
        crew_path = crews_base_path / crew_name
        
        if not crew_path.exists():
            console.print(f"[red]❌ Crew '{crew_name}' not found at {crew_path}[/red]")
            console.print("[dim]Available crews:[/dim]")
            if crews_base_path.exists():
                for crew_dir in crews_base_path.iterdir():
                    if crew_dir.is_dir():
                        console.print(f"[dim]  • {crew_dir.name}[/dim]")
            else:
                console.print("[dim]  No crews directory found[/dim]")
            raise typer.Exit(1)
        
        # Check if main.py exists
        main_py_path = crew_path / "src" / crew_name / "main.py"
        if not main_py_path.exists():
            console.print(f"[red]❌ Crew main.py not found at {main_py_path}[/red]")
            raise typer.Exit(1)
        
        console.print(f"[dim]📁 Executing crew from: {crew_path}[/dim]")
        
        # Build command
        cmd = [sys.executable, "-m", f"{crew_name}.main"]
        if input_data:
            cmd.append(input_data)
        
        # Execute the crew
        result = subprocess.run(
            cmd,
            cwd=crew_path / "src",
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            console.print(f"\n[bold green]✅ Crew execution completed![/bold green]")
            console.print(Panel(result.stdout, title="📄 Result", border_style="green"))
        else:
            console.print(f"\n[bold red]❌ Crew execution failed![/bold red]")
            console.print(Panel(result.stderr, title="💥 Error", border_style="red"))
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error running crew:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command()
def providers(
    configure: Optional[str] = typer.Option(None, "--configure", "-c", help="Configure provider (openai, anthropic, google, deepseek, custom)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for provider"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Base URL for provider endpoint (optional for standard providers)"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name for provider")
):
    """Show available LLM providers and configuration examples.
    
    Lists all supported providers (OpenAI, Anthropic, Google, DeepSeek, Custom)
    with their required environment variables and example configurations.
    """
    
    # Handle configuration if requested
    if configure:
        if not api_key or not model:
            console.print("[red]❌ For provider configuration, both --api-key and --model are required[/red]")
            raise typer.Exit(1)
        
        provider_name = configure.lower()
        supported_providers = ["openai", "anthropic", "google", "deepseek", "custom"]
        
        if provider_name not in supported_providers:
            console.print(f"[red]❌ Unsupported provider: {configure}[/red]")
            console.print(f"[yellow]Supported providers: {', '.join(supported_providers)}[/yellow]")
            raise typer.Exit(1)
        
        config = Config()
        
        # Set default base URLs for standard providers if not provided
        if base_url is None:
            default_base_urls = {
                "openai": "https://api.openai.com/v1",
                "anthropic": "https://api.anthropic.com/v1", 
                "google": "https://generativelanguage.googleapis.com/v1beta",
                "deepseek": "https://api.deepseek.com/v1"
            }
            if provider_name == "custom":
                console.print("[red]❌ Custom provider requires --base-url parameter[/red]")
                raise typer.Exit(1)
            else:
                base_url = default_base_urls.get(provider_name)
        
        # Configure the provider
        config._config.llm.provider = provider_name
        config._config.llm.api_key = api_key
        config._config.llm.base_url = base_url
        config._config.llm.model = model
        config.save_config()
        
        console.print(f"[green]✅ {provider_name.title()} provider configured successfully![/green]")
        console.print(f"[dim]Provider: {provider_name}[/dim]")
        console.print(f"[dim]Base URL: {base_url}[/dim]")
        console.print(f"[dim]Model: {model}[/dim]")
        console.print(f"[dim]Config saved to: {config.config_path}[/dim]")
        return
    
    console.print("\n[bold blue]🔧 Available LLM Providers[/bold blue]")
    
    try:        
        console.print(f"\n[bold green]🔧 CLI Configuration (All Providers):[/bold green]")
        console.print("[bold]OpenAI:[/bold]")
        console.print("• [cyan]crewaimaster providers --configure openai --api-key \"your-openai-key\" --model \"gpt-4\"[/cyan]")
        console.print()
        console.print("[bold]Anthropic:[/bold]")
        console.print("• [cyan]crewaimaster providers --configure anthropic --api-key \"your-anthropic-key\" --model \"claude-3-sonnet-20240229\"[/cyan]")
        console.print()
        console.print("[bold]Google:[/bold]")
        console.print("• [cyan]crewaimaster providers --configure google --api-key \"your-google-key\" --model \"gemini-pro\"[/cyan]")
        console.print()
        console.print("[bold]DeepSeek:[/bold]")
        console.print("• [cyan]crewaimaster providers --configure deepseek --api-key \"your-deepseek-key\" --model \"deepseek-chat\"[/cyan]")
        console.print()
        console.print("[bold]Custom Provider:[/bold]")
        console.print("• [cyan]crewaimaster providers --configure custom --api-key \"your-key\" --base-url \"https://api.example.com/v1\" --model \"gpt-4o-mini\"[/cyan]")
        
        config = Config()
        console.print(f"\n[dim]💡 Current provider: {config.llm.provider}[/dim]")
        console.print(f"[dim]💡 Current model: {config.llm.model}[/dim]")
        console.print(f"[dim]💡 Edit advanced settings in: {config.config_path}[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Error showing providers: {str(e)}[/red]")

@app.command()
def version():
    """Show CrewAIMaster version."""
    try:
        from . import __version__
        console.print(f"[bold green]CrewAIMaster[/bold green] version [cyan]{__version__}[/cyan]")
    except ImportError:
        console.print(f"[bold green]CrewAIMaster[/bold green] version [cyan]1.0.0[/cyan]")

def main():
    """Main CLI entry point."""
    app()

if __name__ == "__main__":
    main()