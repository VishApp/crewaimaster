# CrewMaster

**A Python package for building intelligent multi-agent systems using CrewAI**

CrewMaster is an advanced framework that automatically generates, manages, and executes multi-agent crews based on natural language task descriptions. It provides a CLI interface and comprehensive backend system for creating intelligent AI agents with memory, tools, and safety guardrails.

## 🚀 Key Features

### 🤖 **AI-Powered Orchestration**
- **Intelligent Crew Creation**: AI agents analyze tasks and design optimal multi-agent crews
- **Advanced Task Analysis**: NLP-powered understanding of requirements and complexity
- **Smart Agent Design**: Automatic role assignment and tool selection based on task needs
- **Performance Optimization**: AI-driven optimization of crew composition and execution

### 🎯 **Developer Experience**
- **One-Command Creation**: `crewmaster create "your task description"`
- **Rich CLI Interface**: Beautiful terminal UI with progress indicators and feedback
- **Flexible Configuration**: Environment variables, YAML configs, and CLI options
- **Extensible Architecture**: Plugin system for custom agents, tools, and workflows

## 📦 Installation

```bash
# Install from source (recommended for development)
git clone https://github.com/yourusername/crewmaster
cd crewmaster
pip install -e .

# Or install from PyPI (when available)
pip install crewmaster
```

### Dependencies

CrewMaster requires Python 3.9+ and the following packages:
- `crewai>=0.70.0` - Core multi-agent framework
- `click>=8.0.0` - CLI interface
- `sqlalchemy>=2.0.0` - Database ORM
- `sentence-transformers>=2.2.0` - Text embeddings
- `faiss-cpu>=1.7.0` - Vector search

## 🏃 Quick Start

### Prerequisites
```bash
# Install Python 3.9+
python --version

# Configure your LLM provider (see supported providers)
crewmaster providers

# Quick setup with OpenAI (most common)
crewmaster providers --configure openai --api-key "your-openai-key" --model "gpt-4"
```

### 1. Create Your First Crew with AI Orchestration

```bash
# Create an intelligent crew using AI analysis
crewmaster create "Write a comprehensive market analysis report for electric vehicles in 2024" --name electric_vehicles_market_analysis_crew

# Output with AI orchestration:
# 🤖 Using AI orchestration for intelligent crew creation
# 📋 AI orchestration completed:
#    Crew: electric_vehicles_market_analysis_crew
#    Agents: 3
#    Complexity: complex
# ✅ AI-orchestrated crew created with ID: electric_vehicles_market_analysis_crew
# 📊 Predicted performance: High accuracy with specialized research agents
# 
# 👥 Agents:
# ├── MarketResearcher (Research Specialist) - Tools: web_search, document_search, data_processing
# ├── DataAnalyst (Data Analysis Expert) - Tools: data_processing, api_calls, file_operations
# └── ReportWriter (Technical Writer) - Tools: file_operations, web_search
```

### 2. Execute the Crew

```bash
# Run the crew (requires configured LLM provider)
crewmaster run electric_vehicles_market_analysis_crew

# With additional context:
crewmaster run electric_vehicles_market_analysis_crew --input "Focus on Tesla, BMW, and Volkswagen specifically"

# Output:
# 🏃 Running crew: electric_vehicles_market_analysis_crew
# 📝 With additional context: Focus on Tesla, BMW, and Volkswagen specifically
# 🔍 MarketResearcher executing web search for current data...
# 📊 DataAnalyst processing market data and trends...
# ✍️ ReportWriter compiling comprehensive analysis...
# ✅ Crew execution completed in 45s!
# 📄 Result: [Comprehensive 2024 EV market analysis with specific focus on requested companies...]
```

### 3. Alternative Execution (Direct Script)

Generated crews can also be executed directly using environment variables:

```bash
# Navigate to the generated crew directory
cd crews/electric_vehicles_market_analysis_crew

# Run using standard environment variables
export OPENAI_API_KEY="your-openai-key"
./run.sh

# Or run using CrewMaster-specific environment variables
export CREWMASTER_LLM_PROVIDER="openai"
export CREWMASTER_LLM_MODEL="gpt-4"
export CREWMASTER_LLM_API_KEY="your-openai-key"
export CREWMASTER_LLM_BASE_URL="https://api.openai.com/v1"
python src/electric_vehicles_market_analysis_crew/main.py
```

## 🔄 Development Workflow

### Typical CrewMaster Workflow

```mermaid
flowchart LR
    A["`**1. Task Definition**
    Natural Language Task`"] --> B["`**2. AI Analysis**
    🤖 Task Complexity
    🎯 Agent Requirements
    🛠️ Tool Selection`"]
    
    B --> C["`**3. Crew Creation**
    👥 Agent Design
    🔧 Tool Assignment
    📋 Task Orchestration`"]
    
    C --> D["`**4. Execution**
    🏃 Multi-Agent Coordination
    🔄 Real-time Processing
    📊 Progress Monitoring`"]
    
    D --> E["`**5. Results & Analytics**
    📄 Output Generation
    📈 Performance Metrics
    💾 Persistent Storage`"]
    
    E --> F["`**6. Optimization**
    🔧 Crew Modification
    ⚡ Performance Tuning
    📤 Export/Backup`"]
    
    F --> G["`**7. Reuse & Scale**
    🔄 Crew Reusability
    📚 Knowledge Building
    🚀 Production Deployment`"]

    classDef stepStyle fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#333
    class A,B,C,D,E,F,G stepStyle
```

## 🏗️ Architecture

CrewMaster follows a layered, modular architecture with AI-powered orchestration:

```mermaid
graph TB
    %% User Interface & Configuration
    subgraph "🎯 User Interface & Configuration"
        CLI[CLI Interface<br/>• Rich Terminal UI<br/>• Command Processing<br/>• User Interactions]
        CONFIG[Configuration System<br/>• .crewmaster/config.yaml<br/>• Provider Management<br/>• Settings Persistence]
        PROVIDERS[LLM Providers<br/>• OpenAI, Anthropic, Google<br/>• DeepSeek, Custom<br/>• Auto Base URL Setup]
    end
    
    %% AI Orchestration Layer
    subgraph "🤖 AI Master Agent System"
        MAC[MasterAgentCrew<br/>• Main Orchestrator<br/>• AI-Powered Decisions<br/>• Intelligent Crew Creation]
        
        subgraph "AI Specialists"
            TAA[TaskAnalyzerAgent<br/>• NLP Task Analysis<br/>• Complexity Assessment<br/>• Requirements Extraction]
            ADA[AgentDesignerAgent<br/>• Role-Based Design<br/>• Tool Assignment<br/>• Agent Optimization]
            COA[CrewOrchestratorAgent<br/>• Crew Assembly<br/>• Process Selection<br/>• Performance Prediction]
        end
    end
    
    %% Core Engine
    subgraph "⚙️ Core Processing Engine"
        CD[CrewDesigner<br/>• File-Based Generation<br/>• CrewAI Integration<br/>• YAML Configuration]
        FG[FileGenerator<br/>• Project Structure<br/>• Python Modules<br/>• Documentation]
        LP[LLMProvider Factory<br/>• Provider Abstraction<br/>• Model Configuration<br/>• API Management]
    end
    
    %% File System & Generation
    subgraph "📁 File System & Generation"
        CREWS[Crews Directory<br/>• Generated Projects<br/>• YAML Configs<br/>• Python Modules]
        YAML[Configuration Files<br/>• agents.yaml<br/>• tasks.yaml<br/>• settings.yaml]
        PY[Python Modules<br/>• crew.py<br/>• main.py<br/>• __init__.py]
        SCRIPTS[Execution Scripts<br/>• run.sh<br/>• requirements.txt<br/>• README.md]
    end
    
    %% CrewAI Integration
    subgraph "🔄 CrewAI Execution Engine"
        CREWAI[CrewAI Framework<br/>• Multi-Agent Coordination<br/>• Task Execution<br/>• Process Management]
        AGENTS[Generated Agents<br/>• Specialized Roles<br/>• Tool Integration<br/>• Memory Management]
        TASKS[Agent Tasks<br/>• Sequential/Hierarchical<br/>• Context Sharing<br/>• Output Processing]
    end
    
    %% Tool Ecosystem
    subgraph "🛠️ Tool Ecosystem"
        TR[Tool Registry<br/>• Built-in Tools<br/>• CrewAI Tools<br/>• Tool Validation]
        TOOLS[Available Tools<br/>• SerperDevTool<br/>• FileReadTool<br/>• ScrapeWebsiteTool<br/>• CodeInterpreterTool<br/>• Custom Tools]
    end
    
    %% Data & Caching
    subgraph "💾 Data & Caching System"
        CACHE[Analysis Cache<br/>• Task Analysis Results<br/>• Performance Metrics<br/>• Temporary Data]
        STATS[Execution Stats<br/>• Performance Tracking<br/>• Success Rates<br/>• Usage Analytics]
    end
    
    %% User Flow
    CLI --> CONFIG
    CLI --> MAC
    CONFIG --> PROVIDERS
    PROVIDERS --> LP
    
    %% AI Orchestration Flow
    MAC --> TAA
    MAC --> ADA
    MAC --> COA
    MAC --> CD
    
    %% Core Processing Flow
    CD --> LP
    CD --> FG
    CD --> CREWAI
    
    %% File Generation Flow
    FG --> CREWS
    FG --> YAML
    FG --> PY
    FG --> SCRIPTS
    
    %% Execution Flow
    CREWAI --> AGENTS
    CREWAI --> TASKS
    AGENTS --> TOOLS
    TASKS --> TR
    
    %% Tool Integration
    TR --> TOOLS
    CD --> TR
    
    %% Data Flow
    MAC --> CACHE
    CREWAI --> STATS
    CD --> CACHE
    
    %% Configuration Flow
    CONFIG --> LP
    CONFIG --> CD
    
    %% Styling
    classDef uiLayer fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef aiLayer fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    classDef coreLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef fileLayer fill:#fff8e1,stroke:#ff8f00,stroke-width:2px,color:#000
    classDef execLayer fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#000
    classDef toolLayer fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef dataLayer fill:#f1f8e9,stroke:#33691e,stroke-width:2px,color:#000
    
    class CLI,CONFIG,PROVIDERS uiLayer
    class MAC,TAA,ADA,COA aiLayer
    class CD,FG,LP coreLayer
    class CREWS,YAML,PY,SCRIPTS fileLayer
    class CREWAI,AGENTS,TASKS execLayer
    class TR,TOOLS toolLayer
    class CACHE,STATS dataLayer
```

### 🏛️ Architecture Overview

CrewMaster's architecture is designed for scalability, modularity, and intelligent automation:

#### 🎯 **User Interface Layer**
- **CLI Interface**: Rich terminal experience with typer and rich libraries
- **Command Processing**: Handles user commands and provides interactive feedback
- **Input Validation**: Ensures commands are properly formatted and validated

#### 🤖 **AI Orchestration Layer** (Core Innovation)
- **MasterAgentCrew**: Main orchestrator using AI agents for intelligent decision-making
- **TaskAnalyzerAgent**: Advanced NLP analysis of user tasks and requirements
- **AgentDesignerAgent**: Intelligent design of agents based on task requirements
- **CrewOrchestratorAgent**: Optimizes crew composition and execution strategies

#### ⚙️ **Core Processing Layer**
- **CrewDesigner**: Handles CrewAI integration and agent instantiation
- **TaskAnalyzer**: Legacy fallback for task analysis with pattern matching

#### 🛠️ **Tool Ecosystem**
- **Tool Registry**: Centralized management of all available tools
- **Available Tools**: Comprehensive library of built-in and custom tools
- **Guardrails**: Safety and validation systems for secure operation

#### 🔄 **Execution Engine**
- **CrewAI Engine**: Core execution engine for running multi-agent crews
- **Agent Memory**: Sophisticated memory management for agent learning and context

### 🔄 Data Flow

1. **User Input** → CLI processes commands and validates input
2. **AI Analysis** → MasterAgentCrew analyzes task using specialized AI agents
3. **Crew Creation** → CrewDesigner instantiates agents with appropriate tools
4. **Execution** → CrewAI Engine runs the crew with real-time monitoring

## 🛠️ Configuration

### LLM Provider Setup

CrewMaster uses a `.crewmaster/config.yaml` configuration file for all settings. Environment variables are **no longer supported** - all configuration must be done via CLI commands or direct config file editing.

#### 📋 **View Available Providers**
```bash
# See all supported providers and configuration examples
crewmaster providers
```

#### 🚀 **CLI Configuration (All Providers)**

Configure any supported provider using the CLI:

**OpenAI:**
```bash
crewmaster providers --configure openai --api-key "your-openai-key" --model "gpt-4"
# Automatically sets base_url to https://api.openai.com/v1
```

**Anthropic:**
```bash
crewmaster providers --configure anthropic --api-key "your-anthropic-key" --model "claude-3-sonnet-20240229"
# Automatically sets base_url to https://api.anthropic.com/v1
```

**Google:**
```bash
crewmaster providers --configure google --api-key "your-google-key" --model "gemini-pro"
# Automatically sets base_url to https://generativelanguage.googleapis.com/v1beta
```

**DeepSeek:**
```bash
crewmaster providers --configure deepseek --api-key "your-deepseek-key" --model "deepseek-chat"
# Automatically sets base_url to https://api.deepseek.com/v1
```

**Custom Provider:**
```bash
crewmaster providers --configure custom --api-key "your-key" --base-url "https://api.example.com/v1" --model "gpt-4o-mini"
# Requires explicit base_url for custom endpoints
```

### 🔄 Configuration Management

**View Current Config:**
```bash
crewmaster providers  # Shows current provider settings and configuration examples
```

**Config File Location:** `.crewmaster/config.yaml` (in current directory)


#### Supported LLM Parameters

All standard LLM parameters are supported in per-agent configuration:

- **`model`** (string): Model name (e.g., "gpt-4", "claude-3-sonnet-20240229")
- **`temperature`** (float): Creativity level (0.0 to 1.0)
- **`max_tokens`** (int): Maximum response length
- **`top_p`** (float): Nucleus sampling parameter
- **`frequency_penalty`** (float): Penalty for frequent tokens
- **`presence_penalty`** (float): Penalty for repeated topics
- **`stop`** (list): Stop sequences to end generation
- **`timeout`** (int): Request timeout in seconds
- **`max_retries`** (int): Maximum retry attempts
- **`api_key`** (string): Agent-specific API key
- **`base_url`** (string): Agent-specific API endpoint
- **`api_version`** (string): API version for specific providers
- **`organization`** (string): Organization ID for OpenAI

#### Environment Variable Override

Generated crews support environment variable overrides with the following priority:

1. **CrewMaster Environment Variables** (highest priority)
   - `CREWMASTER_LLM_PROVIDER`
   - `CREWMASTER_LLM_MODEL`
   - `CREWMASTER_LLM_API_KEY`
   - `CREWMASTER_LLM_BASE_URL`

2. **Agent-Specific Configuration** (medium priority)
   - Values from `config/agents.yaml`

3. **Default Values** (lowest priority)
   - Fallback defaults

This allows for flexible deployment where you can override specific settings via environment variables while maintaining detailed per-agent configurations in your YAML files.

### Environment Variables for Generated Crews

**Important:** While CrewMaster CLI configuration doesn't use environment variables, the **generated crews** still support them for flexibility:

#### Standard Provider Environment Variables
```bash
# Standard provider environment variables (for generated crews)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
```

#### CrewMaster-Specific Environment Variables
```bash
# CrewMaster-specific environment variables (for generated crews)
export CREWMASTER_LLM_PROVIDER="openai"          # Provider selection
export CREWMASTER_LLM_MODEL="gpt-4"              # Model selection  
export CREWMASTER_LLM_API_KEY="your-api-key"     # API key
export CREWMASTER_LLM_BASE_URL="https://api.openai.com/v1"  # Base URL
```

#### Other Optional Environment Variables
```bash
# Web Search (optional)
export SERPER_API_KEY="your-serper-key"

```

## 🛡️ Safety & Guardrails

CrewMaster includes built-in safety measures:

- **PII Detection**: Automatically detects and blocks personal information
- **Toxicity Filtering**: Prevents harmful or offensive content
- **Code Safety**: Scans for potentially dangerous code patterns
- **Output Validation**: Ensures appropriate response length and quality
- **Hallucination Detection**: Identifies potential AI hallucinations

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/VishApp/crewmaster
cd crewmaster

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black crewmaster/
ruff check crewmaster/
```

## 📝 Documentation

- [User Guide](docs/user-guide.md) - Comprehensive usage guide
- [API Reference](docs/api-reference.md) - Complete API documentation
- [Developer Guide](docs/developer-guide.md) - Contributing and extending CrewMaster
- [Examples](examples/) - Real-world usage examples

## 📄 License

CrewMaster is released under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) - Core multi-agent framework
- [LangChain](https://github.com/langchain-ai/langchain) - LLM integration tools  
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) - Text embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search

## 🔗 Links

- [GitHub Repository](https://github.com/yourusername/crewmaster)
- [Documentation](https://crewmaster.readthedocs.io)
- [PyPI Package](https://pypi.org/project/crewmaster)
- [Discord Community](https://discord.gg/crewmaster)

---

**Built with ❤️ for the AI community**