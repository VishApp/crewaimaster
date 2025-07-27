# CrewMaster

**A Python package for building intelligent multi-agent systems using CrewAI**

CrewMaster is an advanced framework that automatically generates, manages, and executes multi-agent crews based on natural language task descriptions. It provides a CLI interface and comprehensive backend system for creating intelligent AI agents with memory, knowledge base access, tools, and safety guardrails.

## 🚀 Features

- **🎯 One-Command Crew Creation**: `crewmaster create "your task description"`
- **🧠 Intelligent Task Analysis**: Automatically analyzes tasks and designs optimal agent crews
- **🔄 Agent Reusability**: Smart agent matching and reuse across different tasks
- **📚 Knowledge Base Integration**: Support for documents, URLs, and structured data
- **🛡️ Safety Guardrails**: Built-in PII detection, toxicity filtering, and code safety
- **🗄️ Persistent Storage**: SQLite/PostgreSQL database for crews, agents, and execution history
- **🛠️ Extensible Tools**: Modular tool system with built-in and custom tool support
- **📊 Performance Tracking**: Detailed metrics and execution logging

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

### 1. Create Your First Crew

```bash
# Create a crew for any task
crewmaster create "Write a comprehensive market analysis report for electric vehicles in 2024"

# Output:
# ✅ Created Crew: electric_vehicles_2024_crew
# 🆔 Crew ID: abc123...
# 
# 👥 Agents:
# ├── market_researcher (Research Specialist)
# ├── data_analyst (Data Analysis Expert)  
# └── report_writer (Technical Writer)
```

### 2. Execute the Crew

```bash
# Run the crew
crewmaster run abc123

# Output:
# 🏃 Running crew: electric_vehicles_2024_crew
# ✅ Crew execution completed!
# 📄 Result: [Comprehensive market analysis report content...]
```

### 3. Manage Your Crews

```bash
# List all crews
crewmaster list

# Inspect a specific crew
crewmaster inspect abc123

# View execution history
crewmaster history abc123
```

## 🎯 Use Cases

CrewMaster excels at automating complex, multi-step tasks that benefit from specialized agent collaboration:

### 📊 Research & Analysis
```bash
crewmaster create "Research the top 10 AI startups, analyze their funding, and create a competitive landscape report"
```

### 💻 Software Development
```bash
crewmaster create "Build a REST API for a todo app with authentication, database integration, and comprehensive tests"
```

### 📝 Content Creation
```bash
crewmaster create "Write a series of blog posts about sustainable energy, including SEO optimization and social media promotion"
```

### 📈 Business Intelligence
```bash
crewmaster create "Analyze our quarterly sales data, identify trends, and prepare an executive presentation with recommendations"
```

## 🏗️ Architecture

CrewMaster follows a modular architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Interface │    │   Master Agent   │    │  Task Analyzer  │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • crewmaster    │───▶│ • Orchestration  │───▶│ • NLP Analysis  │
│   create        │    │ • Execution      │    │ • Agent Design  │
│ • crewmaster    │    │ • Monitoring     │    │ • Tool Selection│
│   run           │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Database      │    │  Crew Designer   │    │ Knowledge Base  │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Agents        │◀───│ • Agent Creation │───▶│ • Document RAG  │
│ • Crews         │    │ • Crew Assembly  │    │ • Vector Search │
│ • Execution     │    │ • Reuse Logic    │    │ • Embeddings    │
│   Logs          │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                   ┌──────────────────┐    ┌─────────────────┐
                   │  Tool Registry   │    │   Guardrails    │
                   ├──────────────────┤    ├─────────────────┤
                   │ • Web Search     │    │ • PII Detection │
                   │ • File Ops       │    │ • Safety Checks │
                   │ • Code Exec      │    │ • Quality Gates │
                   │ • Custom Tools   │    │                 │
                   └──────────────────┘    └─────────────────┘
```

## 🛠️ Advanced Configuration

### Environment Variables

```bash
# LLM Configuration
export CREWMASTER_LLM_MODEL="gpt-4"
export CREWMASTER_LLM_API_KEY="your-openai-key"
export OPENAI_API_KEY="your-openai-key"

# Database Configuration  
export CREWMASTER_DATABASE_URL="postgresql://user:pass@localhost/crewmaster"

# Tools Configuration
export SERPER_API_KEY="your-serper-key"  # For web search
```

### Configuration File

Create `~/.crewmaster/config.yaml`:

```yaml
# LLM Settings
llm:
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2000

# Database Settings
database:
  url: "sqlite:///~/.crewmaster/crewmaster.db"
  
# Agent Settings
max_agents_per_crew: 5
default_agent_verbose: true
default_process: "sequential"

# Tool Settings
tools:
  enabled_categories: ["web_search", "file_ops", "code_exec"]
  max_tools_per_agent: 3

# Memory Settings
memory:
  enabled: true
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
```

## 🧠 Knowledge Base

Add knowledge sources to enhance agent capabilities:

```bash
# Add documents
crewmaster kb create "company_docs" --description "Internal company documentation"
crewmaster kb add company_docs --file "./docs/employee_handbook.pdf"
crewmaster kb add company_docs --url "https://company.com/policies"

# Search knowledge base
crewmaster kb search company_docs "vacation policy"
```

## 🛡️ Safety & Guardrails

CrewMaster includes built-in safety measures:

- **PII Detection**: Automatically detects and blocks personal information
- **Toxicity Filtering**: Prevents harmful or offensive content
- **Code Safety**: Scans for potentially dangerous code patterns
- **Output Validation**: Ensures appropriate response length and quality
- **Hallucination Detection**: Identifies potential AI hallucinations

## 🔌 Custom Tools

Extend CrewMaster with custom tools:

```python
# custom_tools.py
from crewmaster.tools.registry import ToolRegistry

def my_custom_tool(input_data: str) -> str:
    # Your custom logic here
    return f"Processed: {input_data}"

# Register the tool
registry = ToolRegistry()
registry.register_custom_tool(
    name="my_tool",
    description="My custom processing tool",
    category="custom",
    tool_factory=lambda: my_custom_tool
)
```

## 📊 Performance & Monitoring

Track crew performance and resource usage:

```bash
# View system stats
crewmaster stats

# Export crew configuration
crewmaster export abc123 --output crew_config.json

# View detailed performance metrics
crewmaster performance abc123
```

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
git clone https://github.com/yourusername/crewmaster
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

## 🐛 Troubleshooting

### Common Issues

**Issue**: "CrewAI instance not found"
```bash
# Solution: Recreate the crew
crewmaster delete abc123
crewmaster create "your task description"
```

**Issue**: Database connection errors
```bash
# Solution: Reset database
crewmaster db reset
```

**Issue**: Tool import errors
```bash
# Solution: Install missing dependencies
pip install -r requirements.txt
```

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