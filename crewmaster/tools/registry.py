"""
Tool Registry for CrewMaster.

This module manages the registration and instantiation of tools
that can be used by agents.
"""

from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
import importlib
import json
import os

from crewai.tools import BaseTool
try:
    from langchain_community.tools import DuckDuckGoSearchRun
except ImportError:
    DuckDuckGoSearchRun = None

# Import official CrewAI tools - gracefully handle missing imports
CREWAI_TOOLS = {}
try:
    from crewai_tools import (
        FileReadTool, DirectoryReadTool, DirectorySearchTool, CodeDocsSearchTool,
        CSVSearchTool, DOCXSearchTool, TXTSearchTool, JSONSearchTool, MDXSearchTool,
        PDFSearchTool, PGSearchTool, RagTool, ScrapeElementFromWebsiteTool,
        ScrapeWebsiteTool, WebsiteSearchTool, XMLSearchTool, YoutubeChannelSearchTool,
        YoutubeVideoSearchTool, SerperDevTool, EXASearchTool, BrowserbaseLoadTool,
        GithubSearchTool, CodeInterpreterTool, FirecrawlSearchTool, FirecrawlCrawlWebsiteTool,
        FirecrawlScrapeWebsiteTool, LlamaIndexTool, ComposioTool, ApifyActorsTool
    )
    CREWAI_TOOLS.update({
        'FileReadTool': FileReadTool,
        'DirectoryReadTool': DirectoryReadTool,
        'DirectorySearchTool': DirectorySearchTool,
        'CodeDocsSearchTool': CodeDocsSearchTool,
        'CSVSearchTool': CSVSearchTool,
        'DOCXSearchTool': DOCXSearchTool,
        'TXTSearchTool': TXTSearchTool,
        'JSONSearchTool': JSONSearchTool,
        'MDXSearchTool': MDXSearchTool,
        'PDFSearchTool': PDFSearchTool,
        'PGSearchTool': PGSearchTool,
        'RagTool': RagTool,
        'ScrapeElementFromWebsiteTool': ScrapeElementFromWebsiteTool,
        'ScrapeWebsiteTool': ScrapeWebsiteTool,
        'WebsiteSearchTool': WebsiteSearchTool,
        'XMLSearchTool': XMLSearchTool,
        'YoutubeChannelSearchTool': YoutubeChannelSearchTool,
        'YoutubeVideoSearchTool': YoutubeVideoSearchTool,
        'SerperDevTool': SerperDevTool,
        'EXASearchTool': EXASearchTool,
        'BrowserbaseLoadTool': BrowserbaseLoadTool,
        'GithubSearchTool': GithubSearchTool,
        'CodeInterpreterTool': CodeInterpreterTool,
        'FirecrawlSearchTool': FirecrawlSearchTool,
        'FirecrawlCrawlWebsiteTool': FirecrawlCrawlWebsiteTool,
        'FirecrawlScrapeWebsiteTool': FirecrawlScrapeWebsiteTool,
        'LlamaIndexTool': LlamaIndexTool,
        'ComposioTool': ComposioTool,
        'ApifyActorsTool': ApifyActorsTool,
    })
except ImportError as e:
    print(f"Some CrewAI tools not available: {e}")

# Try to import additional tools
try:
    from crewai_tools import VisionTool, DALLEImageGeneratorTool
    CREWAI_TOOLS.update({
        'VisionTool': VisionTool,
        'DALLEImageGeneratorTool': DALLEImageGeneratorTool,
    })
except ImportError:
    pass

class ToolBase(ABC):
    """Base class for CrewMaster tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Tool category."""
        pass
    
    @abstractmethod
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get tool instance."""
        pass

class WebSearchTool(ToolBase):
    """Web search tool wrapper."""
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for current information using SerperDev API"
    
    @property
    def category(self) -> str:
        return "search"
    
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get web search tool instance."""
        import os
        
        # Try SerperDev first if API key is configured
        if 'SerperDevTool' in CREWAI_TOOLS and os.getenv('SERPER_API_KEY'):
            try:
                return CREWAI_TOOLS['SerperDevTool']()
            except Exception:
                pass
        
        # Fallback to DuckDuckGo (free) - this actually works!
        if DuckDuckGoSearchRun:
            try:
                return DuckDuckGoSearchRun()
            except Exception:
                pass
        
        # Return a mock tool if no search tools available
        return MockWebSearchTool()

class FileOperationsTool(ToolBase):
    """File operations tool wrapper."""
    
    @property
    def name(self) -> str:
        return "file_operations"
    
    @property
    def description(self) -> str:
        return "Read and write files using CrewAI FileReadTool"
    
    @property
    def category(self) -> str:
        return "file_ops"
    
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get file operations tool instance."""
        if 'FileReadTool' in CREWAI_TOOLS:
            try:
                return CREWAI_TOOLS['FileReadTool']()
            except Exception:
                pass
        return MockFileOperationsTool()

class CodeExecutionTool(ToolBase):
    """Code execution tool wrapper."""
    
    @property
    def name(self) -> str:
        return "code_execution"
    
    @property
    def description(self) -> str:
        return "Execute Python code safely using CrewAI CodeInterpreterTool"
    
    @property
    def category(self) -> str:
        return "development"
    
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get code execution tool instance."""
        if 'CodeInterpreterTool' in CREWAI_TOOLS:
            try:
                return CREWAI_TOOLS['CodeInterpreterTool']()
            except Exception:
                pass
        return MockCodeExecutionTool()

# New CrewAI Tools

class WebScrapingTool(ToolBase):
    """Web scraping tool using multiple CrewAI scraping tools."""
    
    @property
    def name(self) -> str:
        return "web_scraping"
    
    @property
    def description(self) -> str:
        return "Scrape websites and extract data using FireCrawl and other tools"
    
    @property
    def category(self) -> str:
        return "scraping"
    
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get web scraping tool instance."""
        # Try FireCrawl tools first
        for tool_name in ['FirecrawlScrapeWebsiteTool', 'ScrapeWebsiteTool']:
            if tool_name in CREWAI_TOOLS:
                try:
                    return CREWAI_TOOLS[tool_name]()
                except Exception:
                    continue
        return MockWebScrapingTool()

class DocumentSearchTool(ToolBase):
    """Document search across multiple formats."""
    
    @property
    def name(self) -> str:
        return "document_search"
    
    @property
    def description(self) -> str:
        return "Search within documents (PDF, DOCX, TXT, CSV, JSON, XML)"
    
    @property
    def category(self) -> str:
        return "search"
    
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get document search tool instance."""
        # Return the appropriate search tool based on file type
        file_type = config.get('file_type', 'pdf') if config else 'pdf'
        tool_mapping = {
            'pdf': 'PDFSearchTool',
            'docx': 'DOCXSearchTool', 
            'txt': 'TXTSearchTool',
            'csv': 'CSVSearchTool',
            'json': 'JSONSearchTool',
            'xml': 'XMLSearchTool',
            'md': 'MDXSearchTool'
        }
        
        tool_name = tool_mapping.get(file_type, 'PDFSearchTool')
        if tool_name in CREWAI_TOOLS:
            try:
                return CREWAI_TOOLS[tool_name]()
            except Exception:
                pass
        return MockDocumentSearchTool()

class GitHubTool(ToolBase):
    """GitHub repository search and analysis."""
    
    @property
    def name(self) -> str:
        return "github_search"
    
    @property
    def description(self) -> str:
        return "Search GitHub repositories and code"
    
    @property
    def category(self) -> str:
        return "development"
    
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get GitHub search tool instance."""
        if 'GithubSearchTool' in CREWAI_TOOLS:
            try:
                return CREWAI_TOOLS['GithubSearchTool']()
            except Exception:
                pass
        return MockGitHubTool()

class YouTubeTool(ToolBase):
    """YouTube video and channel analysis."""
    
    @property
    def name(self) -> str:
        return "youtube_search"
    
    @property
    def description(self) -> str:
        return "Search YouTube videos and channels"
    
    @property
    def category(self) -> str:
        return "content"
    
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get YouTube search tool instance."""
        search_type = config.get('search_type', 'video') if config else 'video'
        tool_name = 'YoutubeVideoSearchTool' if search_type == 'video' else 'YoutubeChannelSearchTool'
        
        if tool_name in CREWAI_TOOLS:
            try:
                return CREWAI_TOOLS[tool_name]()
            except Exception:
                pass
        return MockYouTubeTool()

class VisionTool(ToolBase):
    """Image analysis and generation tool."""
    
    @property
    def name(self) -> str:
        return "vision"
    
    @property
    def description(self) -> str:
        return "Analyze images and generate images using DALL-E"
    
    @property
    def category(self) -> str:
        return "ai_generation"
    
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get vision tool instance."""
        for tool_name in ['VisionTool', 'DALLEImageGeneratorTool']:
            if tool_name in CREWAI_TOOLS:
                try:
                    return CREWAI_TOOLS[tool_name]()
                except Exception:
                    continue
        return MockVisionTool()

class DatabaseTool(ToolBase):
    """Database search and query tool."""
    
    @property
    def name(self) -> str:
        return "database_search"
    
    @property
    def description(self) -> str:
        return "Search and query PostgreSQL databases"
    
    @property
    def category(self) -> str:
        return "database"
    
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get database tool instance."""
        if 'PGSearchTool' in CREWAI_TOOLS:
            try:
                return CREWAI_TOOLS['PGSearchTool']()
            except Exception:
                pass
        return MockDatabaseTool()

class BrowserAutomationTool(ToolBase):
    """Browser automation and data extraction."""
    
    @property
    def name(self) -> str:
        return "browser_automation"
    
    @property
    def description(self) -> str:
        return "Automate browsers and extract data using Browserbase"
    
    @property
    def category(self) -> str:
        return "automation"
    
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get browser automation tool instance."""
        if 'BrowserbaseLoadTool' in CREWAI_TOOLS:
            try:
                return CREWAI_TOOLS['BrowserbaseLoadTool']()
            except Exception:
                pass
        return MockBrowserAutomationTool()

class DataProcessingTool(ToolBase):
    """Data processing tool wrapper."""
    
    @property
    def name(self) -> str:
        return "data_processing"
    
    @property
    def description(self) -> str:
        return "Process and analyze data files"
    
    @property
    def category(self) -> str:
        return "data_processing"
    
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get data processing tool instance."""
        return MockDataProcessingTool()

class APICallsTool(ToolBase):
    """API calls tool wrapper."""
    
    @property
    def name(self) -> str:
        return "api_calls"
    
    @property
    def description(self) -> str:
        return "Make HTTP API calls"
    
    @property
    def category(self) -> str:
        return "api_calls"
    
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get API calls tool instance."""
        return MockAPICallsTool()

class EmailTool(ToolBase):
    """Email tool wrapper."""
    
    @property
    def name(self) -> str:
        return "email"
    
    @property
    def description(self) -> str:
        return "Send emails and notifications"
    
    @property
    def category(self) -> str:
        return "email"
    
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get email tool instance."""
        return MockEmailTool()

class SchedulingTool(ToolBase):
    """Scheduling tool wrapper."""
    
    @property
    def name(self) -> str:
        return "scheduling"
    
    @property
    def description(self) -> str:
        return "Schedule and manage tasks"
    
    @property
    def category(self) -> str:
        return "scheduling"
    
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get scheduling tool instance."""
        return MockSchedulingTool()

# Mock tools for development/testing
class MockWebSearchTool:
    """Mock web search tool for development."""
    
    def __call__(self, query: str) -> str:
        return f"Mock search results for: {query}"

class MockFileOperationsTool:
    """Mock file operations tool for development."""
    
    def __call__(self, file_path: str) -> str:
        return f"Mock file content for: {file_path}"

class MockCodeExecutionTool:
    """Mock code execution tool for development."""
    
    def __call__(self, code: str) -> str:
        return f"Mock execution result for code: {code[:50]}..."

class MockDataProcessingTool:
    """Mock data processing tool for development."""
    
    def __call__(self, data: str) -> str:
        return f"Mock data processing result for: {data[:50]}..."

class MockAPICallsTool:
    """Mock API calls tool for development."""
    
    def __call__(self, url: str, method: str = "GET", **kwargs) -> str:
        return f"Mock API call to {method} {url}"

class MockEmailTool:
    """Mock email tool for development."""
    
    def __call__(self, to: str, subject: str, body: str) -> str:
        return f"Mock email sent to {to} with subject: {subject}"

class MockSchedulingTool:
    """Mock scheduling tool for development."""
    
    def __call__(self, task: str, when: str) -> str:
        return f"Mock scheduled task: {task} for {when}"

class MockWebScrapingTool:
    """Mock web scraping tool for development."""
    
    def __call__(self, url: str) -> str:
        return f"Mock scraped content from: {url}"

class MockDocumentSearchTool:
    """Mock document search tool for development."""
    
    def __call__(self, query: str, document: str = "") -> str:
        return f"Mock document search for '{query}' in {document}"

class MockGitHubTool:
    """Mock GitHub tool for development."""
    
    def __call__(self, query: str, repo: str = "") -> str:
        return f"Mock GitHub search for '{query}' in {repo}"

class MockYouTubeTool:
    """Mock YouTube tool for development."""
    
    def __call__(self, query: str) -> str:
        return f"Mock YouTube search for: {query}"

class MockVisionTool:
    """Mock vision tool for development."""
    
    def __call__(self, image_input: str) -> str:
        return f"Mock vision analysis for: {image_input}"

class MockDatabaseTool:
    """Mock database tool for development."""
    
    def __call__(self, query: str) -> str:
        return f"Mock database query result for: {query}"

class MockBrowserAutomationTool:
    """Mock browser automation tool for development."""
    
    def __call__(self, action: str, target: str = "") -> str:
        return f"Mock browser {action} on {target}"

class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, ToolBase] = {}
        self.config = self._load_tool_config()
        self._register_default_tools()
        self._load_custom_tools()
    
    def _register_default_tools(self):
        """Register default tools."""
        default_tools = [
            # Core tools
            WebSearchTool(),
            FileOperationsTool(),
            CodeExecutionTool(),
            DataProcessingTool(),
            APICallsTool(),
            EmailTool(),
            SchedulingTool(),
            
            # New CrewAI tools
            WebScrapingTool(),
            DocumentSearchTool(),
            GitHubTool(),
            YouTubeTool(),
            VisionTool(),
            DatabaseTool(),
            BrowserAutomationTool()
        ]
        
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: ToolBase):
        """Register a new tool."""
        self.tools[tool.name] = tool
    
    def unregister_tool(self, tool_name: str):
        """Unregister a tool."""
        if tool_name in self.tools:
            del self.tools[tool_name]
    
    def get_tool(self, tool_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Get a tool instance by name."""
        if tool_name not in self.tools:
            return None
        
        return self.tools[tool_name].get_instance(config)
    
    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        """List available tools."""
        tools = []
        
        for tool in self.tools.values():
            if category is None or tool.category == category:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category
                })
        
        return tools
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tool names by category."""
        return [
            tool.name for tool in self.tools.values()
            if tool.category == category
        ]
    
    def register_custom_tool(self, name: str, description: str, category: str, 
                           tool_factory: Callable[[], Any]):
        """Register a custom tool with a factory function."""
        class CustomTool(ToolBase):
            @property
            def name(self) -> str:
                return name
            
            @property
            def description(self) -> str:
                return description
            
            @property
            def category(self) -> str:
                return category
            
            def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
                return tool_factory()
        
        self.register_tool(CustomTool())
    
    def load_tools_from_config(self, config_path: str):
        """Load custom tools from configuration file."""
        if not os.path.exists(config_path):
            return
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            for tool_config in config.get('custom_tools', []):
                self._load_custom_tool(tool_config)
        
        except Exception as e:
            print(f"Error loading tools from config: {e}")
    
    def _load_custom_tool(self, tool_config: Dict[str, Any]):
        """Load a single custom tool from configuration."""
        try:
            module_name = tool_config['module']
            class_name = tool_config['class']
            
            module = importlib.import_module(module_name)
            tool_class = getattr(module, class_name)
            
            # Instantiate and register the tool
            tool_instance = tool_class()
            self.register_tool(tool_instance)
            
        except Exception as e:
            print(f"Error loading custom tool {tool_config}: {e}")
    
    def get_recommended_tools(self, task_description: str) -> List[str]:
        """Get recommended tools based on task description."""
        task_lower = task_description.lower()
        recommended = []
        
        # Enhanced keyword-based recommendation including new tools
        recommendations = {
            'web_search': ['search', 'find', 'current', 'latest', 'news', 'online', 'research'],
            'file_operations': ['file', 'document', 'read', 'write', 'save', 'export'],
            'code_execution': ['code', 'python', 'script', 'programming', 'execute', 'interpret'],
            'data_processing': ['data', 'analyze', 'process', 'dataset', 'statistics'],
            'api_calls': ['api', 'service', 'integration', 'webhook', 'endpoint'],
            'email': ['email', 'send', 'notify', 'alert', 'message'],
            'scheduling': ['schedule', 'remind', 'calendar', 'appointment', 'time'],
            
            # New CrewAI tools
            'web_scraping': ['scrape', 'crawl', 'extract', 'harvest', 'website', 'webpage'],
            'document_search': ['pdf', 'docx', 'txt', 'csv', 'json', 'xml', 'document', 'paper'],
            'github_search': ['github', 'git', 'repository', 'repo', 'source code', 'codebase'],
            'youtube_search': ['youtube', 'video', 'channel', 'content', 'video analysis'],
            'vision': ['image', 'photo', 'picture', 'visual', 'dall-e', 'generate image'],
            'database_search': ['postgresql', 'postgres', 'database', 'sql', 'query'],
            'browser_automation': ['browser', 'automation', 'browserbase', 'automate', 'selenium']
        }
        
        for tool_name, keywords in recommendations.items():
            if any(keyword in task_lower for keyword in keywords):
                recommended.append(tool_name)
        
        # Always include basic tools if none matched
        if not recommended:
            recommended = ['web_search', 'file_operations']
        
        return recommended[:4]  # Increased limit to 4 tools for more comprehensive coverage
    
    def create_custom_tool(self, name: str, description: str, category: str, 
                          tool_function: Callable, config: Optional[Dict[str, Any]] = None, 
                          command: Optional[str] = None) -> bool:
        """Dynamically create and register a custom tool."""
        try:
            class DynamicCustomTool(ToolBase):
                def __init__(self, tool_name: str, tool_desc: str, tool_cat: str, tool_func: Callable, cmd: Optional[str] = None):
                    self._name = tool_name
                    self._description = tool_desc
                    self._category = tool_cat
                    self._function = tool_func
                    self._is_custom = True
                    self._command = cmd
                
                @property
                def name(self) -> str:
                    return self._name
                
                @property
                def description(self) -> str:
                    return self._description
                
                @property
                def category(self) -> str:
                    return self._category
                
                def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
                    return self._function
            
            custom_tool = DynamicCustomTool(name, description, category, tool_function, command)
            self.register_tool(custom_tool)
            self._save_custom_tools()  # Save after creating
            return True
            
        except Exception as e:
            print(f"Error creating custom tool '{name}': {e}")
            return False
    
    def create_crewai_tool(self, tool_class_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Dynamically create and register a CrewAI tool."""
        if tool_class_name not in CREWAI_TOOLS:
            print(f"CrewAI tool '{tool_class_name}' not available")
            return False
        
        try:
            tool_class = CREWAI_TOOLS[tool_class_name]
            
            # Create a wrapper for the CrewAI tool
            class CrewAIToolWrapper(ToolBase):
                def __init__(self, crewai_tool_class, tool_name: str):
                    self._tool_class = crewai_tool_class
                    self._tool_name = tool_name
                
                @property 
                def name(self) -> str:
                    return self._tool_name.lower().replace('tool', '')
                
                @property
                def description(self) -> str:
                    return f"CrewAI {self._tool_name} for advanced functionality"
                
                @property
                def category(self) -> str:
                    # Categorize based on tool name
                    if 'search' in self._tool_name.lower():
                        return 'search'
                    elif 'scrape' in self._tool_name.lower() or 'crawl' in self._tool_name.lower():
                        return 'scraping'
                    elif 'file' in self._tool_name.lower() or 'directory' in self._tool_name.lower():
                        return 'file_ops'
                    elif 'code' in self._tool_name.lower():
                        return 'development'
                    elif 'vision' in self._tool_name.lower() or 'dalle' in self._tool_name.lower():
                        return 'ai_generation'
                    else:
                        return 'utility'
                
                def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
                    if config:
                        return self._tool_class(**config)
                    else:
                        return self._tool_class()
            
            wrapper = CrewAIToolWrapper(tool_class, tool_class_name)
            self.register_tool(wrapper)
            return True
            
        except Exception as e:
            print(f"Error creating CrewAI tool '{tool_class_name}': {e}")
            return False
    
    def get_available_crewai_tools(self) -> List[str]:
        """Get list of available CrewAI tools."""
        return list(CREWAI_TOOLS.keys())
    
    def auto_register_crewai_tools(self) -> int:
        """Automatically register all available CrewAI tools."""
        registered = 0
        for tool_name in CREWAI_TOOLS.keys():
            if self.create_crewai_tool(tool_name):
                registered += 1
        return registered
    
    def _load_tool_config(self) -> Dict[str, Any]:
        """Load tool configuration including API keys."""
        config = {
            'api_keys': {
                'serper_dev': os.getenv('SERPER_API_KEY'),
                'openai': os.getenv('OPENAI_API_KEY'),
                'anthropic': os.getenv('ANTHROPIC_API_KEY'),
                'google': os.getenv('GOOGLE_API_KEY'),
                'firecrawl': os.getenv('FIRECRAWL_API_KEY'),
                'browserbase': os.getenv('BROWSERBASE_API_KEY'),
                'github': os.getenv('GITHUB_TOKEN'),
                'youtube': os.getenv('YOUTUBE_API_KEY'),
                'composio': os.getenv('COMPOSIO_API_KEY'),
                'apify': os.getenv('APIFY_API_KEY'),
                'llama_index': os.getenv('LLAMA_INDEX_API_KEY'),
            },
            'tool_settings': {}
        }
        
        # Load from config file if exists
        config_file = os.path.expanduser('~/.crewmaster/tools_config.json')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except Exception as e:
                print(f"Warning: Could not load tool config from {config_file}: {e}")
        
        return config
    
    def get_tool_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tools including API key availability."""
        status = {}
        
        for tool_name, tool_instance in self.tools.items():
            tool_status = {
                'available': True,
                'category': tool_instance.category,
                'description': tool_instance.description,
                'requires_api_key': False,
                'api_key_configured': False,
                'crewai_tool': False
            }
            
            # Check if this is a CrewAI tool that might need API keys
            if tool_name in ['web_search', 'web_scraping', 'vision', 'browser_automation']:
                tool_status['requires_api_key'] = True
                if tool_name == 'web_search':
                    tool_status['api_key_configured'] = bool(self.config['api_keys'].get('serper_dev'))
                elif tool_name == 'web_scraping':
                    tool_status['api_key_configured'] = bool(self.config['api_keys'].get('firecrawl'))
                elif tool_name == 'vision':
                    tool_status['api_key_configured'] = bool(self.config['api_keys'].get('openai'))
                elif tool_name == 'browser_automation':
                    tool_status['api_key_configured'] = bool(self.config['api_keys'].get('browserbase'))
            
            # Check if this wraps a CrewAI tool
            try:
                instance = tool_instance.get_instance()
                if hasattr(instance, '__class__') and 'crewai' in str(type(instance)).lower():
                    tool_status['crewai_tool'] = True
            except:
                tool_status['available'] = False
            
            status[tool_name] = tool_status
        
        return status
    
    def _load_custom_tools(self):
        """Load previously created custom tools."""
        custom_tools_file = os.path.expanduser('~/.crewmaster/custom_tools.json')
        if os.path.exists(custom_tools_file):
            try:
                with open(custom_tools_file, 'r') as f:
                    custom_tools = json.load(f)
                    
                for tool_data in custom_tools:
                    name = tool_data.get('name')
                    description = tool_data.get('description')
                    category = tool_data.get('category', 'custom')
                    command = tool_data.get('command')
                    
                    if name and description:
                        # Recreate the tool function
                        if command:
                            def make_tool_function(cmd):
                                def tool_function(input_data: str = "") -> str:
                                    import subprocess
                                    try:
                                        full_cmd = cmd.replace("{input}", input_data)
                                        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=30)
                                        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
                                    except Exception as e:
                                        return f"Tool execution error: {str(e)}"
                                return tool_function
                            
                            tool_function = make_tool_function(command)
                        else:
                            def tool_function(input_data: str = "") -> str:
                                return f"Custom tool '{name}' executed with input: {input_data}"
                        
                        self.create_custom_tool(name, description, category, tool_function)
                        
            except Exception as e:
                print(f"Warning: Could not load custom tools: {e}")
    
    def _save_custom_tools(self):
        """Save custom tools to persistent storage."""
        try:
            # Create directory if it doesn't exist
            custom_dir = os.path.expanduser('~/.crewmaster')
            os.makedirs(custom_dir, exist_ok=True)
            
            # Find custom tools (those not in default categories)
            custom_tools = []
            for tool_name, tool_instance in self.tools.items():
                if hasattr(tool_instance, '_is_custom') and tool_instance._is_custom:
                    tool_data = {
                        'name': tool_name,
                        'description': tool_instance.description,
                        'category': tool_instance.category,
                        'command': getattr(tool_instance, '_command', None)
                    }
                    custom_tools.append(tool_data)
            
            # Save to file
            custom_tools_file = os.path.join(custom_dir, 'custom_tools.json')
            with open(custom_tools_file, 'w') as f:
                json.dump(custom_tools, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save custom tools: {e}")