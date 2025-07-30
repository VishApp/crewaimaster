"""
Tool Designer Agent for CrewAIMaster.

This agent analyzes user requirements and generates proper CrewAI custom tools
with complete code implementation following the BaseTool pattern.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import re
import json


@dataclass
class ToolRequirement:
    """Tool requirement specification."""
    name: str
    description: str
    category: str
    inputs: List[Dict[str, str]]
    functionality: str
    dependencies: List[str]
    use_cases: List[str]


@dataclass
class GeneratedTool:
    """Generated tool specification with code."""
    name: str
    description: str
    category: str
    input_schema_code: str
    tool_class_code: str
    registration_code: str
    test_code: str
    dependencies: List[str]
    file_path: str


class ToolDesignerAgent:
    """Agent that designs and generates custom CrewAI tools."""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the tool designer agent."""
        self.llm_config = llm_config or {}
        
    def analyze_tool_requirements(self, user_description: str) -> ToolRequirement:
        """Analyze user description and extract tool requirements."""
        
        # Parse user description using pattern matching and NLP-like analysis
        normalized_desc = user_description.lower().strip()
        
        # Extract tool name
        name = self._extract_tool_name(normalized_desc)
        
        # Determine category
        category = self._determine_category(normalized_desc)
        
        # Extract functionality
        functionality = self._extract_functionality(normalized_desc)
        
        # Determine inputs
        inputs = self._extract_inputs(normalized_desc)
        
        # Identify dependencies
        dependencies = self._identify_dependencies(normalized_desc)
        
        # Generate use cases
        use_cases = self._generate_use_cases(normalized_desc)
        
        return ToolRequirement(
            name=name,
            description=user_description,
            category=category,
            inputs=inputs,
            functionality=functionality,
            dependencies=dependencies,
            use_cases=use_cases
        )
    
    def generate_tool_code(self, requirement: ToolRequirement) -> GeneratedTool:
        """Generate complete CrewAI tool code from requirements."""
        
        # Generate input schema
        input_schema = self._generate_input_schema(requirement)
        
        # Generate tool class
        tool_class = self._generate_tool_class(requirement)
        
        # Generate registration code
        registration = self._generate_registration_code(requirement)
        
        # Generate test code
        test_code = self._generate_test_code(requirement)
        
        # Determine file path
        file_path = f"/tmp/crewaimaster_custom_tools/{requirement.name.lower()}_tool.py"
        
        return GeneratedTool(
            name=requirement.name,
            description=requirement.description,
            category=requirement.category,
            input_schema_code=input_schema,
            tool_class_code=tool_class,
            registration_code=registration,
            test_code=test_code,
            dependencies=requirement.dependencies,
            file_path=file_path
        )
    
    def _extract_tool_name(self, description: str) -> str:
        """Extract tool name from description."""
        # Look for explicit names
        name_patterns = [
            r'call it\s+(\w+)',
            r'name it\s+(\w+)',
            r'called\s+(\w+)',
            r'tool\s+(\w+)',
            r'^(\w+)\s+tool',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, description)
            if match:
                return self._format_tool_name(match.group(1))
        
        # Generate name from functionality
        if 'api' in description:
            return 'CustomAPITool'
        elif 'file' in description:
            return 'FileProcessorTool'
        elif 'web' in description or 'http' in description:
            return 'WebInteractionTool'
        elif 'data' in description:
            return 'DataProcessorTool'
        elif 'email' in description:
            return 'EmailTool'
        elif 'notification' in description or 'alert' in description:
            return 'NotificationTool'
        else:
            return 'CustomTool'
    
    def _determine_category(self, description: str) -> str:
        """Determine tool category from description."""
        category_keywords = {
            'api': ['api', 'rest', 'http', 'request', 'endpoint'],
            'file': ['file', 'document', 'csv', 'json', 'excel', 'pdf'],
            'data': ['data', 'process', 'analyze', 'transform', 'calculate'],
            'web': ['web', 'scrape', 'browser', 'website', 'html'],
            'communication': ['email', 'slack', 'notification', 'message', 'alert'],
            'database': ['database', 'sql', 'query', 'table', 'postgres', 'mysql'],
            'automation': ['automate', 'schedule', 'trigger', 'workflow'],
            'ai': ['ai', 'ml', 'model', 'predict', 'analyze', 'classify'],
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in description for keyword in keywords):
                return category
        
        return 'utility'
    
    def _extract_functionality(self, description: str) -> str:
        """Extract core functionality from description."""
        # Extract action verbs and objects
        action_patterns = [
            r'(send|get|fetch|retrieve|process|analyze|create|generate|convert|transform|validate|check)',
            r'(monitor|track|watch|observe|scan|search|find|extract)',
            r'(upload|download|save|load|read|write|parse|format)',
        ]
        
        functionality_parts = []
        for pattern in action_patterns:
            matches = re.findall(pattern, description)
            functionality_parts.extend(matches)
        
        if functionality_parts:
            return f"Tool that can {', '.join(set(functionality_parts))} based on user requirements"
        
        return "Custom tool that processes user input and returns results"
    
    def _extract_inputs(self, description: str) -> List[Dict[str, str]]:
        """Extract input parameters from description."""
        inputs = []
        
        # Common input patterns
        if 'url' in description:
            inputs.append({
                'name': 'url',
                'type': 'str',
                'description': 'URL to process or interact with'
            })
        
        if 'file' in description:
            inputs.append({
                'name': 'file_path',
                'type': 'str', 
                'description': 'Path to the file to process'
            })
        
        if 'data' in description:
            inputs.append({
                'name': 'data',
                'type': 'str',
                'description': 'Data to process or analyze'
            })
        
        if 'query' in description or 'search' in description:
            inputs.append({
                'name': 'query',
                'type': 'str',
                'description': 'Search query or search terms'
            })
        
        if 'message' in description or 'text' in description:
            inputs.append({
                'name': 'message',
                'type': 'str',
                'description': 'Message or text content'
            })
        
        # Default input if none found
        if not inputs:
            inputs.append({
                'name': 'input_data',
                'type': 'str',
                'description': 'Input data for the tool'
            })
        
        return inputs
    
    def _identify_dependencies(self, description: str) -> List[str]:
        """Identify required dependencies from description."""
        deps = []
        
        dependency_map = {
            'requests': ['api', 'http', 'url', 'web', 'rest'],
            'pandas': ['csv', 'excel', 'data', 'dataframe'],
            'beautifulsoup4': ['html', 'scrape', 'parse', 'web'],
            'smtplib': ['email', 'smtp', 'mail'],
            'sqlalchemy': ['database', 'sql', 'postgres', 'mysql'],
            'openpyxl': ['excel', 'xlsx'],
            'pillow': ['image', 'photo', 'picture'],
            'pydantic': [],  # Always needed for CrewAI tools
        }
        
        for dep, keywords in dependency_map.items():
            if any(keyword in description for keyword in keywords) or dep == 'pydantic':
                deps.append(dep)
        
        return deps
    
    def _generate_use_cases(self, description: str) -> List[str]:
        """Generate example use cases."""
        if 'api' in description:
            return [
                "Fetch data from external APIs",
                "Send data to web services",
                "Integrate with third-party platforms"
            ]
        elif 'file' in description:
            return [
                "Process local files",
                "Parse document contents", 
                "Generate file reports"
            ]
        else:
            return [
                "Automate custom workflows",
                "Process specialized data",
                "Extend CrewAI capabilities"
            ]
    
    def _generate_input_schema(self, req: ToolRequirement) -> str:
        """Generate Pydantic input schema code."""
        schema_name = f"{req.name}Input"
        
        fields = []
        for input_def in req.inputs:
            field_code = f'    {input_def["name"]}: {input_def["type"]} = Field(..., description="{input_def["description"]}")'
            fields.append(field_code)
        
        return f'''class {schema_name}(BaseModel):
    """Input schema for {req.name}."""
{chr(10).join(fields)}'''
    
    def _generate_tool_class(self, req: ToolRequirement) -> str:
        """Generate CrewAI tool class code."""
        schema_name = f"{req.name}Input"
        
        # Generate _run method parameters
        params = ', '.join([f'{inp["name"]}: {inp["type"]}' for inp in req.inputs])
        
        # Generate basic implementation based on category
        implementation = self._generate_tool_implementation(req)
        
        return f'''class {req.name}(BaseTool):
    name: str = "{req.name}"
    description: str = "{req.description}"
    args_schema: Type[BaseModel] = {schema_name}

    def _run(self, {params}) -> str:
        """Execute the tool logic."""
        try:
{implementation}
        except Exception as e:
            return f"Error in {req.name}: {{str(e)}}"'''
    
    def _generate_tool_implementation(self, req: ToolRequirement) -> str:
        """Generate tool implementation based on category."""
        if req.category == 'api':
            return '''            import requests
            # TODO: Implement your API logic here
            # response = requests.get(url)
            # return response.text
            return f"API tool executed with input: {locals()}"'''
        
        elif req.category == 'file':
            return '''            import os
            # TODO: Implement your file processing logic here
            # with open(file_path, 'r') as f:
            #     content = f.read()
            # return f"Processed file: {content[:100]}..."
            return f"File tool executed with input: {locals()}"'''
        
        elif req.category == 'data':
            return '''            # TODO: Implement your data processing logic here
            # Process the data according to your requirements
            # Apply transformations, calculations, or analysis
            return f"Data processed: {locals()}"'''
        
        else:
            return '''            # TODO: Implement your custom tool logic here
            # Add your specific functionality
            # Process inputs and return meaningful results
            return f"Custom tool executed with input: {locals()}"'''
    
    def _generate_registration_code(self, req: ToolRequirement) -> str:
        """Generate tool registration code."""
        return f'''# Registration code for {req.name}
from typing import Dict, Any, Optional
from crewaimaster.tools.registry import ToolRegistry, ToolBase

class {req.name}Wrapper(ToolBase):
    """Wrapper to make {req.name} compatible with CrewAIMaster registry."""
    
    def __init__(self):
        self.tool_instance = {req.name}()
    
    @property
    def name(self) -> str:
        return self.tool_instance.name
    
    @property
    def description(self) -> str:
        return "{req.description}"
    
    @property
    def category(self) -> str:
        return "{req.category}"
    
    def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get tool instance."""
        return self.tool_instance
    
    def __call__(self, *args, **kwargs):
        return self.tool_instance(*args, **kwargs)

def register_{req.name.lower()}():
    """Register the {req.name} with CrewAIMaster."""
    registry = ToolRegistry()
    wrapped_tool = {req.name}Wrapper()
    registry.register_tool(wrapped_tool)
    print(f"✅ Registered {req.name} successfully")

# Auto-register when imported
if __name__ != "__main__":
    register_{req.name.lower()}()'''
    
    def _generate_test_code(self, req: ToolRequirement) -> str:
        """Generate test code for the tool."""
        test_params = {}
        for inp in req.inputs:
            if inp['name'] == 'url':
                test_params[inp['name']] = '"https://api.example.com/test"'
            elif inp['name'] == 'file_path':
                test_params[inp['name']] = '"/tmp/test_file.txt"'
            elif inp['name'] == 'query':
                test_params[inp['name']] = '"test query"'
            else:
                test_params[inp['name']] = '"test input"'
        
        params_str = ', '.join([f'{k}={v}' for k, v in test_params.items()])
        
        return f'''# Test code for {req.name}
def test_{req.name.lower()}():
    """Test the {req.name} tool."""
    tool = {req.name}()
    
    try:
        result = tool._run({params_str})
        print(f"✅ {req.name} test passed: {{result}}")
        return True
    except Exception as e:
        print(f"❌ {req.name} test failed: {{e}}")
        return False

if __name__ == "__main__":
    test_{req.name.lower()}()'''
    
    def _format_tool_name(self, name: str) -> str:
        """Format tool name to PascalCase."""
        # Convert to PascalCase
        formatted = ''.join(word.capitalize() for word in re.split(r'[_\s]+', name))
        
        # Ensure it ends with 'Tool' if not already
        if not formatted.endswith('Tool'):
            formatted += 'Tool'
        
        return formatted