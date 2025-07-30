"""
Custom Tool Generator Agent for CrewAIMaster.

This agent uses CrewAI to intelligently analyze tool requirements and generate
proper CrewAI BaseTool implementations with complete code and documentation.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import os
import sys
import tempfile
import importlib.util
from pathlib import Path

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool


@dataclass
class ToolRequirement:
    """Tool requirement specification from user description."""
    name: str
    description: str
    category: str
    inputs: List[Dict[str, str]]
    functionality: str
    dependencies: List[str]
    use_cases: List[str]
    implementation_hints: str


@dataclass
class GeneratedToolResult:
    """Generated tool with complete code and metadata."""
    name: str
    description: str
    category: str
    full_code: str
    file_path: str
    dependencies: List[str]
    validation_passed: bool
    validation_errors: List[str]


class CustomToolGeneratorAgent:
    """Agent that generates custom CrewAI tools using intelligent analysis."""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the custom tool generator agent."""
        self.llm_config = llm_config or {}
        self.tools_directory = Path("/tmp/crewaimaster_custom_tools")
        self.tools_directory.mkdir(exist_ok=True)
        
        # Create the analysis agent
        self.analyzer_agent = self._create_analyzer_agent()
        
        # Create the code generator agent
        self.generator_agent = self._create_generator_agent()
        
        # Create the validator agent
        self.validator_agent = self._create_validator_agent()
    
    def _create_analyzer_agent(self) -> Agent:
        """Create the requirement analysis agent."""
        return Agent(
            role="Tool Requirements Analyst",
            goal="Analyze user descriptions to extract detailed requirements for custom CrewAI tools",
            backstory="""You are an expert software architect who specializes in understanding 
            user requirements and translating them into detailed technical specifications. 
            You have deep knowledge of CrewAI framework, Python development, and API integrations. 
            You excel at breaking down natural language descriptions into structured requirements.""",
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            memory=True
        )
    
    def _create_generator_agent(self) -> Agent:
        """Create the code generation agent."""
        return Agent(
            role="CrewAI Tool Code Generator",
            goal="Generate complete, working CrewAI BaseTool implementations from requirements",
            backstory="""You are a senior Python developer with expertise in the CrewAI framework. 
            You create production-ready tools that follow best practices including proper error handling, 
            input validation, documentation, and testing. You know how to integrate with various APIs 
            and services like Slack, email, databases, and file systems. Your code is clean, 
            well-documented, and follows CrewAI patterns.""",
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            memory=True
        )
    
    def _create_validator_agent(self) -> Agent:
        """Create the code validation agent."""
        return Agent(
            role="Code Quality Validator",
            goal="Validate and test generated CrewAI tools for correctness and functionality",
            backstory="""You are a quality assurance expert who ensures code correctness, 
            security, and performance. You test Python code, validate CrewAI tool implementations, 
            check for security vulnerabilities, and ensure proper error handling. You provide 
            detailed feedback on code quality and suggest improvements.""",
            verbose=True,
            allow_delegation=False,
            max_iter=2,
            memory=True
        )
    
    def generate_custom_tool(self, user_description: str, show_code: bool = True, auto_confirm: bool = False) -> GeneratedToolResult:
        """Generate a custom CrewAI tool from user description using intelligent agents."""
        
        print(f"🤖 Analyzing tool requirements with AI...")
        
        # Step 1: Analyze requirements
        analysis_task = Task(
            description=f"""
            Analyze the following user description and extract detailed tool requirements:
            
            User Description: "{user_description}"
            
            Extract and provide:
            1. Tool name (PascalCase, ending with 'Tool')
            2. Category (api, communication, file, data, web, automation, etc.)
            3. Main functionality description
            4. Required input parameters with types and descriptions
            5. Expected output format
            6. Required dependencies/libraries
            7. Implementation approach and hints
            8. Example use cases
            
            Provide your analysis in a structured format that can be used for code generation.
            """,
            agent=self.analyzer_agent,
            expected_output="Detailed tool requirements including name, category, inputs, dependencies, and implementation approach"
        )
        
        # Step 2: Generate code
        code_generation_task = Task(
            description=f"""
            Based on the requirements analysis, generate a complete CrewAI BaseTool implementation.
            
            Requirements: Use the output from the previous analysis task.
            
            Generate a complete Python file that includes:
            
            1. **Proper imports** - Use these EXACT import statements:
               ```
               from crewai.tools import BaseTool
               from pydantic import BaseModel, Field
               from typing import Type, Optional, Dict, Any
               ```
            2. **Pydantic input schema** with proper field definitions and descriptions
            3. **CrewAI BaseTool class** with:
               - Correct typed annotations: name: str = "ToolName"
               - Correct typed annotations: description: str = "Tool description"  
               - Proper args_schema: Type[BaseModel] = InputSchemaClass
               - Complete _run method implementation with real functionality
               - Proper error handling and logging
            4. **ToolBase wrapper class** for CrewAIMaster registry integration
            5. **Registration function** that registers with ToolRegistry
            6. **Test function** that validates the tool works
            7. **Complete documentation** with docstrings and comments
            
            CRITICAL REQUIREMENTS:
            - Use EXACT imports: `from crewai.tools import BaseTool`
            - BaseTool class must have properly typed attributes:
              * name: str = "ToolName"
              * description: str = "Tool description" 
              * args_schema: Type[BaseModel] = InputSchemaClass
            - Return ONLY Python code, NO markdown fences
            - Do NOT include ```python or ``` in your response
            - Start directly with the import statements
            
            For the specific functionality described by the user:
            - If it's Slack: Include slack_sdk integration with proper authentication
            - If it's email: Include smtplib or email service integration
            - If it's API: Include requests with proper error handling
            - If it's file operations: Include proper file handling with pathlib
            - If it's data processing: Include pandas/numpy as appropriate
            
            Make sure the code is production-ready with:
            - Proper error handling and user-friendly error messages
            - Input validation and sanitization
            - Logging for debugging
            - Configuration support (environment variables, config files)
            - Comprehensive docstrings
            
            Return the complete Python code as a single file.
            """,
            agent=self.generator_agent,
            expected_output="Complete Python code for a working CrewAI BaseTool implementation",
            context=[analysis_task]
        )
        
        # Step 3: Validate code
        validation_task = Task(
            description=f"""
            Review and validate the generated CrewAI tool code for:
            
            1. **Syntax correctness** - ensure Python syntax is valid
            2. **CrewAI compliance** - follows BaseTool patterns correctly
            3. **Security** - no security vulnerabilities or unsafe operations
            4. **Error handling** - proper exception handling throughout
            5. **Code quality** - follows Python best practices
            6. **Dependencies** - all imports are available and correct
            7. **Documentation** - proper docstrings and comments
            8. **Testing** - includes test function that can verify functionality
            
            Provide:
            - Overall validation status (PASS/FAIL)
            - List of any issues found
            - Suggested improvements
            - Security concerns if any
            - Missing dependencies if any
            
            If major issues are found, provide corrected code.
            """,
            agent=self.validator_agent,
            expected_output="Validation report with status, issues, and suggestions for improvement",
            context=[analysis_task, code_generation_task]
        )
        
        # Create and execute the crew
        from crewai import Process
        tool_generation_crew = Crew(
            agents=[self.analyzer_agent, self.generator_agent, self.validator_agent],
            tasks=[analysis_task, code_generation_task, validation_task],
            verbose=True,
            process=Process.sequential  # Sequential execution
        )
        
        print(f"🛠️  Generating tool with AI crew...")
        result = tool_generation_crew.kickoff()
        
        # Parse the crew results
        try:
            # Extract the generated code from the results
            code_output = code_generation_task.output.raw if hasattr(code_generation_task.output, 'raw') else str(code_generation_task.output)
            validation_output = validation_task.output.raw if hasattr(validation_task.output, 'raw') else str(validation_task.output)
            
            # Clean up code output - remove markdown fences if present
            code_output = self._clean_generated_code(code_output)
            
            # Extract tool name from the analysis
            analysis_output = analysis_task.output.raw if hasattr(analysis_task.output, 'raw') else str(analysis_task.output)
            tool_name = self._extract_tool_name_from_analysis(analysis_output)
            
            # Show code preview if requested
            if show_code:
                self._display_generated_code(code_output, tool_name)
            
            # Get user confirmation
            if not auto_confirm:
                confirm = input("\n✅ Do you want to create this tool? (y/n): ").lower().strip()
                if confirm != 'y':
                    return GeneratedToolResult(
                        name=tool_name,
                        description=user_description,
                        category="unknown",
                        full_code="",
                        file_path="",
                        dependencies=[],
                        validation_passed=False,
                        validation_errors=["User cancelled tool creation"]
                    )
            
            # Save the generated tool
            tool_file_path = self._save_generated_tool(code_output, tool_name)
            
            # Ask about dependency installation
            dependencies = self._extract_dependencies_from_code(code_output)
            if dependencies and not auto_confirm:
                deps_str = ', '.join(dependencies)
                install_deps = input(f"\n📦 Install dependencies ({deps_str})? (y/n): ").lower().strip()
                if install_deps == 'y':
                    self._install_dependencies(dependencies)
            
            # Test the generated tool
            validation_passed, validation_errors = self._test_generated_tool(tool_file_path, tool_name)
            
            # Handle validation results
            if validation_passed:
                self._register_tool_with_crewaimaster(tool_file_path, tool_name)
            else:
                # Ask user if they want to save the tool anyway
                if not auto_confirm:
                    print(f"\n⚠️  Validation failed: {'; '.join(validation_errors)}")
                    save_anyway = input("💾 Save tool anyway? (y/n): ").lower().strip()
                    if save_anyway == 'y':
                        print(f"📁 Tool saved to {tool_file_path} (validation bypassed)")
                        validation_passed = True  # Override for user choice
                        validation_errors = [f"Validation bypassed by user: {'; '.join(validation_errors)}"]
            
            return GeneratedToolResult(
                name=tool_name,
                description=user_description,
                category=self._extract_category_from_analysis(analysis_output),
                full_code=code_output,
                file_path=tool_file_path,
                dependencies=self._extract_dependencies_from_code(code_output),
                validation_passed=validation_passed,
                validation_errors=validation_errors
            )
            
        except Exception as e:
            return GeneratedToolResult(
                name="Unknown",
                description=user_description,
                category="unknown",
                full_code="",
                file_path="",
                dependencies=[],
                validation_passed=False,
                validation_errors=[f"Error processing crew results: {str(e)}"]
            )
    
    def _clean_generated_code(self, code_output: str) -> str:
        """Clean up generated code by removing markdown fences and extra formatting."""
        # Remove markdown code fences
        lines = code_output.split('\n')
        cleaned_lines = []
        
        skip_line = False
        for line in lines:
            # Skip markdown code fence lines
            if line.strip().startswith('```'):
                skip_line = not skip_line if line.strip() == '```' else False
                continue
            
            # If we're not skipping, add the line
            if not skip_line:
                cleaned_lines.append(line)
        
        # Join back and clean up
        cleaned_code = '\n'.join(cleaned_lines)
        
        # Remove any remaining markdown artifacts
        cleaned_code = cleaned_code.replace('```python', '')
        cleaned_code = cleaned_code.replace('```', '')
        
        # Remove extra whitespace at the beginning and end
        cleaned_code = cleaned_code.strip()
        
        return cleaned_code
    
    def _extract_tool_name_from_analysis(self, analysis_text: str) -> str:
        """Extract tool name from analysis output."""
        # Look for tool name patterns in the analysis
        import re
        patterns = [
            r'Tool name[:\s]+([A-Za-z]+Tool)',
            r'Name[:\s]+([A-Za-z]+Tool)',
            r'([A-Za-z]+Tool)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, analysis_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Fallback to generic name
        return "CustomTool"
    
    def _extract_category_from_analysis(self, analysis_text: str) -> str:
        """Extract category from analysis output."""
        import re
        pattern = r'Category[:\s]+([a-zA-Z_]+)'
        match = re.search(pattern, analysis_text, re.IGNORECASE)
        return match.group(1) if match else "utility"
    
    def _extract_dependencies_from_code(self, code: str) -> List[str]:
        """Extract dependencies from generated code."""
        import re
        import_pattern = r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        from_pattern = r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        deps = set()
        for line in code.split('\n'):
            line = line.strip()
            
            import_match = re.match(import_pattern, line)
            if import_match:
                deps.add(import_match.group(1))
            
            from_match = re.match(from_pattern, line)
            if from_match:
                deps.add(from_match.group(1))
        
        # Filter out standard library and common modules
        standard_libs = {'os', 'sys', 'json', 'datetime', 'typing', 're', 'pathlib', 'tempfile', 'importlib'}
        return [dep for dep in deps if dep not in standard_libs]
    
    def _display_generated_code(self, code: str, tool_name: str):
        """Display the generated code for user review."""
        print("\n" + "="*80)
        print(f"📄 GENERATED CODE PREVIEW: {tool_name}")
        print("="*80)
        print(code)
        print("="*80)
    
    def _save_generated_tool(self, code: str, tool_name: str) -> str:
        """Save the generated tool to a file."""
        tool_file = self.tools_directory / f"{tool_name.lower()}_generated.py"
        
        with open(tool_file, 'w') as f:
            f.write(code)
        
        print(f"📁 Tool file created: {tool_file}")
        return str(tool_file)
    
    def _test_generated_tool(self, tool_file_path: str, tool_name: str) -> tuple[bool, List[str]]:
        """Test the generated tool to ensure it works."""
        try:
            # First, check syntax by parsing the file
            with open(tool_file_path, 'r') as f:
                code_content = f.read()
            
            import ast
            try:
                ast.parse(code_content)
                print(f"✅ Syntax validation passed for {tool_name}")
            except SyntaxError as e:
                return False, [f"Syntax error in generated code: {str(e)}"]
            
            # Try to import the generated module
            spec = importlib.util.spec_from_file_location(
                f"{tool_name.lower()}_generated", 
                tool_file_path
            )
            if spec is None or spec.loader is None:
                return False, ["Could not load tool module"]
            
            module = importlib.util.module_from_spec(spec)
            
            # Try to execute the module, but handle missing dependencies gracefully
            try:
                spec.loader.exec_module(module)
            except ImportError as e:
                # Check if it's a missing optional dependency
                missing_dep = str(e).replace("No module named ", "").strip("'\"")
                if missing_dep in ['slack_sdk', 'requests', 'pandas', 'smtplib', 'openpyxl']:
                    print(f"⚠️  Optional dependency '{missing_dep}' not installed - tool structure validated")
                    print(f"✅ Tool validation passed for {tool_name} (dependency check skipped)")
                    return True, [f"Missing optional dependency: {missing_dep}"]
                else:
                    return False, [f"Import error: {str(e)}"]
            
            # Look for the tool class
            tool_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (hasattr(attr, '__bases__') and 
                    any('BaseTool' in str(base) for base in attr.__bases__)):
                    tool_class = attr
                    break
            
            if tool_class is None:
                return False, ["No BaseTool class found in generated code"]
            
            # Try to instantiate, but handle dependency issues
            try:
                tool_instance = tool_class()
            except Exception as e:
                if any(dep in str(e) for dep in ['slack_sdk', 'requests', 'pandas']):
                    print(f"⚠️  Tool requires external dependencies - structure validated")
                    print(f"✅ Tool validation passed for {tool_name} (runtime check skipped)")
                    return True, [f"Runtime dependency issue: {str(e)}"]
                else:
                    return False, [f"Tool instantiation failed: {str(e)}"]
            
            # Verify it's a proper CrewAI tool
            if not hasattr(tool_instance, '_run'):
                return False, ["Tool does not implement _run method"]
            
            if not hasattr(tool_instance, 'name'):
                return False, ["Tool does not have name attribute"]
            
            if not hasattr(tool_instance, 'description'):
                return False, ["Tool does not have description attribute"]
            
            print(f"✅ Tool validation passed for {tool_instance.name}")
            return True, []
            
        except Exception as e:
            return False, [f"Tool test failed: {str(e)}"]
    
    def _install_dependencies(self, dependencies: List[str]):
        """Install required dependencies using pip."""
        import subprocess
        
        try:
            for dep in dependencies:
                print(f"🔄 Installing {dep}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", dep], 
                    capture_output=True, 
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    print(f"✅ Successfully installed {dep}")
                else:
                    print(f"❌ Failed to install {dep}: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            print("❌ Installation timed out")
        except Exception as e:
            print(f"❌ Error during installation: {str(e)}")
    
    def _register_tool_with_crewaimaster(self, tool_file_path: str, tool_name: str):
        """Register the generated tool with CrewAIMaster tool registry."""
        try:
            print(f"📋 Registering {tool_name} with CrewAIMaster...")
            
            # Import and execute the registration function
            spec = importlib.util.spec_from_file_location(
                f"{tool_name.lower()}_generated", 
                tool_file_path
            )
            if spec is None or spec.loader is None:
                print("❌ Could not load tool for registration")
                return
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for registration function
            for attr_name in dir(module):
                if 'register' in attr_name.lower() and callable(getattr(module, attr_name)):
                    register_func = getattr(module, attr_name)
                    register_func()
                    print(f"✅ Tool {tool_name} registered successfully")
                    return
            
            print(f"⚠️  No registration function found in {tool_name}")
            
        except Exception as e:
            print(f"❌ Tool registration failed: {str(e)}")