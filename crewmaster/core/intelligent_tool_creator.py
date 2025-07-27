"""
Intelligent Tool Creator for CrewMaster.

This module provides AI-powered custom tool creation that generates proper
CrewAI BaseTool implementations with code preview and validation.
"""

import os
import tempfile
import importlib.util
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..agents.tool_designer_agent import ToolDesignerAgent, GeneratedTool
from ..tools.registry import ToolRegistry


class IntelligentToolCreator:
    """AI-powered tool creator that generates proper CrewAI tools."""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the intelligent tool creator."""
        self.designer_agent = ToolDesignerAgent(llm_config)
        self.tool_registry = ToolRegistry()
        self.tools_directory = Path("/tmp/crewmaster_custom_tools")
        self.tools_directory.mkdir(exist_ok=True)
    
    def create_custom_tool(self, user_description: str, 
                          show_code: bool = True,
                          auto_confirm: bool = False) -> Dict[str, Any]:
        """Create a custom tool from user description."""
        
        try:
            # Step 1: Analyze requirements
            print("🔍 Analyzing your tool requirements...")
            requirements = self.designer_agent.analyze_tool_requirements(user_description)
            
            print(f"✅ Analysis complete:")
            print(f"   Tool Name: {requirements.name}")
            print(f"   Category: {requirements.category}")
            print(f"   Inputs: {[inp['name'] for inp in requirements.inputs]}")
            print(f"   Dependencies: {requirements.dependencies}")
            
            # Step 2: Generate tool code
            print("\\n🛠️  Generating CrewAI tool code...")
            generated_tool = self.designer_agent.generate_tool_code(requirements)
            
            # Step 3: Show code preview
            if show_code:
                self._display_generated_code(generated_tool)
            
            # Step 4: Get user confirmation
            if not auto_confirm:
                confirm = input("\\n✅ Do you want to create this tool? (y/n): ").lower().strip()
                if confirm != 'y':
                    return {"success": False, "message": "Tool creation cancelled by user"}
            
            # Step 5: Create and validate tool
            print("\\n📝 Creating tool files...")
            tool_file_path = self._create_tool_file(generated_tool)
            
            print("🧪 Testing tool implementation...")
            test_result = self._test_generated_tool(generated_tool, tool_file_path)
            
            # Step 6: Register tool
            if test_result["success"]:
                print("📋 Registering tool with CrewMaster...")
                registration_result = self._register_tool(generated_tool, tool_file_path)
                
                if registration_result["success"]:
                    return {
                        "success": True,
                        "message": f"Successfully created {generated_tool.name}",
                        "tool_name": generated_tool.name,
                        "tool_file": tool_file_path,
                        "category": generated_tool.category,
                        "description": generated_tool.description
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Tool created but registration failed: {registration_result['error']}"
                    }
            else:
                return {
                    "success": False,
                    "message": f"Tool test failed: {test_result['error']}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Tool creation failed: {str(e)}"
            }
    
    def _display_generated_code(self, tool: GeneratedTool):
        """Display the generated tool code for user review."""
        print("\\n" + "="*80)
        print(f"📄 GENERATED CODE PREVIEW: {tool.name}")
        print("="*80)
        
        print("\\n🔧 Input Schema:")
        print("-" * 40)
        print(tool.input_schema_code)
        
        print("\\n🛠️  Tool Class:")
        print("-" * 40)
        print(tool.tool_class_code)
        
        print("\\n📋 Registration Code:")
        print("-" * 40)
        print(tool.registration_code)
        
        print("\\n🧪 Test Code:")
        print("-" * 40)
        print(tool.test_code)
        
        if tool.dependencies:
            print("\\n📦 Required Dependencies:")
            print("-" * 40)
            for dep in tool.dependencies:
                print(f"  - {dep}")
        
        print("\\n" + "="*80)
    
    def _create_tool_file(self, tool: GeneratedTool) -> str:
        """Create the actual tool file with all necessary code."""
        tool_file = self.tools_directory / f"{tool.name.lower()}_tool.py"
        
        # Generate complete file content
        file_content = f'''"""
Custom CrewAI Tool: {tool.name}
Generated by CrewMaster Intelligent Tool Creator

Description: {tool.description}
Category: {tool.category}
"""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

{tool.input_schema_code}

{tool.tool_class_code}

{tool.registration_code}

{tool.test_code}
'''
        
        # Write file
        with open(tool_file, 'w') as f:
            f.write(file_content)
        
        print(f"📁 Tool file created: {tool_file}")
        return str(tool_file)
    
    def _test_generated_tool(self, tool: GeneratedTool, tool_file_path: str) -> Dict[str, Any]:
        """Test the generated tool to ensure it works."""
        try:
            # Import the generated module
            spec = importlib.util.spec_from_file_location(
                f"{tool.name.lower()}_tool", 
                tool_file_path
            )
            if spec is None or spec.loader is None:
                return {"success": False, "error": "Could not load tool module"}
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the tool class
            tool_class = getattr(module, tool.name, None)
            if tool_class is None:
                return {"success": False, "error": f"Tool class {tool.name} not found in module"}
            
            # Instantiate and test basic functionality
            tool_instance = tool_class()
            
            # Verify it's a proper CrewAI tool
            if not hasattr(tool_instance, '_run'):
                return {"success": False, "error": "Tool does not implement _run method"}
            
            if not hasattr(tool_instance, 'name'):
                return {"success": False, "error": "Tool does not have name attribute"}
            
            if not hasattr(tool_instance, 'description'):
                return {"success": False, "error": "Tool does not have description attribute"}
            
            print(f"✅ Tool structure validation passed")
            print(f"   Name: {tool_instance.name}")
            print(f"   Description: {tool_instance.description}")
            
            return {"success": True, "tool_instance": tool_instance}
            
        except Exception as e:
            return {"success": False, "error": f"Tool test failed: {str(e)}"}
    
    def _register_tool(self, tool: GeneratedTool, tool_file_path: str) -> Dict[str, Any]:
        """Register the tool with CrewMaster tool registry."""
        try:
            # Import and get tool instance
            spec = importlib.util.spec_from_file_location(
                f"{tool.name.lower()}_tool", 
                tool_file_path
            )
            if spec is None or spec.loader is None:
                return {"success": False, "error": "Could not load tool for registration"}
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            tool_class = getattr(module, tool.name)
            tool_instance = tool_class()
            
            # Create a ToolBase wrapper for the CrewAI tool
            from ..tools.registry import ToolBase
            
            class CrewAIToolWrapper(ToolBase):
                def __init__(self, crewai_tool, category_val, description_val):
                    self.crewai_tool = crewai_tool
                    self._category = category_val
                    self._description = description_val
                
                @property
                def name(self) -> str:
                    return self.crewai_tool.name
                
                @property
                def description(self) -> str:
                    return self._description
                
                @property
                def category(self) -> str:
                    return self._category
                
                def get_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
                    """Get tool instance."""
                    return self.crewai_tool
                
                def __call__(self, *args, **kwargs):
                    return self.crewai_tool(*args, **kwargs)
            
            # Wrap the tool for registration
            wrapped_tool = CrewAIToolWrapper(tool_instance, tool.category, tool.description)
            
            # Register with the tool registry
            self.tool_registry.register_tool(wrapped_tool)
            
            print(f"✅ Tool {tool.name} registered successfully")
            return {"success": True}
                
        except Exception as e:
            return {"success": False, "error": f"Tool registration failed: {str(e)}"}
    
    def list_custom_tools(self) -> List[Dict[str, Any]]:
        """List all custom tools that have been created."""
        tools = []
        
        if not self.tools_directory.exists():
            return tools
        
        for tool_file in self.tools_directory.glob("*_tool.py"):
            try:
                # Read file to extract metadata
                with open(tool_file, 'r') as f:
                    content = f.read()
                
                # Extract tool name and description from file
                tool_info = {
                    "file": str(tool_file),
                    "name": tool_file.stem.replace("_tool", ""),
                    "created": tool_file.stat().st_mtime,
                }
                
                # Try to extract description from file content
                if 'Description:' in content:
                    desc_line = [line for line in content.split('\\n') if 'Description:' in line][0]
                    tool_info["description"] = desc_line.split('Description:')[1].strip()
                else:
                    tool_info["description"] = "Custom CrewAI tool"
                
                tools.append(tool_info)
                
            except Exception:
                continue
        
        return sorted(tools, key=lambda x: x['created'], reverse=True)
    
    def delete_custom_tool(self, tool_name: str) -> Dict[str, Any]:
        """Delete a custom tool."""
        try:
            tool_file = self.tools_directory / f"{tool_name.lower()}_tool.py"
            
            if not tool_file.exists():
                return {"success": False, "message": f"Tool file not found: {tool_name}"}
            
            tool_file.unlink()
            
            # Try to unregister from tool registry
            try:
                self.tool_registry.unregister_tool(tool_name.lower())
            except:
                pass  # Ignore if unregistration fails
            
            return {"success": True, "message": f"Tool {tool_name} deleted successfully"}
            
        except Exception as e:
            return {"success": False, "message": f"Failed to delete tool: {str(e)}"}