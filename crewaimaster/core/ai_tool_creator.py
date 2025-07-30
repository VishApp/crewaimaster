"""
AI-Powered Tool Creator for CrewAIMaster.

This module provides an intelligent tool creation system using CrewAI agents
that can analyze user requirements and generate complete, working tools.
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..agents.custom_tool_generator_agent import CustomToolGeneratorAgent, GeneratedToolResult


class AIToolCreator:
    """AI-powered tool creator using CrewAI agents for intelligent tool generation."""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None, require_llm: bool = True):
        """Initialize the AI tool creator."""
        self.llm_config = llm_config or {}
        
        # Only validate LLM config if we require it (not for stats/listing)
        if require_llm and not self._validate_llm_config():
            raise ValueError("LLM configuration required for AI tool creation. Please set OPENAI_API_KEY.")
        
        # Only initialize the generator agent if LLM is required
        self.tool_generator_agent = None
        if require_llm:
            self.tool_generator_agent = CustomToolGeneratorAgent(llm_config)
        
        self.tools_directory = Path("/tmp/crewaimaster_custom_tools")
        self.tools_directory.mkdir(exist_ok=True)
    
    def _validate_llm_config(self) -> bool:
        """Validate that LLM configuration is available."""
        # Check for OpenAI API key
        if os.getenv('OPENAI_API_KEY'):
            return True
        
        # Check for other LLM configurations
        if self.llm_config.get('api_key'):
            return True
        
        return False
    
    def create_custom_tool(self, user_description: str, 
                          show_code: bool = True,
                          auto_confirm: bool = False) -> Dict[str, Any]:
        """Create a custom tool from user description using AI agents."""
        
        if not self.tool_generator_agent:
            return {
                "success": False,
                "message": "AI tool generator not initialized. Please provide LLM configuration.",
                "error": "Missing LLM configuration"
            }
        
        try:
            print(f"\\n[bold blue]🤖 AI-Powered Tool Creation with CrewAI Agents[/bold blue]")
            print(f"[cyan]Description:[/cyan] {user_description}")
            print(f"[dim]Using intelligent agents to analyze and generate your tool...[/dim]")
            
            # Generate the tool using AI agents
            result = self.tool_generator_agent.generate_custom_tool(
                user_description=user_description,
                show_code=show_code,
                auto_confirm=auto_confirm
            )
            
            # Process the results
            if result.validation_passed:
                # Check if there were validation warnings (like missing dependencies)
                warnings = [err for err in result.validation_errors if 'dependency' in err.lower()]
                
                return {
                    "success": True,
                    "message": f"Successfully created {result.name}",
                    "tool_name": result.name,
                    "tool_file": result.file_path,
                    "category": result.category,
                    "description": result.description,
                    "dependencies": result.dependencies,
                    "generated_with": "AI Agents",
                    "warnings": warnings
                }
            else:
                return {
                    "success": False,
                    "message": f"Tool creation failed validation: {'; '.join(result.validation_errors)}",
                    "errors": result.validation_errors,
                    "generated_code": result.full_code if result.full_code else None
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"AI tool creation failed: {str(e)}",
                "error": str(e)
            }
    
    def list_ai_generated_tools(self) -> List[Dict[str, Any]]:
        """List all AI-generated custom tools."""
        tools = []
        
        if not self.tools_directory.exists():
            return tools
        
        for tool_file in self.tools_directory.glob("*_generated.py"):
            try:
                # Read file to extract metadata
                with open(tool_file, 'r') as f:
                    content = f.read()
                
                # Extract tool info
                tool_info = {
                    "file": str(tool_file),
                    "name": tool_file.stem.replace("_generated", ""),
                    "created": tool_file.stat().st_mtime,
                    "type": "AI Generated",
                    "size": len(content)
                }
                
                # Try to extract description from docstring or comments
                lines = content.split('\\n')
                for line in lines:
                    if 'description' in line.lower() and '=' in line:
                        # Extract description from assignment
                        desc_part = line.split('=', 1)[1].strip().strip('"').strip("'")
                        tool_info["description"] = desc_part[:100] + "..." if len(desc_part) > 100 else desc_part
                        break
                else:
                    tool_info["description"] = "AI-generated CrewAI tool"
                
                tools.append(tool_info)
                
            except Exception as e:
                # Skip files that can't be processed
                continue
        
        return sorted(tools, key=lambda x: x['created'], reverse=True)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about AI-generated tools."""
        tools = self.list_ai_generated_tools()
        
        if not tools:
            return {
                "total_tools": 0,
                "total_size": 0,
                "avg_size": 0,
                "newest_tool": None,
                "oldest_tool": None,
                "tools_directory": str(self.tools_directory)
            }
        
        total_size = sum(tool['size'] for tool in tools)
        
        return {
            "total_tools": len(tools),
            "total_size": total_size,
            "avg_size": total_size // len(tools) if tools else 0,
            "newest_tool": tools[0] if tools else None,
            "oldest_tool": tools[-1] if tools else None,
            "tools_directory": str(self.tools_directory)
        }
    
    def delete_ai_tool(self, tool_name: str) -> Dict[str, Any]:
        """Delete an AI-generated tool."""
        try:
            tool_file = self.tools_directory / f"{tool_name.lower()}_generated.py"
            
            if not tool_file.exists():
                return {"success": False, "message": f"AI-generated tool file not found: {tool_name}"}
            
            tool_file.unlink()
            
            return {"success": True, "message": f"AI-generated tool {tool_name} deleted successfully"}
            
        except Exception as e:
            return {"success": False, "message": f"Failed to delete AI tool: {str(e)}"}
    
    def validate_ai_tool(self, tool_name: str) -> Dict[str, Any]:
        """Validate an AI-generated tool."""
        try:
            tool_file = self.tools_directory / f"{tool_name.lower()}_generated.py"
            
            if not tool_file.exists():
                return {"valid": False, "errors": [f"Tool file not found: {tool_name}"]}
            
            # Use the same validation logic as the generator agent
            validation_passed, validation_errors = self.tool_generator_agent._test_generated_tool(
                str(tool_file), tool_name
            )
            
            return {
                "valid": validation_passed,
                "errors": validation_errors,
                "file_path": str(tool_file)
            }
            
        except Exception as e:
            return {"valid": False, "errors": [f"Validation failed: {str(e)}"]}


# Compatibility function for existing code
def create_intelligent_tool(user_description: str, llm_config: Optional[Dict[str, Any]] = None, 
                          show_code: bool = True, auto_confirm: bool = False) -> Dict[str, Any]:
    """
    Convenience function for creating intelligent tools.
    
    This function provides backward compatibility and a simple interface
    for creating AI-powered tools.
    """
    try:
        creator = AIToolCreator(llm_config)
        return creator.create_custom_tool(user_description, show_code, auto_confirm)
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to initialize AI tool creator: {str(e)}",
            "error": str(e)
        }