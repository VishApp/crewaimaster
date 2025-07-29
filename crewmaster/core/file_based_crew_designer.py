"""
File-based Crew Designer for CrewMaster.

This module creates CrewAI crews as file-based projects with YAML configurations
and Python modules, while storing minimal metadata in the database.
"""

import os
import importlib.util
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .task_analyzer import CrewSpec, AgentSpec
from .config import Config
from .file_generator import CrewFileGenerator


class FileBasedCrewDesigner:
    """Creates and manages file-based CrewAI crews."""
    
    def __init__(self, config: Config, crews_base_path: str = "crews"):
        """Initialize the file-based crew designer."""
        self.config = config
        self.crews_base_path = Path(crews_base_path)
        self.crews_base_path.mkdir(exist_ok=True)
        
        # Initialize file generator
        self.file_generator = CrewFileGenerator(str(self.crews_base_path))
    
    def create_crew_from_spec(self, spec: CrewSpec) -> Dict[str, Any]:
        """Create a new file-based crew from a crew specification."""
        print(f"🔧 Creating file-based crew: {spec.name}")
        
        try:
            # Generate the crew project files
            crew_path = self.file_generator.generate_crew_project(spec)
            print(f"✅ Generated crew files at: {crew_path}")
            
            # Store minimal metadata in database
            metadata = self._create_crew_metadata(spec, crew_path)
            self._store_crew_metadata(metadata)
            
            return {
                'name': spec.name,
                'path': crew_path,
                'status': 'created',
                'agents_count': len(spec.agents),
                'tools_used': self._extract_all_tools(spec),
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Failed to create crew: {str(e)}")
            raise
    
    def run_crew(self, crew_name: str, task_input: str = None) -> str:
        """Run a file-based crew by executing its Python module."""
        crew_path = self.crews_base_path / crew_name
        
        if not crew_path.exists():
            raise ValueError(f"Crew '{crew_name}' does not exist at {crew_path}")
        
        try:
            # Import and execute the crew module
            crew_module = self._import_crew_module(crew_name)
            
            # Create crew instance and run
            crew_class = getattr(crew_module, f"{self._to_class_name(crew_name)}")
            crew_instance = crew_class()
            
            print(f"🚀 Running crew: {crew_name}")
            result = crew_instance.run(task_input)
            
            # Update execution metadata
            self._update_execution_metadata(crew_name, 'completed')
            
            return result
            
        except Exception as e:
            self._update_execution_metadata(crew_name, 'failed', str(e))
            raise Exception(f"Failed to run crew '{crew_name}': {str(e)}")
    
    def list_crews(self) -> List[Dict[str, Any]]:
        """List all available file-based crews."""
        crews = []
        
        for crew_dir in self.crews_base_path.iterdir():
            if crew_dir.is_dir() and (crew_dir / "config" / "agents.yaml").exists():
                crew_info = self._get_crew_info(crew_dir.name)
                crews.append(crew_info)
        
        return crews
    
    def get_crew_details(self, crew_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific crew."""
        crew_path = self.crews_base_path / crew_name
        
        if not crew_path.exists():
            raise ValueError(f"Crew '{crew_name}' does not exist")
        
        return self._get_crew_info(crew_name, detailed=True)
    
    def export_crew(self, crew_name: str, output_path: str = None) -> str:
        """Export a crew as a ZIP file for sharing."""
        try:
            zip_path = self.file_generator.export_crew_as_zip(crew_name, output_path)
            print(f"✅ Crew exported to: {zip_path}")
            return zip_path
        except Exception as e:
            print(f"❌ Failed to export crew: {str(e)}")
            raise
    
    def delete_crew(self, crew_name: str) -> bool:
        """Delete a file-based crew and its metadata."""
        crew_path = self.crews_base_path / crew_name
        
        if not crew_path.exists():
            return False
        
        try:
            import shutil
            shutil.rmtree(crew_path)
            
            # Remove metadata from database
            self._remove_crew_metadata(crew_name)
            
            print(f"✅ Deleted crew: {crew_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to delete crew: {str(e)}")
            return False
    
    def update_crew_config(self, crew_name: str, config_updates: Dict[str, Any]) -> bool:
        """Update crew configuration files."""
        crew_path = self.crews_base_path / crew_name
        
        if not crew_path.exists():
            return False
        
        try:
            import yaml
            
            # Update agents.yaml if provided
            if 'agents' in config_updates:
                agents_file = crew_path / "config" / "agents.yaml"
                with open(agents_file, 'r') as f:
                    agents_config = yaml.safe_load(f)
                
                # Update agent configurations
                for agent_name, updates in config_updates['agents'].items():
                    if agent_name in agents_config:
                        agents_config[agent_name].update(updates)
                
                with open(agents_file, 'w') as f:
                    yaml.dump(agents_config, f, default_flow_style=False, indent=2)
            
            # Update tasks.yaml if provided
            if 'tasks' in config_updates:
                tasks_file = crew_path / "config" / "tasks.yaml"
                with open(tasks_file, 'r') as f:
                    tasks_config = yaml.safe_load(f)
                
                # Update task configurations
                for task_name, updates in config_updates['tasks'].items():
                    if task_name in tasks_config:
                        tasks_config[task_name].update(updates)
                
                with open(tasks_file, 'w') as f:
                    yaml.dump(tasks_config, f, default_flow_style=False, indent=2)
            
            print(f"✅ Updated crew configuration: {crew_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update crew config: {str(e)}")
            return False
    
    def _import_crew_module(self, crew_name: str):
        """Dynamically import a crew module."""
        crew_path = self.crews_base_path / crew_name
        src_path = crew_path / "src" / crew_name
        
        # Add the src directory to Python path temporarily
        src_str = str(src_path.parent)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
        
        try:
            # Import the crew module
            module_name = f"{crew_name}.crew"
            spec = importlib.util.spec_from_file_location(
                module_name, 
                src_path / "crew.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return module
            
        finally:
            # Remove from path to avoid conflicts
            if src_str in sys.path:
                sys.path.remove(src_str)
    
    def _create_crew_metadata(self, spec: CrewSpec, crew_path: str) -> Dict[str, Any]:
        """Create minimal metadata for database storage."""
        return {
            'name': spec.name,
            'description': spec.description,
            'task': spec.task,
            'expected_output': spec.expected_output,
            'process_type': spec.process_type,
            'file_path': crew_path,
            'agents_count': len(spec.agents),
            'tools_used': self._extract_all_tools(spec),
            'created_at': datetime.now(),
            'execution_count': 0,
            'last_executed': None,
            'status': 'created'
        }
    
    def _extract_all_tools(self, spec: CrewSpec) -> List[str]:
        """Extract all unique tools used across all agents."""
        all_tools = set()
        for agent in spec.agents:
            all_tools.update(agent.required_tools)
        return list(all_tools)
    
    def _store_crew_metadata(self, metadata: Dict[str, Any]):
        """Store crew metadata in database."""
        # For now, store in a simple file-based cache
        # TODO: Implement proper database storage with minimal schema
        cache_file = self.crews_base_path / ".crew_metadata.json"
        
        try:
            import json
            
            # Load existing metadata
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    all_metadata = json.load(f)
            else:
                all_metadata = {}
            
            # Add new metadata
            all_metadata[metadata['name']] = {
                **metadata,
                'created_at': metadata['created_at'].isoformat(),
                'last_executed': metadata['last_executed'].isoformat() if metadata['last_executed'] else None
            }
            
            # Save back to file
            with open(cache_file, 'w') as f:
                json.dump(all_metadata, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to store metadata: {str(e)}")
    
    def _update_execution_metadata(self, crew_name: str, status: str, error_msg: str = None):
        """Update execution metadata."""
        cache_file = self.crews_base_path / ".crew_metadata.json"
        
        try:
            import json
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    all_metadata = json.load(f)
                
                if crew_name in all_metadata:
                    all_metadata[crew_name]['execution_count'] += 1
                    all_metadata[crew_name]['last_executed'] = datetime.now().isoformat()
                    all_metadata[crew_name]['status'] = status
                    
                    if error_msg:
                        all_metadata[crew_name]['last_error'] = error_msg
                    
                    with open(cache_file, 'w') as f:
                        json.dump(all_metadata, f, indent=2)
                        
        except Exception as e:
            print(f"⚠️ Failed to update execution metadata: {str(e)}")
    
    def _remove_crew_metadata(self, crew_name: str):
        """Remove crew metadata from cache."""
        cache_file = self.crews_base_path / ".crew_metadata.json"
        
        try:
            import json
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    all_metadata = json.load(f)
                
                if crew_name in all_metadata:
                    del all_metadata[crew_name]
                    
                    with open(cache_file, 'w') as f:
                        json.dump(all_metadata, f, indent=2)
                        
        except Exception as e:
            print(f"⚠️ Failed to remove metadata: {str(e)}")
    
    def _get_crew_info(self, crew_name: str, detailed: bool = False) -> Dict[str, Any]:
        """Get crew information from files and metadata."""
        crew_path = self.crews_base_path / crew_name
        
        # Basic info from directory
        info = {
            'name': crew_name,
            'path': str(crew_path),
            'created_at': datetime.fromtimestamp(crew_path.stat().st_ctime).isoformat()
        }
        
        # Load metadata from cache
        cache_file = self.crews_base_path / ".crew_metadata.json"
        if cache_file.exists():
            try:
                import json
                with open(cache_file, 'r') as f:
                    all_metadata = json.load(f)
                
                if crew_name in all_metadata:
                    info.update(all_metadata[crew_name])
            except Exception:
                pass
        
        if detailed:
            # Load configuration details
            try:
                import yaml
                
                # Load agents config
                agents_file = crew_path / "config" / "agents.yaml"
                if agents_file.exists():
                    with open(agents_file, 'r') as f:
                        info['agents'] = yaml.safe_load(f)
                
                # Load tasks config
                tasks_file = crew_path / "config" / "tasks.yaml"
                if tasks_file.exists():
                    with open(tasks_file, 'r') as f:
                        info['tasks'] = yaml.safe_load(f)
                
                # Check for README
                readme_file = crew_path / "README.md"
                info['has_readme'] = readme_file.exists()
                
                # Check for requirements
                req_file = crew_path / "requirements.txt"
                info['has_requirements'] = req_file.exists()
                
            except Exception as e:
                info['config_error'] = str(e)
        
        return info
    
    def _to_class_name(self, name: str) -> str:
        """Convert crew name to Python class name."""
        clean_name = ''.join(c for c in name if c.isalnum() or c == '_')
        words = clean_name.replace('_', ' ').split()
        return ''.join(word.capitalize() for word in words) + 'Crew'