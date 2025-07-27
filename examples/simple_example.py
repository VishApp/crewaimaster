#!/usr/bin/env python3
"""
Simple example demonstrating CrewMaster usage.

This example shows how to:
1. Create a crew for a research task
2. Execute the crew
3. View the results
"""

import os
import sys

# Add the parent directory to the path so we can import crewmaster
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from crewmaster.core.config import Config
from crewmaster.core.master_agent import MasterAgent

def main():
    """Run a simple CrewMaster example."""
    print("🚀 CrewMaster Simple Example")
    print("=" * 40)
    
    try:
        # Initialize configuration
        config = Config()
        
        # Initialize master agent
        master = MasterAgent(config)
        
        print("\n📋 Creating crew for market research task...")
        
        # Create a crew for a market research task
        task_description = "Research the latest trends in artificial intelligence and machine learning for 2024"
        
        crew = master.create_crew(
            task_description=task_description,
            crew_name="ai_research_crew",
            reuse_agents=True,
            verbose=True
        )
        
        print(f"\n✅ Created crew: {crew.name}")
        print(f"🆔 Crew ID: {crew.id}")
        print(f"👥 Number of agents: {len(crew.agents)}")
        
        # Display agent information
        print("\n👥 Agents:")
        for i, agent in enumerate(crew.agents, 1):
            print(f"  {i}. {agent.name} ({agent.role})")
            print(f"     Goal: {agent.goal}")
            if agent.tools:
                tools = ", ".join([tool.name for tool in agent.tools])
                print(f"     Tools: {tools}")
            print()
        
        print("📊 System Statistics:")
        stats = master.get_system_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n🎯 Example completed successfully!")
        print("You can now use the CLI to execute this crew:")
        print(f"  crewmaster run {crew.id}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())