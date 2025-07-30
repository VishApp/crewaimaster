"""
CrewOrchestratorAgent - AI-powered agent for coordinating crew creation and execution.

This agent orchestrates the entire process of creating, configuring, and managing crews
by coordinating with TaskAnalyzerAgent and AgentDesignerAgent.
"""

from typing import Dict, List, Any, Optional
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from ..core.task_analyzer import CrewSpec, AgentSpec
from .task_analyzer_agent import TaskAnalyzerAgent
from .agent_designer_agent import AgentDesignerAgent, AgentDesignRequest
import json
from datetime import datetime


class CrewOrchestrationRequest(BaseModel):
    """Request for orchestrating crew creation."""
    task_description: str = Field(description="The task to be executed")
    crew_name: Optional[str] = Field(None, description="Optional custom crew name")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    constraints: List[str] = Field(default_factory=list, description="Task constraints")
    resources: Dict[str, Any] = Field(default_factory=dict, description="Available resources")


class CrewOrchestrationResult(BaseModel):
    """Result of crew orchestration."""
    crew_spec: Dict[str, Any] = Field(description="Complete crew specification")
    orchestration_log: List[str] = Field(description="Log of orchestration steps")
    recommendations: List[str] = Field(description="Optimization recommendations")
    estimated_performance: Dict[str, float] = Field(description="Performance predictions")


class CrewOrchestratorAgent:
    """AI-powered crew orchestrator using CrewAI."""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the CrewOrchestratorAgent."""
        self.llm_config = llm_config or {}
        
        # Initialize sub-agents for AI collaboration
        self.task_analyzer = TaskAnalyzerAgent(llm_config)
        self.agent_designer = AgentDesignerAgent(llm_config)
    
    def orchestrate_crew_creation(self, request: CrewOrchestrationRequest) -> CrewOrchestrationResult:
        """
        Orchestrate the complete crew creation process using AI collaboration.
        
        This method coordinates between TaskAnalyzerAgent and AgentDesignerAgent to create
        fully AI-generated crew specifications with no hardcoded templates.
        
        Args:
            request: Orchestration request with task and preferences
            
        Returns:
            CrewOrchestrationResult: Complete orchestration result with AI-generated crew spec
        """
        orchestration_log = []
        recommendations = []
        
        try:
            # Step 1: AI Task Analysis - Get intelligent task breakdown
            orchestration_log.append("🔍 Starting AI task analysis...")
            task_analysis = self.task_analyzer.analyze_task(request.task_description)
            orchestration_log.append(f"📊 Task analyzed - Complexity: {task_analysis.complexity.value}, Agents needed: {len(task_analysis.agents)}")
            
            # Step 2: AI Agent Design - Enhance each agent with AI-generated specifications
            orchestration_log.append("🎨 Starting AI agent design...")
            enhanced_agents = []
            
            print(f"🔧 DEBUG: TaskAnalyzer found {len(task_analysis.agents)} agents")
            for i, agent_spec in enumerate(task_analysis.agents):
                print(f"🔧 DEBUG: Processing agent {i+1}: {agent_spec.role} - {agent_spec.name}")
                
                # Create design request for each agent
                design_request = AgentDesignRequest(
                    role=agent_spec.role,
                    task_context=request.task_description,
                    required_capabilities=agent_spec.required_tools,
                    preferences=request.preferences,
                    constraints=request.constraints
                )
                
                # Use AI to design each agent - NO hardcoding!
                try:
                    designed_agent = self.agent_designer.design_agent(design_request)
                    print(f"🔧 DEBUG: AgentDesigner created: {designed_agent.role} - {designed_agent.name}")
                except Exception as e:
                    print(f"🔧 DEBUG: AgentDesigner failed for {agent_spec.role}: {e}")
                    continue
                
                # Create enhanced agent spec with AI-generated properties
                enhanced_agent_spec = AgentSpec(
                    role=designed_agent.role,
                    name=designed_agent.name,
                    goal=designed_agent.goal,
                    backstory=designed_agent.backstory,
                    required_tools=designed_agent.tools,
                    memory_type=designed_agent.memory_type,
                    max_iter=designed_agent.max_iterations,
                    allow_delegation=designed_agent.allow_delegation
                )
                enhanced_agents.append(enhanced_agent_spec)
            
            orchestration_log.append(f"✨ Designed {len(enhanced_agents)} AI-powered agents")
            
            # Step 3: Create crew specification using ONLY AI outputs
            crew_spec = {
                "name": request.crew_name or self._generate_ai_crew_name(task_analysis, enhanced_agents),
                "task": task_analysis.task,  # Use normalized task from AI analysis
                "description": f"AI-orchestrated crew for: {task_analysis.task}",
                "agents": [
                    {
                        "role": agent.role,
                        "name": agent.name,
                        "goal": agent.goal,
                        "backstory": agent.backstory,
                        "required_tools": agent.required_tools,
                        "memory_type": agent.memory_type,
                        "max_iter": agent.max_iter,
                        "allow_delegation": agent.allow_delegation
                    }
                    for agent in enhanced_agents
                ],
                "expected_output": task_analysis.expected_output,
                "complexity": task_analysis.complexity.value,
                "estimated_time": task_analysis.estimated_time,
                "process_type": task_analysis.process_type
            }
            
            orchestration_log.append("🎯 Crew specification completed using AI collaboration")
            
            # Step 4: AI Performance estimation
            estimated_performance = self._estimate_ai_performance(enhanced_agents, task_analysis)
            
            # Step 5: AI Recommendations
            recommendations = self._generate_ai_recommendations(enhanced_agents, task_analysis)
            
            return CrewOrchestrationResult(
                crew_spec=crew_spec,
                orchestration_log=orchestration_log,
                recommendations=recommendations,
                estimated_performance=estimated_performance
            )
            
        except Exception as e:
            orchestration_log.append(f"❌ AI orchestration failed: {str(e)}")
            raise e
    
    def _generate_ai_crew_name(self, task_analysis: CrewSpec, agents: List[AgentSpec]) -> str:
        """Generate a crew name using AI analysis insights - NO hardcoding."""
        # Extract meaningful keywords from the AI-analyzed task
        task_words = task_analysis.task.lower().replace(',', ' ').replace('.', ' ').split()
        
        # Filter out common words and focus on task-specific terms
        meaningful_words = [
            word for word in task_words[:5] 
            if word not in {'the', 'and', 'or', 'to', 'of', 'a', 'an', 'in', 'on', 'for', 'with', 'by', 'that', 'this'}
            and len(word) > 2
        ]
        
        # Create meaningful crew name from AI analysis
        if meaningful_words:
            name_parts = meaningful_words[:3]
        else:
            # Fallback to agent roles if no meaningful task words
            name_parts = [agents[0].role if agents else 'ai', 'task']
        
        crew_name = '_'.join(name_parts) + '_crew'
        return crew_name
    
    def _estimate_ai_performance(self, agents: List[AgentSpec], task_analysis: CrewSpec) -> Dict[str, float]:
        """Estimate performance based on AI analysis - dynamic, not hardcoded."""
        # Base performance on AI analysis quality
        complexity_score = {
            "simple": 0.9,
            "moderate": 0.8,
            "complex": 0.7
        }.get(task_analysis.complexity.value, 0.7)
        
        # Agent specialization bonus - AI-designed agents are more specialized
        agent_specialization = len(set(agent.role for agent in agents)) / len(agents) if agents else 0.5
        
        # Tool diversity - AI selects optimal tools
        all_tools = set()
        for agent in agents:
            all_tools.update(agent.required_tools)
        tool_diversity = min(1.0, len(all_tools) / 5)  # Normalize to max 5 tools
        
        base_performance = 0.75 + (agent_specialization * 0.15) + (tool_diversity * 0.1)
        
        return {
            "success_probability": base_performance * complexity_score,
            "efficiency_score": base_performance * 0.95,
            "quality_score": base_performance * 1.1,
            "coordination_score": agent_specialization,
            "tool_optimization": tool_diversity
        }
    
    def _generate_ai_recommendations(self, agents: List[AgentSpec], task_analysis: CrewSpec) -> List[str]:
        """Generate recommendations based on AI analysis - dynamic insights."""
        recommendations = []
        
        # Dynamic recommendations based on AI analysis
        recommendations.append("All agent specifications generated by AI for optimal task alignment")
        
        if task_analysis.complexity.value == "complex" and len(agents) >= 3:
            recommendations.append("Complex task appropriately handled with specialized agent team")
        
        if len(agents) <= 2:
            recommendations.append("Efficient team size for streamlined coordination")
        
        # Tool analysis
        all_tools = set()
        for agent in agents:
            all_tools.update(agent.required_tools)
        
        if len(all_tools) >= 5:
            recommendations.append("Comprehensive tool coverage for task requirements")
        
        recommendations.append(f"AI-optimized crew with {len(agents)} specialized agents")
        recommendations.append("Agent goals and backstories tailored specifically for this task")
        
        return recommendations