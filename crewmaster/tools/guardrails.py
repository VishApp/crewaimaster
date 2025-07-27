"""
Guardrails system for CrewMaster.

This module provides safety and quality controls for agent operations.
"""

from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
from enum import Enum
import re

class GuardrailSeverity(Enum):
    """Severity levels for guardrail violations."""
    WARNING = "warning"
    BLOCK = "block"
    CRITICAL = "critical"

class GuardrailResult:
    """Result of a guardrail check."""
    
    def __init__(self, passed: bool, severity: GuardrailSeverity = GuardrailSeverity.WARNING,
                 message: str = "", details: Optional[Dict[str, Any]] = None):
        self.passed = passed
        self.severity = severity
        self.message = message
        self.details = details or {}

class GuardrailBase(ABC):
    """Base class for all guardrails."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Guardrail name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Guardrail description."""
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Guardrail category."""
        pass
    
    @abstractmethod
    def check(self, content: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        """Check if content passes the guardrail."""
        pass

class PIIDetectionGuardrail(GuardrailBase):
    """Detects and blocks personally identifiable information."""
    
    def __init__(self):
        """Initialize PII detection patterns."""
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
    
    @property
    def name(self) -> str:
        return "pii_detection"
    
    @property
    def description(self) -> str:
        return "Detects and blocks personally identifiable information"
    
    @property
    def category(self) -> str:
        return "security"
    
    def check(self, content: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        """Check for PII in content."""
        detected_pii = []
        
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                detected_pii.append({
                    'type': pii_type,
                    'count': len(matches),
                    'samples': matches[:3]  # Show first 3 matches
                })
        
        if detected_pii:
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.BLOCK,
                message=f"Detected PII: {', '.join([item['type'] for item in detected_pii])}",
                details={'detected_pii': detected_pii}
            )
        
        return GuardrailResult(passed=True)

class ToxicityDetectionGuardrail(GuardrailBase):
    """Detects toxic or harmful content."""
    
    def __init__(self):
        """Initialize toxicity detection patterns."""
        self.toxic_keywords = [
            'hate', 'violence', 'harm', 'kill', 'destroy', 'attack',
            'discriminat', 'racist', 'sexist', 'toxic'
        ]
    
    @property
    def name(self) -> str:
        return "toxicity_detection"
    
    @property
    def description(self) -> str:
        return "Detects toxic or harmful content"
    
    @property
    def category(self) -> str:
        return "safety"
    
    def check(self, content: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        """Check for toxic content."""
        content_lower = content.lower()
        detected_toxic = []
        
        for keyword in self.toxic_keywords:
            if keyword in content_lower:
                detected_toxic.append(keyword)
        
        if detected_toxic:
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.WARNING,
                message=f"Potential toxic content detected: {', '.join(detected_toxic)}",
                details={'toxic_keywords': detected_toxic}
            )
        
        return GuardrailResult(passed=True)

class OutputLengthGuardrail(GuardrailBase):
    """Enforces output length limits."""
    
    def __init__(self, max_length: int = 10000, min_length: int = 10):
        """Initialize length limits."""
        self.max_length = max_length
        self.min_length = min_length
    
    @property
    def name(self) -> str:
        return "output_length"
    
    @property
    def description(self) -> str:
        return f"Enforces output length between {self.min_length} and {self.max_length} characters"
    
    @property
    def category(self) -> str:
        return "quality"
    
    def check(self, content: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        """Check content length."""
        length = len(content)
        
        if length > self.max_length:
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.WARNING,
                message=f"Output too long: {length} characters (max: {self.max_length})",
                details={'length': length, 'max_length': self.max_length}
            )
        
        if length < self.min_length:
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.WARNING,
                message=f"Output too short: {length} characters (min: {self.min_length})",
                details={'length': length, 'min_length': self.min_length}
            )
        
        return GuardrailResult(passed=True)

class CodeSafetyGuardrail(GuardrailBase):
    """Detects potentially unsafe code patterns."""
    
    def __init__(self):
        """Initialize code safety patterns."""
        self.dangerous_patterns = [
            r'os\.system\(',
            r'subprocess\.call\(',
            r'eval\(',
            r'exec\(',
            r'__import__\(',
            r'open\(.+[\'"]w[\'"]',  # File writing
            r'rm\s+-rf',
            r'sudo\s+',
            r'curl.*\|.*sh'
        ]
    
    @property
    def name(self) -> str:
        return "code_safety"
    
    @property
    def description(self) -> str:
        return "Detects potentially unsafe code patterns"
    
    @property
    def category(self) -> str:
        return "security"
    
    def check(self, content: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        """Check for unsafe code patterns."""
        detected_patterns = []
        
        for pattern in self.dangerous_patterns:
            matches = re.findall(pattern, content)
            if matches:
                detected_patterns.append(pattern)
        
        if detected_patterns:
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.BLOCK,
                message=f"Dangerous code patterns detected: {len(detected_patterns)} patterns",
                details={'patterns': detected_patterns}
            )
        
        return GuardrailResult(passed=True)

class HallucinationDetectionGuardrail(GuardrailBase):
    """Detects potential hallucinations in AI-generated content."""
    
    def __init__(self):
        """Initialize hallucination detection."""
        self.suspicious_phrases = [
            "according to my database",
            "in my files",
            "i remember",
            "i know for certain",
            "definitely true",
            "without a doubt"
        ]
    
    @property
    def name(self) -> str:
        return "hallucination_detection"
    
    @property
    def description(self) -> str:
        return "Detects potential hallucinations in AI-generated content"
    
    @property
    def category(self) -> str:
        return "quality"
    
    def check(self, content: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        """Check for potential hallucinations."""
        content_lower = content.lower()
        detected_phrases = []
        
        for phrase in self.suspicious_phrases:
            if phrase in content_lower:
                detected_phrases.append(phrase)
        
        if detected_phrases:
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.WARNING,
                message=f"Potential hallucination indicators detected: {len(detected_phrases)}",
                details={'suspicious_phrases': detected_phrases}
            )
        
        return GuardrailResult(passed=True)

class GuardrailEngine:
    """Engine for managing and executing guardrails."""
    
    def __init__(self):
        """Initialize the guardrail engine."""
        self.guardrails: Dict[str, GuardrailBase] = {}
        self._register_default_guardrails()
    
    def _register_default_guardrails(self):
        """Register default guardrails."""
        default_guardrails = [
            PIIDetectionGuardrail(),
            ToxicityDetectionGuardrail(),
            OutputLengthGuardrail(),
            CodeSafetyGuardrail(),
            HallucinationDetectionGuardrail()
        ]
        
        for guardrail in default_guardrails:
            self.register_guardrail(guardrail)
    
    def register_guardrail(self, guardrail: GuardrailBase):
        """Register a new guardrail."""
        self.guardrails[guardrail.name] = guardrail
    
    def unregister_guardrail(self, name: str):
        """Unregister a guardrail."""
        if name in self.guardrails:
            del self.guardrails[name]
    
    def check_content(self, content: str, guardrail_names: Optional[List[str]] = None,
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, GuardrailResult]:
        """Check content against specified guardrails."""
        if guardrail_names is None:
            guardrail_names = list(self.guardrails.keys())
        
        results = {}
        
        for name in guardrail_names:
            if name in self.guardrails:
                guardrail = self.guardrails[name]
                result = guardrail.check(content, context)
                results[name] = result
        
        return results
    
    def should_block(self, results: Dict[str, GuardrailResult]) -> bool:
        """Determine if content should be blocked based on results."""
        for result in results.values():
            if not result.passed and result.severity in [GuardrailSeverity.BLOCK, GuardrailSeverity.CRITICAL]:
                return True
        return False
    
    def get_warnings(self, results: Dict[str, GuardrailResult]) -> List[str]:
        """Get warning messages from results."""
        warnings = []
        for result in results.values():
            if not result.passed and result.severity == GuardrailSeverity.WARNING:
                warnings.append(result.message)
        return warnings
    
    def list_guardrails(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        """List available guardrails."""
        guardrails = []
        
        for guardrail in self.guardrails.values():
            if category is None or guardrail.category == category:
                guardrails.append({
                    'name': guardrail.name,
                    'description': guardrail.description,
                    'category': guardrail.category
                })
        
        return guardrails
    
    def get_recommended_guardrails(self, task_description: str) -> List[str]:
        """Get recommended guardrails based on task description."""
        task_lower = task_description.lower()
        recommended = []
        
        # Always include basic safety guardrails
        recommended.extend(['pii_detection', 'toxicity_detection'])
        
        # Add specific guardrails based on task type
        if any(word in task_lower for word in ['code', 'script', 'programming', 'execute']):
            recommended.append('code_safety')
        
        if any(word in task_lower for word in ['research', 'information', 'facts', 'data']):
            recommended.append('hallucination_detection')
        
        if any(word in task_lower for word in ['long', 'detailed', 'comprehensive', 'summary']):
            recommended.append('output_length')
        
        return list(set(recommended))  # Remove duplicates
    
    def create_custom_guardrail(self, name: str, description: str, category: str,
                              check_function: Callable[[str, Optional[Dict[str, Any]]], GuardrailResult]):
        """Create and register a custom guardrail."""
        class CustomGuardrail(GuardrailBase):
            @property
            def name(self) -> str:
                return name
            
            @property
            def description(self) -> str:
                return description
            
            @property
            def category(self) -> str:
                return category
            
            def check(self, content: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
                return check_function(content, context)
        
        self.register_guardrail(CustomGuardrail())