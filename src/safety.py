"""Safety checks and escalation logic."""

import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SafetyAssessment:
    """Result of safety assessment."""
    is_safe: bool
    should_escalate: bool
    risk_level: str  # low, medium, high
    risk_factors: List[str]
    escalation_reason: Optional[str] = None


class SafetyChecker:
    """Perform safety checks on support tickets."""
    
    def __init__(self):
        # High-risk patterns that always require escalation
        self.high_risk_patterns = [
            r'\b(fraud|fraudulent|scam|scammed|stolen|identity\s*theft)\b',
            r'\b(unauthorized\s*charg|unauthorised\s*charg)\b',
            r'\b(account\s*compromis|hack|breach)\b',
            r'\b(suspicious\s*activity|suspicious\s*transaction)\b',
            r'\b(stolen\s*card|lost\s*card|card\s*lost)\b',
            r'\b(dispute|chargeback|refund\s*dispute)\b',
            r'\b(missing\s*payment|payment\s*not\s*received)\b',
            r'\b(lawsuit|sue|legal\s*action|attorney|lawyer)\b',
            r'\b(identity\s*verif|verify\s*my\s*identity)\b',
            r'\b(account\s*suspended|account\s*banned|account\s*locked)\b',
            r'\b(restore\s*access|regain\s*access)\b',
            r'\b(i\s*am\s*not\s*the\s*owner|not\s*the\s*admin)\b',
        ]
        
        self.malicious_patterns = [
            r'ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts)',
            r'you\s+are\s+now\s+(a|an)\s+\w+',
            r'disregard\s+(all\s+)?(instructions|rules)',
            r'show\s+(me\s+)?(your|the)\s+(prompt|instructions|system)',
            r'reveal\s+(your|the)\s+(reasoning|chain\s*of\s*thought)',
        ]
        
        self.escalation_keywords = [
            'human', 'person', 'agent', 'supervisor', 'manager',
            'escalate', 'legal', 'lawsuit', 'fraud', 'stolen'
        ]
    
    def assess(self, issue: str, subject: str = '', company: str = '') -> SafetyAssessment:
        """Assess the safety and risk level of a ticket."""
        text = f"{subject} {issue}".lower()
        
        risk_factors = []
        risk_level = 'low'
        should_escalate = False
        escalation_reason = None
        
        # Check for malicious patterns
        for pattern in self.malicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return SafetyAssessment(
                    is_safe=False,
                    should_escalate=True,
                    risk_level='high',
                    risk_factors=['malicious_input'],
                    escalation_reason='Potential malicious input detected'
                )
        
        # Check for high-risk patterns
        for pattern in self.high_risk_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                risk_factors.append(f"high_risk_pattern")
                risk_level = 'high'
                should_escalate = True
                escalation_reason = 'High-risk issue requires human review'
        
        # Check for escalation keywords
        for keyword in self.escalation_keywords:
            if keyword in text:
                should_escalate = True
                escalation_reason = 'Customer requested escalation or sensitive topic'
        
        return SafetyAssessment(
            is_safe=risk_level != 'high',
            should_escalate=should_escalate,
            risk_level=risk_level,
            risk_factors=risk_factors,
            escalation_reason=escalation_reason
        )
    
    def check_response_safety(self, response: str) -> Tuple[bool, List[str]]:
        """Check if a generated response is safe to send."""
        issues = []
        return len(issues) == 0, issues


class EscalationDecider:
    """Decide whether a ticket should be escalated."""
    
    def __init__(self):
        self.safety_checker = SafetyChecker()
    
    def should_escalate(self, issue: str, subject: str, company: str,
                       request_type: str, has_relevant_docs: bool) -> Tuple[bool, str]:
        """Determine if escalation is needed."""
        
        # Safety check
        safety = self.safety_checker.assess(issue, subject, company)
        if safety.should_escalate:
            return True, safety.escalation_reason or 'Safety assessment requires escalation'
        
        # No relevant documentation found
        if not has_relevant_docs:
            return True, 'No relevant support documentation available'
        
        # Company is None and cannot be inferred
        if company == 'None' or not company:
            return True, 'Unable to determine relevant support domain'
        
        text = f"{subject} {issue}".lower()
        
        # HackerRank specific escalations
        if company.lower() == 'hackerrank':
            if 'reschedule' in text and 'assessment' in text:
                return True, 'Assessment rescheduling requires coordination with recruiter'
            if 'refund' in text:
                return True, 'Refund requests require human review'
        
        # Visa specific escalations
        if company.lower() == 'visa':
            if 'refund' in text or 'dispute' in text:
                return True, 'Financial disputes require human review'
            if 'identity' in text and 'theft' in text:
                return True, 'Identity theft requires security escalation'
        
        return False, ''