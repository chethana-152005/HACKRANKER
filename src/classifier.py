"""Classification module for request types and product areas."""

import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Classification:
    """Result of ticket classification."""
    request_type: str
    product_area: str
    inferred_domain: Optional[str]
    risk_level: str


class TicketClassifier:
    """Classify support tickets."""
    
    def __init__(self):
        # Product areas by domain
        self.product_areas = {
            'hackerrank': [
                'Authentication', 'Assessments', 'Interviews', 'Certifications',
                'Account Access', 'Plagiarism', 'IDE Issues', 'Recruiter',
                'Test Creation', 'Candidate Management', 'Reports', 'Billing',
                'Community', 'For Work', 'Mock Interviews', 'Resume Builder'
            ],
            'claude': [
                'Billing', 'Subscription', 'API Access', 'Account Access',
                'Team Permissions', 'Usage Limits', 'Safety Policies',
                'Privacy', 'Conversation Management', 'Integrations'
            ],
            'visa': [
                'Fraud & Disputes', 'Card Declined', 'Payments', 'Travel Support',
                'Chargebacks', 'ATM Issues', 'Account Security', 'Card Services',
                'General Support', 'Lost/Stolen Card'
            ]
        }
        
        # Request type patterns
        self.request_patterns = {
            'feature_request': [
                r'\b(add|create|implement|build|develop)\s+(a\s+)?(new\s+)?(feature|option|capability)',
                r'\b(would\s+like|want|need)\s+(to\s+)?(have|see|add)',
                r'\b(suggestion|improvement|enhancement)',
                r'\b(it\s+would\s+be\s+(great|helpful|nice)\s+if)',
                r'\b(please\s+add|please\s+include)',
            ],
            'bug': [
                r'\b(not\s+working|doesn\'t\s+work|does\s+not\s+work)',
                r'\b(error|crash|bug|glitch|issue)',
                r'\b(failed|failure|failing)',
                r'\b(can\'t\s+(access|load|open|submit|login))',
                r'\b(unable\s+to\s+(access|load|open|submit))',
                r'\b(broken|stuck|frozen|hanging)',
                r'\b(down|offline|unavailable)',
            ],
            'invalid': [
                r'^\s*(test|testing|hello|hi|hey)\s*$',
                r'^\s*(thank|thanks|thx)',
                r'who\s+is\s+(the\s+)?(actor|actress|president)',
                r'what\s+is\s+the\s+(capital|weather|meaning)',
                r'ignore\s+(all\s+)?(previous|above)',
            ]
        }
        
        # Domain keywords
        self.domain_keywords = {
            'HackerRank': [
                'hackerrank', 'hacker rank', 'assessment', 'test', 'candidate',
                'recruiter', 'interview', 'code', 'challenge', 'submission',
                'ide', 'compiler', 'score', 'certificate', 'mock interview'
            ],
            'Claude': [
                'claude', 'anthropic', 'ai', 'assistant', 'chat', 'conversation',
                'api', 'subscription', 'team', 'workspace', 'prompt', 'model'
            ],
            'Visa': [
                'visa', 'card', 'credit', 'debit', 'atm', 'payment', 'transaction',
                'bank', 'merchant', 'charge', 'dispute', 'fraud', 'travel',
                'currency', 'cash', 'pin', 'cardholder'
            ]
        }
    
    def classify(self, issue: str, subject: str = '', company: str = '') -> Classification:
        """Classify a support ticket."""
        text = f"{subject} {issue}".lower()
        
        # Determine domain
        inferred_domain = None
        if company and company.lower() != 'none':
            inferred_domain = company
        else:
            inferred_domain = self._infer_domain(text)
        
        # Determine request type
        request_type = self._classify_request_type(text)
        
        # Determine product area
        product_area = self._classify_product_area(text, inferred_domain)
        
        # Determine risk level
        risk_level = self._assess_risk(text)
        
        return Classification(
            request_type=request_type,
            product_area=product_area,
            inferred_domain=inferred_domain,
            risk_level=risk_level
        )
    
    def _infer_domain(self, text: str) -> Optional[str]:
        """Infer the domain from text."""
        scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[domain] = score
        
        if max(scores.values(), default=0) > 0:
            return max(scores, key=scores.get)
        
        return None
    
    def _classify_request_type(self, text: str) -> str:
        """Classify the request type."""
        # Check for invalid patterns first
        for pattern in self.request_patterns['invalid']:
            if re.search(pattern, text, re.IGNORECASE):
                return 'invalid'
        
        # Check for feature request
        for pattern in self.request_patterns['feature_request']:
            if re.search(pattern, text, re.IGNORECASE):
                return 'feature_request'
        
        # Check for bug
        for pattern in self.request_patterns['bug']:
            if re.search(pattern, text, re.IGNORECASE):
                return 'bug'
        
        # Default to product_issue
        return 'product_issue'
    
    def _classify_product_area(self, text: str, domain: Optional[str]) -> str:
        """Classify the product area."""
        if not domain:
            return 'General Support'
        
        domain_lower = domain.lower()
        areas = self.product_areas.get(domain_lower, ['General Support'])
        
        # Match keywords to product areas
        area_scores = {}
        for area in areas:
            area_keywords = area.lower().replace(' ', '_').replace('&', 'and').split('_')
            area_keywords.extend(area.lower().replace('&', 'and').split())
            score = sum(1 for kw in area_keywords if kw in text)
            area_scores[area] = score
        
        # Special handling for specific terms
        if 'login' in text or 'password' in text or 'access' in text:
            if 'account' in text:
                return 'Account Access'
            return 'Authentication'
        
        if 'payment' in text or 'billing' in text or 'invoice' in text or 'refund' in text:
            return 'Billing'
        
        if 'card' in text or 'transaction' in text or 'atm' in text:
            if domain_lower == 'visa':
                return 'Card Services'
        
        if 'fraud' in text or 'stolen' in text or 'unauthorized' in text:
            if domain_lower == 'visa':
                return 'Fraud & Disputes'
        
        if 'test' in text or 'assessment' in text or 'challenge' in text:
            if domain_lower == 'hackerrank':
                return 'Assessments'
        
        if 'interview' in text:
            if domain_lower == 'hackerrank':
                return 'Interviews'
        
        if 'api' in text:
            return 'API Access'
        
        if 'privacy' in text or 'data' in text:
            return 'Privacy'
        
        # Return highest scoring area
        if max(area_scores.values(), default=0) > 0:
            return max(area_scores, key=area_scores.get)
        
        return 'General Support'
    
    def _assess_risk(self, text: str) -> str:
        """Assess the risk level of a ticket."""
        high_risk_keywords = [
            'fraud', 'stolen', 'unauthorized', 'compromise', 'hack',
            'legal', 'lawsuit', 'attorney', 'identity theft',
            'security', 'vulnerability', 'breach'
        ]
        
        medium_risk_keywords = [
            'refund', 'dispute', 'billing', 'payment', 'cancel',
            'urgent', 'immediately', 'escalate', 'complaint'
        ]
        
        for kw in high_risk_keywords:
            if kw in text:
                return 'high'
        
        for kw in medium_risk_keywords:
            if kw in text:
                return 'medium'
        
        return 'low'