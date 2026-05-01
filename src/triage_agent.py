"""Main triage agent implementation."""

import os
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

# --- STABLE LIBRARY IMPORT ---
import google.generativeai as genai

from ingest import CorpusLoader, TicketLoader, Document
from retriever import SemanticRetriever, SearchResult
from classifier import TicketClassifier, Classification
from safety import SafetyChecker, EscalationDecider
from prompts import (
    SYSTEM_PROMPT, TRIAGE_PROMPT, CLASSIFICATION_PROMPT,
    ESCALATION_RESPONSE, OUT_OF_SCOPE_RESPONSE
)

logger = logging.getLogger(__name__)


@dataclass
class TriageResult:
    """Result of triaging a ticket."""
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TriageAgent:
    """Multi-domain support triage agent."""
    
    def __init__(self, api_key: str = None, corpus_dir: str = "data/corpus",
                 use_embeddings: bool = True):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.corpus_dir = corpus_dir
        self.use_embeddings = use_embeddings
        
        # Initialize components
        self.corpus_loader = CorpusLoader(corpus_dir)
        self.ticket_loader = TicketLoader()
        self.retriever = SemanticRetriever()
        self.classifier = TicketClassifier()
        self.safety_checker = SafetyChecker()
        self.escalation_decider = EscalationDecider()
        
        # Gemini model
        self.model = None
        self._init_gemini()
        
        # Load corpus
        self.documents = []
        self._load_corpus()
    
    def _init_gemini(self):
        """Initialize Gemini API client."""
        if not self.api_key:
            logger.warning("No Gemini API key provided")
            return
        
        try:
            genai.configure(api_key=self.api_key)
            # Use the stable model name
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Initialized Gemini API client (stable library)")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
    
    def _load_corpus(self):
        """Load and index the support corpus."""
        logger.info("Loading support corpus...")
        self.documents = self.corpus_loader.load_all()
        
        if self.documents:
            logger.info(f"Loaded {len(self.documents)} documents")
            if self.use_embeddings:
                self.retriever.build_index(self.documents)
        else:
            logger.warning("No documents loaded from corpus")
    
    def triage(self, issue: str, subject: str = '', company: str = '') -> TriageResult:
        """Process a single support ticket."""
        logger.info(f"Processing ticket: {subject[:50] if subject else 'No subject'}...")
        
        # Step 1: Initial classification
        classification = self.classifier.classify(issue, subject, company)
        logger.info(f"Classification: {classification}")
        
        # Step 2: Safety assessment
        safety = self.safety_checker.assess(issue, subject, company)
        logger.info(f"Safety assessment: risk={safety.risk_level}, escalate={safety.should_escalate}")
        
        # Step 3: Determine domain
        domain = company if company and company.lower() != 'none' else classification.inferred_domain
        if not domain:
            domain = 'Unknown'
        
        # Step 4: Retrieve relevant documents
        search_results = []
        if domain and domain.lower() != 'unknown':
            domain_key = domain.lower()
            search_results = self.retriever.search(
                query=f"{subject} {issue}",
                domain=domain_key,
                top_k=5,
                min_score=0.1
            )
        
        logger.info(f"Found {len(search_results)} relevant documents")
        
        # Step 5: Check if escalation is needed
        should_escalate, escalation_reason = self.escalation_decider.should_escalate(
            issue, subject, company,
            classification.request_type,
            len(search_results) > 0
        )
        
        # Override with safety decision if needed
        if safety.should_escalate and safety.risk_level == 'high':
            should_escalate = True
            escalation_reason = safety.escalation_reason
        
        # Step 6: Generate response
        if should_escalate:
            response = self._generate_escalation_response(domain, escalation_reason)
            justification = escalation_reason
            status = 'escalated'
        elif classification.request_type == 'invalid':
            response = OUT_OF_SCOPE_RESPONSE
            justification = 'Request is outside support scope'
            status = 'replied'
        else:
            # Use LLM to generate response
            response, justification = self._generate_response(
                issue, subject, company, domain,
                classification, search_results
            )
            status = 'replied'
        
        # Step 7: Validate response
        if status == 'replied':
            is_safe, issues = self.safety_checker.check_response_safety(response)
            if not is_safe:
                logger.warning(f"Response safety issues: {issues}")
                status = 'escalated'
                response = self._generate_escalation_response(domain, 'Response requires human review')
                justification = f"Response validation failed: {', '.join(issues)}"
        
        return TriageResult(
            status=status,
            product_area=classification.product_area,
            response=response,
            justification=justification,
            request_type=classification.request_type
        )
    
    def _generate_response(self, issue: str, subject: str, company: str,
                          domain: str, classification: Classification,
                          search_results: List[SearchResult]) -> tuple:
        """Generate a response using the LLM."""
        
        # Build context from search results
        context = ""
        if search_results:
            context = "RELEVANT SUPPORT DOCUMENTS:\n\n"
            for i, result in enumerate(search_results[:3], 1):
                context += f"Document {i} ({result.source}):\n"
                context += f"Title: {result.title}\n"
                context += f"Content: {result.content[:1000]}\n\n"
        else:
            context = "No relevant support documents found in corpus."
        
        # Build prompt
        prompt = TRIAGE_PROMPT.format(
            subject=subject or "No subject",
            issue=issue,
            company=company or "Unknown",
            context=context
        )
        
        # Call LLM
        if self.model:
            try:
                response = self.model.generate_content(
                    f"{SYSTEM_PROMPT}\n\n{prompt}",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=1024,
                    )
                )
                
                result_text = response.text.strip()
                
                # Try to parse as JSON
                try:
                    # Remove markdown code blocks if present
                    if result_text.startswith('```'):
                        lines = result_text.split('\n')
                        result_text = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
                    
                    result = json.loads(result_text)
                    
                    return (
                        result.get('response', 'I apologize, but I cannot provide a specific response at this time.'),
                        result.get('justification', 'Generated via LLM analysis')
                    )
                except json.JSONDecodeError:
                    # If not JSON, use as plain text response
                    return (result_text, 'Generated via LLM analysis')
                    
            except Exception as e:
                error_str = str(e)
                # Check if it's a quota error - don't log repeatedly
                if '429' in error_str or 'quota' in error_str.lower():
                    logger.warning(f"API quota exceeded, using fallback response")
                else:
                    logger.error(f"LLM call failed: {e}")
                return self._fallback_response(issue, domain, search_results)
        else:
            return self._fallback_response(issue, domain, search_results)
    
    def _fallback_response(self, issue: str, domain: str,
                          search_results: List[SearchResult]) -> tuple:
        """Generate a fallback response without LLM."""
        if search_results:
            best_match = search_results[0]
            response = f"Based on our support documentation:\n\n{best_match.content[:500]}"
            return (response, f"Based on document: {best_match.title}")
        else:
            return (
                "I apologize, but I need more information to assist you. Please contact our support team directly.",
                "No relevant documentation found"
            )
    
    def _generate_escalation_response(self, domain: str, reason: str) -> str:
        """Generate an escalation response."""
        response = "This issue has been escalated to our support team for further review.\n\n"
        response += "A human agent will review your case and get back to you as soon as possible.\n\n"
        
        if domain:
            domain_lower = domain.lower()
            if domain_lower == 'hackerrank':
                response += "For urgent matters, contact: support@hackerrank.com"
            elif domain_lower == 'claude':
                response += "For urgent matters, visit: support.anthropic.com"
            elif domain_lower == 'visa':
                response += "For urgent card issues, call Visa Global Customer Assistance: +1 303 967 1090"
        
        return response
    
    def process_tickets(self, input_file: str, output_file: str):
        """Process all tickets from a CSV file."""
        import pandas as pd
        
        logger.info(f"Processing tickets from {input_file}")
        
        # Load tickets
        df = pd.read_csv(input_file)
        
        # Process each ticket
        results = []
        for idx, row in df.iterrows():
            issue = str(row.get('issue', row.get('Issue', '')))
            subject = str(row.get('subject', row.get('Subject', '')))
            company = str(row.get('company', row.get('Company', 'None')))
            
            if company == 'nan':
                company = 'None'
            
            logger.info(f"\n--- Processing ticket {idx + 1} ---")
            
            try:
                result = self.triage(issue, subject, company)
                
                results.append({
                    'issue': issue,
                    'subject': subject,
                    'company': company,
                    'response': result.response,
                    'product_area': result.product_area,
                    'status': result.status,
                    'request_type': result.request_type,
                    'justification': result.justification
                })
                
            except Exception as e:
                logger.error(f"Error processing ticket {idx + 1}: {e}")
                results.append({
                    'issue': issue,
                    'subject': subject,
                    'company': company,
                    'response': 'An error occurred while processing your request. Your issue has been escalated.',
                    'product_area': 'General Support',
                    'status': 'escalated',
                    'request_type': 'product_issue',
                    'justification': f'Processing error: {str(e)}'
                })
            
            # Wait 10 seconds between tickets to respect rate limits
            time.sleep(10)
        
        # Save results
        output_df = pd.DataFrame(results)
        columns_order = ['issue', 'subject', 'company', 'response', 'product_area', 'status', 'request_type', 'justification']
        output_df = output_df[columns_order]
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output_df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
        
        return results