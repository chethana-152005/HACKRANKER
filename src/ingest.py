"""Data ingestion and corpus loading module."""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a support document."""
    doc_id: str
    title: str
    content: str
    source: str  # hackerrank, claude, visa
    url: Optional[str] = None
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'content': self.content,
            'source': self.source,
            'url': self.url,
            'category': self.category,
            'metadata': self.metadata
        }


class CorpusLoader:
    """Load and manage support corpus from multiple domains."""
    
    def __init__(self, corpus_dir: str = "data/corpus"):
        self.corpus_dir = Path(corpus_dir)
        self.documents: List[Document] = []
        self.domain_mapping = {
            'hackerrank': 'HackerRank',
            'claude': 'Claude',
            'visa': 'Visa'
        }
        
    def load_all(self) -> List[Document]:
        """Load all documents from all domains."""
        self.documents = []
        
        for domain_folder in self.domain_mapping.keys():
            domain_path = self.corpus_dir / domain_folder
            if domain_path.exists():
                docs = self._load_domain(domain_path, domain_folder)
                self.documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {domain_folder}")
        
        return self.documents
    
    def _load_domain(self, domain_path: Path, domain: str) -> List[Document]:
        """Load documents from a specific domain folder."""
        documents = []
        
        # Load JSON files
        for json_file in domain_path.glob("**/*.json"):
            try:
                docs = self._load_json_file(json_file, domain)
                documents.extend(docs)
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        # Load text files
        for txt_file in domain_path.glob("**/*.txt"):
            try:
                doc = self._load_text_file(txt_file, domain)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"Error loading {txt_file}: {e}")
        
        # Load markdown files
        for md_file in domain_path.glob("**/*.md"):
            try:
                doc = self._load_markdown_file(md_file, domain)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"Error loading {md_file}: {e}")
        
        return documents
    
    def _load_json_file(self, file_path: Path, domain: str) -> List[Document]:
        """Load documents from a JSON file."""
        documents = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if 'articles' in data:
                items = data['articles']
            elif 'documents' in data:
                items = data['documents']
            else:
                items = [data]
        else:
            items = [data]
        
        for item in items:
            doc = self._parse_json_item(item, domain, file_path)
            if doc:
                documents.append(doc)
        
        return documents
    
    def _parse_json_item(self, item: Dict, domain: str, file_path: Path) -> Optional[Document]:
        """Parse a JSON item into a Document."""
        if not isinstance(item, dict):
            return None
        
        content = item.get('content', '') or item.get('body', '') or item.get('text', '')
        if not content:
            return None
        
        doc_id = item.get('id') or item.get('doc_id') or self._generate_id(content)
        title = item.get('title', '') or item.get('name', '')
        url = item.get('url') or item.get('link', '')
        category = item.get('category') or item.get('tags', '')
        
        return Document(
            doc_id=doc_id,
            title=title,
            content=content,
            source=domain,
            url=url,
            category=category,
            metadata={'file_path': str(file_path)}
        )
    
    def _load_text_file(self, file_path: Path, domain: str) -> Optional[Document]:
        """Load a text file as a document."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            return None
        
        return Document(
            doc_id=self._generate_id(content),
            title=file_path.stem,
            content=content,
            source=domain,
            url=None,
            category=None,
            metadata={'file_path': str(file_path)}
        )
    
    def _load_markdown_file(self, file_path: Path, domain: str) -> Optional[Document]:
        """Load a markdown file as a document."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            return None
        
        # Extract title from first heading
        title = file_path.stem
        lines = content.split('\n')
        for line in lines[:5]:
            if line.startswith('# '):
                title = line[2:].strip()
                break
        
        return Document(
            doc_id=self._generate_id(content),
            title=title,
            content=content,
            source=domain,
            url=None,
            category=None,
            metadata={'file_path': str(file_path)}
        )
    
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for a document."""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def get_documents_by_domain(self, domain: str) -> List[Document]:
        """Get all documents for a specific domain."""
        normalized = domain.lower().replace(' ', '')
        return [d for d in self.documents if d.source.lower() == normalized]
    
    def search(self, query: str, domain: Optional[str] = None) -> List[Document]:
        """Simple keyword search over documents."""
        query_terms = set(query.lower().split())
        results = []
        
        docs = self.documents
        if domain:
            docs = self.get_documents_by_domain(domain)
        
        for doc in docs:
            doc_text = (doc.title + ' ' + doc.content).lower()
            score = sum(1 for term in query_terms if term in doc_text)
            if score > 0:
                results.append((score, doc))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in results]


class TicketLoader:
    """Load support tickets from CSV files."""
    
    def __init__(self):
        pass
    
    def load_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Load tickets from a CSV file."""
        import pandas as pd
        
        df = pd.read_csv(file_path)
        tickets = df.to_dict('records')
        
        # Normalize column names
        normalized_tickets = []
        for ticket in tickets:
            normalized = {}
            for key, value in ticket.items():
                key_lower = key.lower().strip()
                if key_lower in ['issue', 'description', 'body']:
                    normalized['issue'] = str(value) if pd.notna(value) else ''
                elif key_lower in ['subject', 'title']:
                    normalized['subject'] = str(value) if pd.notna(value) else ''
                elif key_lower in ['company', 'domain', 'source']:
                    normalized['company'] = str(value) if pd.notna(value) else 'None'
                else:
                    normalized[key_lower] = value
            normalized_tickets.append(normalized)
        
        return normalized_tickets