"""Semantic retrieval module using embeddings and vector search."""

import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not available, using keyword search fallback")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using simple similarity search")


@dataclass
class SearchResult:
    """Represents a search result."""
    doc_id: str
    title: str
    content: str
    source: str
    score: float
    url: Optional[str] = None
    category: Optional[str] = None


class SemanticRetriever:
    """Semantic search over support documents using embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_gpu: bool = False):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model = None
        self.index = None
        self.documents = []
        self.doc_embeddings = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        if EMBEDDINGS_AVAILABLE:
            self._init_model()
    
    def _init_model(self):
        """Initialize the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self.model = None
    
    def build_index(self, documents: List[Any]) -> bool:
        """Build a search index from documents."""
        self.documents = documents
        
        if not documents:
            logger.warning("No documents to index")
            return False
        
        if self.model is None:
            logger.warning("No embedding model available, using keyword search")
            return False
        
        try:
            # Generate embeddings
            texts = []
            for doc in documents:
                text = f"{doc.title}\n{doc.content[:1000]}"  # Truncate for efficiency
                texts.append(text)
            
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            self.doc_embeddings = self.model.encode(texts, show_progress_bar=True)
            self.doc_embeddings = np.array(self.doc_embeddings).astype('float32')
            
            # Build FAISS index if available
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                faiss.normalize_L2(self.doc_embeddings)
                self.index.add(self.doc_embeddings)
                logger.info(f"Built FAISS index with {len(documents)} documents")
            else:
                # Normalize for cosine similarity
                norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
                self.doc_embeddings = self.doc_embeddings / (norms + 1e-8)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, domain: Optional[str] = None,
               min_score: float = 0.0) -> List[SearchResult]:
        """Search for relevant documents."""
        if not self.documents:
            return []
        
        # Filter by domain if specified
        candidate_docs = []
        candidate_indices = []
        for i, doc in enumerate(self.documents):
            if domain is None or doc.source.lower() == domain.lower():
                candidate_docs.append(doc)
                candidate_indices.append(i)
        
        if not candidate_docs:
            return []
        
        # If no embedding model, fall back to keyword search
        if self.model is None or self.doc_embeddings is None:
            return self._keyword_search(query, candidate_docs, top_k, min_score)
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])
            query_embedding = np.array(query_embedding).astype('float32')
            
            # Search
            if FAISS_AVAILABLE and self.index is not None:
                # Use FAISS for search
                faiss.normalize_L2(query_embedding)
                scores, indices = self.index.search(query_embedding, min(top_k * 2, len(self.documents)))
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.documents):
                        doc = self.documents[idx]
                        if domain is None or doc.source.lower() == domain.lower():
                            if score >= min_score:
                                results.append(SearchResult(
                                    doc_id=doc.doc_id,
                                    title=doc.title,
                                    content=doc.content,
                                    source=doc.source,
                                    score=float(score),
                                    url=doc.url,
                                    category=doc.category
                                ))
                
                return results[:top_k]
            else:
                # Manual cosine similarity
                query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
                
                # Get candidate embeddings
                candidate_embeddings = self.doc_embeddings[candidate_indices]
                scores = np.dot(candidate_embeddings, query_norm.T).flatten()
                
                # Sort by score
                sorted_indices = np.argsort(scores)[::-1][:top_k]
                
                results = []
                for idx in sorted_indices:
                    if scores[idx] >= min_score:
                        doc = candidate_docs[idx]
                        results.append(SearchResult(
                            doc_id=doc.doc_id,
                            title=doc.title,
                            content=doc.content,
                            source=doc.source,
                            score=float(scores[idx]),
                            url=doc.url,
                            category=doc.category
                        ))
                
                return results
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return self._keyword_search(query, candidate_docs, top_k, min_score)
    
    def _keyword_search(self, query: str, documents: List[Any], 
                        top_k: int, min_score: float) -> List[SearchResult]:
        """Fallback keyword search."""
        query_terms = set(query.lower().split())
        results = []
        
        for doc in documents:
            doc_text = (doc.title + ' ' + doc.content).lower()
            score = sum(1 for term in query_terms if term in doc_text) / max(len(query_terms), 1)
            
            if score >= min_score:
                results.append(SearchResult(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    content=doc.content,
                    source=doc.source,
                    score=score,
                    url=doc.url,
                    category=doc.category
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def save_index(self, path: str):
        """Save the index to disk."""
        os.makedirs(path, exist_ok=True)
        
        if self.doc_embeddings is not None:
            np.save(os.path.join(path, 'embeddings.npy'), self.doc_embeddings)
        
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, os.path.join(path, 'faiss.index'))
        
        # Save document metadata
        doc_data = [
            {
                'doc_id': d.doc_id,
                'title': d.title,
                'content': d.content[:500],  # Truncate for storage
                'source': d.source,
                'url': d.url,
                'category': d.category
            }
            for d in self.documents
        ]
        
        with open(os.path.join(path, 'documents.json'), 'w') as f:
            json.dump(doc_data, f)
        
        logger.info(f"Saved index to {path}")
    
    def load_index(self, path: str) -> bool:
        """Load the index from disk."""
        try:
            embeddings_path = os.path.join(path, 'embeddings.npy')
            if os.path.exists(embeddings_path):
                self.doc_embeddings = np.load(embeddings_path)
            
            if FAISS_AVAILABLE:
                index_path = os.path.join(path, 'faiss.index')
                if os.path.exists(index_path):
                    self.index = faiss.read_index(index_path)
            
            docs_path = os.path.join(path, 'documents.json')
            if os.path.exists(docs_path):
                with open(docs_path, 'r') as f:
                    doc_data = json.load(f)
                
                from ingest import Document
                self.documents = [
                    Document(
                        doc_id=d['doc_id'],
                        title=d['title'],
                        content=d['content'],
                        source=d['source'],
                        url=d.get('url'),
                        category=d.get('category')
                    )
                    for d in doc_data
                ]
            
            logger.info(f"Loaded index from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False