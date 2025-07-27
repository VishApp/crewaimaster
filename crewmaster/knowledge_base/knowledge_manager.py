"""
Knowledge Base Manager for CrewMaster.

This module handles ingestion, processing, and retrieval of knowledge
for agents to use in their tasks.
"""

import os
import hashlib
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from ..database.database import Database
from ..database.models import KnowledgeBaseModel

class SourceType(Enum):
    """Types of knowledge sources."""
    FILE = "file"
    URL = "url"
    TEXT = "text"
    API = "api"

@dataclass
class Document:
    """Represents a processed document."""
    id: str
    content: str
    metadata: Dict[str, Any]
    source: str
    source_type: SourceType
    chunk_index: int = 0
    
class KnowledgeProcessor:
    """Processes different types of knowledge sources."""
    
    def __init__(self):
        """Initialize the knowledge processor."""
        pass
    
    def process_file(self, file_path: str) -> List[Document]:
        """Process a file into documents."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.txt':
            return self._process_text_file(file_path)
        elif file_extension == '.md':
            return self._process_markdown_file(file_path)
        elif file_extension == '.pdf':
            return self._process_pdf_file(file_path)
        elif file_extension in ['.json']:
            return self._process_json_file(file_path)
        else:
            # Try to read as text
            return self._process_text_file(file_path)
    
    def process_url(self, url: str) -> List[Document]:
        """Process content from a URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            content = response.text
            
            # Basic HTML stripping (in a real implementation, use BeautifulSoup)
            if '<html' in content.lower():
                content = self._strip_html(content)
            
            return self._chunk_content(content, {
                'source': url,
                'source_type': SourceType.URL.value,
                'fetched_at': datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            raise ValueError(f"Failed to fetch URL {url}: {e}")
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Process raw text into documents."""
        metadata = metadata or {}
        metadata.update({
            'source_type': SourceType.TEXT.value,
            'created_at': datetime.now(timezone.utc).isoformat()
        })
        
        return self._chunk_content(text, metadata)
    
    def _process_text_file(self, file_path: Path) -> List[Document]:
        """Process a text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            'source': str(file_path),
            'source_type': SourceType.FILE.value,
            'file_name': file_path.name,
            'file_extension': file_path.suffix,
            'file_size': file_path.stat().st_size,
            'modified_at': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
        
        return self._chunk_content(content, metadata)
    
    def _process_markdown_file(self, file_path: Path) -> List[Document]:
        """Process a markdown file."""
        # For now, treat as text file
        # In a more sophisticated implementation, we could parse markdown structure
        return self._process_text_file(file_path)
    
    def _process_pdf_file(self, file_path: Path) -> List[Document]:
        """Process a PDF file."""
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                content = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    content += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            metadata = {
                'source': str(file_path),
                'source_type': SourceType.FILE.value,
                'file_name': file_path.name,
                'file_extension': file_path.suffix,
                'file_size': file_path.stat().st_size,
                'page_count': len(pdf_reader.pages),
                'modified_at': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            
            return self._chunk_content(content, metadata)
            
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
    
    def _process_json_file(self, file_path: Path) -> List[Document]:
        """Process a JSON file."""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON to text representation
        content = json.dumps(data, indent=2)
        
        metadata = {
            'source': str(file_path),
            'source_type': SourceType.FILE.value,
            'file_name': file_path.name,
            'file_extension': file_path.suffix,
            'file_size': file_path.stat().st_size,
            'modified_at': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'data_type': 'json'
        }
        
        return self._chunk_content(content, metadata)
    
    def _strip_html(self, html_content: str) -> str:
        """Basic HTML stripping."""
        import re
        
        # Remove script and style elements
        html_content = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        html_content = re.sub(r'<[^>]+>', '', html_content)
        
        # Clean up whitespace
        html_content = re.sub(r'\s+', ' ', html_content)
        
        return html_content.strip()
    
    def _chunk_content(self, content: str, metadata: Dict[str, Any], 
                      chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Chunk content into smaller documents."""
        if len(content) <= chunk_size:
            # Content is small enough, return as single document
            doc_id = self._generate_document_id(content, metadata['source'])
            return [Document(
                id=doc_id,
                content=content,
                metadata=metadata,
                source=metadata['source'],
                source_type=SourceType(metadata['source_type']),
                chunk_index=0
            )]
        
        # Split into chunks
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings within the last 100 characters
                sentence_end = content.rfind('.', end - 100, end)
                if sentence_end != -1:
                    end = sentence_end + 1
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = chunk_index
                chunk_metadata['chunk_start'] = start
                chunk_metadata['chunk_end'] = end
                
                doc_id = self._generate_document_id(chunk_content, metadata['source'], chunk_index)
                
                chunks.append(Document(
                    id=doc_id,
                    content=chunk_content,
                    metadata=chunk_metadata,
                    source=metadata['source'],
                    source_type=SourceType(metadata['source_type']),
                    chunk_index=chunk_index
                ))
                
                chunk_index += 1
            
            # Move start position considering overlap
            start = end - chunk_overlap
            if start >= len(content):
                break
        
        return chunks
    
    def _generate_document_id(self, content: str, source: str, chunk_index: int = 0) -> str:
        """Generate a unique document ID."""
        hasher = hashlib.sha256()
        hasher.update(f"{source}:{chunk_index}:{content[:100]}".encode())
        return hasher.hexdigest()[:16]

class VectorStore:
    """Vector store for semantic search using FAISS."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the vector store."""
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
        self.documents: Dict[int, Document] = {}
        self.doc_id_to_index: Dict[str, int] = {}
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        if not documents:
            return
        
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        start_idx = self.index.ntotal
        self.index.add(embeddings.astype('float32'))
        
        # Store document metadata
        for i, doc in enumerate(documents):
            idx = start_idx + i
            self.documents[idx] = doc
            self.doc_id_to_index[doc.id] = idx
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= score_threshold:
                doc = self.documents[idx]
                results.append({
                    'document': doc,
                    'score': float(score),
                    'content': doc.content,
                    'metadata': doc.metadata
                })
        
        return results
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the vector store."""
        if doc_id not in self.doc_id_to_index:
            return False
        
        # FAISS doesn't support direct removal, so we'd need to rebuild
        # For now, we'll just mark as removed in metadata
        idx = self.doc_id_to_index[doc_id]
        if idx in self.documents:
            self.documents[idx].metadata['_removed'] = True
            return True
        
        return False
    
    def save_to_disk(self, path: str):
        """Save the vector store to disk."""
        import pickle
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save documents
        with open(path / "documents.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'doc_id_to_index': self.doc_id_to_index,
                'dimension': self.dimension
            }, f)
    
    def load_from_disk(self, path: str):
        """Load the vector store from disk."""
        import pickle
        
        path = Path(path)
        
        if not (path / "index.faiss").exists() or not (path / "documents.pkl").exists():
            raise FileNotFoundError(f"Vector store files not found in {path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load documents
        with open(path / "documents.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.doc_id_to_index = data['doc_id_to_index']
            self.dimension = data['dimension']

class KnowledgeManager:
    """Main manager for knowledge bases."""
    
    def __init__(self, database: Database, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the knowledge manager."""
        self.database = database
        self.embedding_model = embedding_model
        self.processor = KnowledgeProcessor()
        self.vector_stores: Dict[str, VectorStore] = {}
    
    def create_knowledge_base(self, name: str, description: str = "") -> KnowledgeBaseModel:
        """Create a new knowledge base."""
        with self.database.get_session() as session:
            kb = KnowledgeBaseModel(
                name=name,
                description=description,
                embedding_model=self.embedding_model,
                vector_store_type="faiss"
            )
            session.add(kb)
            session.flush()
            session.refresh(kb)
            
            # Create vector store
            vector_store = VectorStore(self.embedding_model)
            self.vector_stores[kb.id] = vector_store
            
            return kb
    
    def add_source(self, kb_id: str, source: str, source_type: SourceType,
                  source_config: Optional[Dict[str, Any]] = None) -> int:
        """Add a source to a knowledge base."""
        with self.database.get_session() as session:
            kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == kb_id).first()
            if not kb:
                raise ValueError(f"Knowledge base {kb_id} not found")
            
            # Process the source
            if source_type == SourceType.FILE:
                documents = self.processor.process_file(source)
            elif source_type == SourceType.URL:
                documents = self.processor.process_url(source)
            elif source_type == SourceType.TEXT:
                documents = self.processor.process_text(source, source_config)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            # Get or create vector store
            if kb_id not in self.vector_stores:
                self.vector_stores[kb_id] = VectorStore(self.embedding_model)
            
            vector_store = self.vector_stores[kb_id]
            
            # Add documents to vector store
            vector_store.add_documents(documents)
            
            # Update knowledge base metadata
            kb.document_count += len(documents)
            kb.chunk_count += len(documents)  # Each document is a chunk
            kb.is_processed = True
            kb.processing_status = "completed"
            
            # Save vector store to disk
            vector_store_path = self._get_vector_store_path(kb_id)
            vector_store.save_to_disk(vector_store_path)
            kb.vector_store_path = str(vector_store_path)
            
            session.commit()
            
            return len(documents)
    
    def search_knowledge_base(self, kb_id: str, query: str, k: int = 5, 
                            score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search a knowledge base for relevant information."""
        # Load vector store if not in memory
        if kb_id not in self.vector_stores:
            self._load_vector_store(kb_id)
        
        if kb_id not in self.vector_stores:
            return []
        
        vector_store = self.vector_stores[kb_id]
        results = vector_store.search(query, k, score_threshold)
        
        # Filter out removed documents
        filtered_results = []
        for result in results:
            if not result['metadata'].get('_removed', False):
                filtered_results.append(result)
        
        return filtered_results
    
    def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBaseModel]:
        """Get a knowledge base by ID."""
        with self.database.get_session() as session:
            return session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == kb_id).first()
    
    def list_knowledge_bases(self) -> List[KnowledgeBaseModel]:
        """List all knowledge bases."""
        with self.database.get_session() as session:
            return session.query(KnowledgeBaseModel).all()
    
    def delete_knowledge_base(self, kb_id: str) -> bool:
        """Delete a knowledge base."""
        with self.database.get_session() as session:
            kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == kb_id).first()
            if not kb:
                return False
            
            # Remove vector store from memory
            if kb_id in self.vector_stores:
                del self.vector_stores[kb_id]
            
            # Remove vector store files
            if kb.vector_store_path:
                vector_store_path = Path(kb.vector_store_path)
                if vector_store_path.exists():
                    import shutil
                    shutil.rmtree(vector_store_path, ignore_errors=True)
            
            session.delete(kb)
            session.commit()
            
            return True
    
    def _load_vector_store(self, kb_id: str):
        """Load a vector store from disk."""
        with self.database.get_session() as session:
            kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == kb_id).first()
            if not kb or not kb.vector_store_path:
                return
            
            try:
                vector_store = VectorStore(kb.embedding_model)
                vector_store.load_from_disk(kb.vector_store_path)
                self.vector_stores[kb_id] = vector_store
            except Exception as e:
                print(f"Failed to load vector store for {kb_id}: {e}")
    
    def _get_vector_store_path(self, kb_id: str) -> Path:
        """Get the path for storing vector store files."""
        base_path = Path.home() / ".crewmaster" / "vector_stores" / kb_id
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path