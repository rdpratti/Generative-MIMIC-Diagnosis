import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
import pandas as pd

class MedicalRAGRetriever:
    """RAG retriever for medical diagnosis classification."""
    
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2', logger=None):
        """
        Initialize RAG retriever.
        
        Args:
            embedding_model_name: Sentence transformer model for embeddings
            logger: Logger instance
        """
        self.logger = logger
        
        self.logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        self.index = None
        self.examples_df = None
        self.embeddings = None
        
    def build_index(self, train_df, text_column='text', save_path=None):
        """
        Build FAISS index from training data.
        
        Args:
            train_df: Training dataframe with clinical notes
            text_column: Column containing text to embed
            save_path: Optional path to save index
        """
        self.logger.info(f"Building RAG index from {len(train_df)} examples...")
        
        # Store examples
        self.examples_df = train_df.copy()
        
        # Generate embeddings
        self.logger.info("Generating embeddings...")
        texts = train_df[text_column].tolist()
        self.embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Build FAISS index
        self.logger.info("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        self.index.add(self.embeddings.astype('float32'))
        
        self.logger.info(f"Index built with {self.index.ntotal} vectors of dimension {dimension}")
        
        # Save if requested
        if save_path:
            self.save_index(save_path)
        
        return self
    
    def save_index(self, base_path):
        """Save index and metadata to disk."""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(base_path / "faiss_index.bin"))
        
        # Save examples DataFrame
        self.examples_df.to_parquet(base_path / "examples.parquet")
        
        # Save embeddings
        np.save(base_path / "embeddings.npy", self.embeddings)
        
        self.logger.info(f"Index saved to {base_path}")
    
    def load_index(self, base_path):
        """Load index and metadata from disk."""
        base_path = Path(base_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(base_path / "faiss_index.bin"))
        
        # Load examples DataFrame
        self.examples_df = pd.read_parquet(base_path / "examples.parquet")
        
        # Load embeddings
        self.embeddings = np.load(base_path / "embeddings.npy")
        
        self.logger.info(f"Index loaded from {base_path}")
        return self
    
    def retrieve(self, query_text, k=1, exclude_same_id=None, filter_diagnosis=None):
        """
        Retrieve k most similar examples.
        
        Args:
            query_text: Patient note to find similar examples for
            k: Number of examples to retrieve
            exclude_same_id: Optional patient ID to exclude (don't retrieve same patient)
        
        Returns:
            DataFrame with k most similar examples
        """
        # Embed query
        query_embedding = self.embedding_model.encode([query_text])
        
        
        # Get more candidates if we're filtering (since some will be discarded)
        initial_k = k * 5 if filter_diagnosis else k * 2
    
        # Search in FAISS - this returns the most similar vectors
        distances, indices = self.index.search(query_embedding, initial_k)
    
        # Filter results
        retrieved_indices = []
        retrieved_distances = []  # ADD THIS
    
        for i, idx in enumerate(indices[0]):
            if len(retrieved_indices) >= k:
                break
        
            example = self.examples_df.iloc[idx]
        
            # Apply filters
            if exclude_same_id and example.get('patient_id') == exclude_same_id:
                continue
            if filter_diagnosis and example['icd10_code'] != filter_diagnosis:
                continue
        
            retrieved_indices.append(idx)
            retrieved_distances.append(distances[0][i])  # ADD THIS
    
        # Create result DataFrame with distances
        result_df = self.examples_df.iloc[retrieved_indices].copy()
        result_df['similarity_distance'] = retrieved_distances  # ADD THIS
    
        return result_df
    
        

        
