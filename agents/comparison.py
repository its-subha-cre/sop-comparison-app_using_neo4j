import math
import numpy as np
from services.scoring import ScoringService

class ComparisonAgent:
    """Computes semantic embedding + lexical similarity between mapped clauses."""

    def compute_lexical_jaccard(self, text_a: str, text_b: str) -> float:
        """Computes Jaccard similarity coefficient based on word tokens."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a and not words_b:
            return 1.0
        intersection = words_a.intersection(words_b)
        union = words_a.union(words_b)
        return len(intersection) / len(union)

    def mock_embeddings(self, text: str) -> list:
        """Returns a deterministic mock vector if no live embedding client is configured."""
        # Clean deterministic vector based on text characters
        vec = [ord(char) % 100 for char in text[:128]]
        if len(vec) < 128:
            vec += [0] * (128 - len(vec))
        norm = math.sqrt(sum(x*x for x in vec))
        return [x/norm for x in vec] if norm > 0 else vec

    def compute_cosine_similarity(self, vec_a: list, vec_b: list) -> float:
        """Computes cosine similarity between two vector lists."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def compare_clauses(self, text_a: str, text_b: str) -> dict:
        """Performs comparison returning detailed similarities and overall compliance score."""
        if not text_a.strip() or not text_b.strip():
            return {
                "lexical_similarity": 0.0,
                "semantic_similarity": 0.0,
                "combined_score": 0.0
            }
            
        if text_a.strip().lower() == text_b.strip().lower():
            return {
                "lexical_similarity": 1.0,
                "semantic_similarity": 1.0,
                "combined_score": 1.0
            }
            
        lexical = self.compute_lexical_jaccard(text_a, text_b)
        
        # In a real environment, we would call the actual LLM embedding service.
        # Fall back to mock embeddings for local dev environment reliability.
        vec_a = self.mock_embeddings(text_a)
        vec_b = self.mock_embeddings(text_b)
        
        semantic = self.compute_cosine_similarity(vec_a, vec_b)
        
        overall_score = ScoringService.calculate_similarity_score(lexical, semantic)
        
        return {
            "lexical_similarity": round(lexical, 3),
            "semantic_similarity": round(semantic, 3),
            "combined_score": overall_score
        }
