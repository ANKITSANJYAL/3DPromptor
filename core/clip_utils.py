import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import logging

class CLIPUtils:
    """
    CLIP utilities for text embedding and style scoring.
    Handles token embedding and cosine similarity matching.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        """
        Initialize CLIP model and processor.
        
        Args:
            model_name: Hugging Face model identifier
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model {model_name}: {e}")
        
        # Pre-defined style embeddings for comparison
        self.style_embeddings = self._create_style_embeddings()
    
    def _create_style_embeddings(self) -> Dict[str, torch.Tensor]:
        """Create embeddings for common style terms."""
        style_terms = [
            "cubist", "impressionist", "abstract", "realistic", "cartoon",
            "anime", "vaporwave", "cyberpunk", "gothic", "baroque",
            "modern", "minimalist", "surreal", "pop art", "art deco",
            "futuristic", "fantasy", "vintage", "elegant", "rustic",
            "industrial", "organic", "colorful", "monochrome", "metallic",
            "wooden", "crystal", "glowing", "shimmering", "sophisticated"
        ]
        
        embeddings = {}
        with torch.no_grad():
            for style in style_terms:
                inputs = self.processor(text=style, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_features = self.model.get_text_features(**inputs)
                embeddings[style] = text_features.squeeze(0)
        
        return embeddings
    
    def get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Get CLIP text embedding for given text.
        
        Args:
            text: Input text
            
        Returns:
            CLIP text embedding tensor
        """
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.model.get_text_features(**inputs)
            return text_features.squeeze(0)
    
    def get_image_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Get CLIP image embedding for given image.
        
        Args:
            image: PIL Image
            
        Returns:
            CLIP image embedding tensor
        """
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self.model.get_image_features(**inputs)
            return image_features.squeeze(0)
    
    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding tensor
            embedding2: Second embedding tensor
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        embedding1_norm = F.normalize(embedding1, p=2, dim=0)
        embedding2_norm = F.normalize(embedding2, p=2, dim=0)
        
        # Compute cosine similarity
        similarity = torch.dot(embedding1_norm, embedding2_norm).item()
        return similarity
    
    def get_style_score(self, text: str, target_style: str) -> float:
        """
        Get style similarity score between text and target style.
        
        Args:
            text: Input text
            target_style: Target style to compare against
            
        Returns:
            Style similarity score (0-1)
        """
        text_embedding = self.get_text_embedding(text)
        
        if target_style in self.style_embeddings:
            style_embedding = self.style_embeddings[target_style]
        else:
            # If style not in pre-computed embeddings, compute it
            style_embedding = self.get_text_embedding(target_style)
        
        return self.compute_similarity(text_embedding, style_embedding)
    
    def get_top_style_matches(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k style matches for given text.
        
        Args:
            text: Input text
            top_k: Number of top matches to return
            
        Returns:
            List of (style_name, similarity_score) tuples
        """
        text_embedding = self.get_text_embedding(text)
        similarities = []
        
        for style_name, style_embedding in self.style_embeddings.items():
            similarity = self.compute_similarity(text_embedding, style_embedding)
            similarities.append((style_name, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def embed_tokens(self, tokens: List[str]) -> Dict[str, torch.Tensor]:
        """
        Get embeddings for a list of tokens.
        
        Args:
            tokens: List of token strings
            
        Returns:
            Dictionary mapping tokens to their embeddings
        """
        embeddings = {}
        for token in tokens:
            embeddings[token] = self.get_text_embedding(token)
        return embeddings
    
    def compute_token_similarities(self, tokens1: List[str], tokens2: List[str]) -> torch.Tensor:
        """
        Compute similarity matrix between two sets of tokens.
        
        Args:
            tokens1: First set of tokens
            tokens2: Second set of tokens
            
        Returns:
            Similarity matrix of shape (len(tokens1), len(tokens2))
        """
        embeddings1 = self.embed_tokens(tokens1)
        embeddings2 = self.embed_tokens(tokens2)
        
        similarity_matrix = torch.zeros(len(tokens1), len(tokens2))
        
        for i, token1 in enumerate(tokens1):
            for j, token2 in enumerate(tokens2):
                similarity = self.compute_similarity(embeddings1[token1], embeddings2[token2])
                similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def get_style_confidence(self, text: str) -> Dict[str, float]:
        """
        Get confidence scores for different style categories.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping style categories to confidence scores
        """
        style_categories = {
            'artistic': ['cubist', 'impressionist', 'abstract', 'surreal', 'pop art'],
            'modern': ['modern', 'minimalist', 'futuristic', 'cyberpunk'],
            'classical': ['baroque', 'gothic', 'renaissance', 'classical'],
            'colorful': ['colorful', 'vibrant', 'bright', 'shimmering'],
            'monochrome': ['monochrome', 'black and white', 'sepia'],
            'textured': ['metallic', 'wooden', 'crystal', 'rough', 'smooth']
        }
        
        text_embedding = self.get_text_embedding(text)
        confidence_scores = {}
        
        for category, styles in style_categories.items():
            category_scores = []
            for style in styles:
                if style in self.style_embeddings:
                    similarity = self.compute_similarity(text_embedding, self.style_embeddings[style])
                    category_scores.append(similarity)
            
            if category_scores:
                confidence_scores[category] = max(category_scores)
            else:
                confidence_scores[category] = 0.0
        
        return confidence_scores

# Global instance for easy access
clip_utils = CLIPUtils() 