import spacy
import re
from typing import Tuple, List, Dict
import ftfy

class PromptParser:
    """
    Parses prompts into object tokens (early) and style tokens (late)
    using spaCy for NLP analysis.
    """
    
    def __init__(self):
        """Initialize spaCy model for English language processing."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy English model not found. Run: python -m spacy download en_core_web_sm"
            )
        
        # Common style keywords that should be treated as style tokens
        self.style_keywords = {
            'artistic', 'artistic style', 'cubist', 'cubism', 'impressionist', 
            'impressionism', 'abstract', 'realistic', 'photorealistic', 
            'cartoon', 'anime', 'manga', 'vaporwave', 'cyberpunk', 'steampunk',
            'gothic', 'baroque', 'renaissance', 'modern', 'minimalist',
            'surreal', 'surrealist', 'expressionist', 'expressionism',
            'pop art', 'pop-art', 'art deco', 'art nouveau', 'romantic',
            'classical', 'neoclassical', 'byzantine', 'medieval', 'victorian',
            'edwardian', 'georgian', 'tudor', 'gothic revival', 'brutalist',
            'futuristic', 'sci-fi', 'science fiction', 'fantasy', 'mythological',
            'ancient', 'antique', 'vintage', 'retro', 'nostalgic', 'timeless',
            'ethereal', 'mystical', 'magical', 'enchanted', 'dreamy', 'whimsical',
            'elegant', 'luxurious', 'opulent', 'sophisticated', 'refined',
            'rustic', 'country', 'provincial', 'rural', 'urban', 'metropolitan',
            'industrial', 'mechanical', 'organic', 'natural', 'wild', 'untamed',
            'domestic', 'tame', 'civilized', 'primitive', 'tribal', 'ethnic',
            'cultural', 'traditional', 'contemporary', 'avant-garde', 'experimental',
            'innovative', 'revolutionary', 'groundbreaking', 'pioneering',
            'influential', 'iconic', 'legendary', 'mythical', 'fabled',
            'colorful', 'vibrant', 'bright', 'dull', 'muted', 'pastel',
            'monochrome', 'black and white', 'sepia', 'grayscale',
            'textured', 'smooth', 'rough', 'polished', 'matte', 'glossy',
            'metallic', 'wooden', 'stone', 'ceramic', 'glass', 'plastic',
            'fabric', 'leather', 'silk', 'cotton', 'wool', 'linen',
            'golden', 'silver', 'bronze', 'copper', 'brass', 'iron',
            'crystal', 'diamond', 'ruby', 'emerald', 'sapphire', 'pearl',
            'opaque', 'transparent', 'translucent', 'reflective', 'iridescent',
            'luminescent', 'glowing', 'shimmering', 'sparkling', 'glittering',
            'radiant', 'brilliant', 'dazzling', 'stunning', 'magnificent',
            'breathtaking', 'spectacular', 'extraordinary', 'remarkable',
            'unique', 'distinctive', 'characteristic', 'typical', 'standard',
            'custom', 'bespoke', 'handcrafted', 'handmade', 'artisanal',
            'mass-produced', 'industrial', 'commercial', 'professional',
            'amateur', 'hobbyist', 'enthusiast', 'expert', 'master', 'virtuoso'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text input."""
        return ftfy.fix_text(text.strip())
    
    def extract_style_tokens(self, text: str) -> List[str]:
        """Extract style-related tokens from the text."""
        doc = self.nlp(text.lower())
        style_tokens = []
        
        # Check for multi-word style phrases
        text_lower = text.lower()
        for style_phrase in self.style_keywords:
            if style_phrase in text_lower:
                style_tokens.append(style_phrase)
        
        # Extract individual style words
        for token in doc:
            if token.text in self.style_keywords:
                style_tokens.append(token.text)
        
        return list(set(style_tokens))  # Remove duplicates
    
    def extract_object_tokens(self, text: str, style_tokens: List[str]) -> List[str]:
        """Extract object-related tokens by removing style tokens."""
        doc = self.nlp(text)
        object_tokens = []
        
        # Get all tokens that are not style tokens
        text_lower = text.lower()
        for token in doc:
            token_text = token.text.lower()
            is_style_token = any(style_token in token_text or token_text in style_token 
                               for style_token in style_tokens)
            
            if not is_style_token and token.text.strip():
                object_tokens.append(token.text)
        
        return object_tokens
    
    def parse_prompt(self, prompt: str) -> Dict[str, List[str]]:
        """
        Parse prompt into object tokens (early) and style tokens (late).
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Dictionary with 'object_tokens' and 'style_tokens' lists
        """
        prompt = self.clean_text(prompt)
        
        # Extract style tokens first
        style_tokens = self.extract_style_tokens(prompt)
        
        # Extract object tokens by removing style tokens
        object_tokens = self.extract_object_tokens(prompt, style_tokens)
        
        return {
            'object_tokens': object_tokens,
            'style_tokens': style_tokens,
            'original_prompt': prompt
        }
    
    def get_token_positions(self, prompt: str) -> Dict[str, List[int]]:
        """
        Get the positions of object and style tokens in the original prompt.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Dictionary with token positions for early/late injection
        """
        parsed = self.parse_prompt(prompt)
        doc = self.nlp(prompt.lower())
        
        object_positions = []
        style_positions = []
        
        for i, token in enumerate(doc):
            token_text = token.text.lower()
            
            # Check if token is a style token
            is_style = any(style_token in token_text or token_text in style_token 
                          for style_token in parsed['style_tokens'])
            
            if is_style:
                style_positions.append(i)
            else:
                object_positions.append(i)
        
        return {
            'object_positions': object_positions,  # Early injection
            'style_positions': style_positions,    # Late injection
            'total_tokens': len(doc)
        }

# Global instance for easy access
prompt_parser = PromptParser() 