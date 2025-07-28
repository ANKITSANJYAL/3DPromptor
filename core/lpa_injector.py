import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable, Any
import logging
from core.clip_utils import clip_utils

class LPAInjector:
    """
    Local Prompt Adaptation (LPA) injector for injecting prompt tokens
    into UNet attention blocks to preserve spatial structure and style.
    """
    
    def __init__(self):
        """Initialize the LPA injector."""
        # Use MPS for macOS, CUDA for Linux/Windows, CPU as fallback
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.attention_hooks = []
        self.original_forward_fns = {}
        self.injection_config = {}
        
    def setup_injection(self, model, object_tokens: List[str], style_tokens: List[str]):
        """
        Setup token injection for the model.
        
        Args:
            model: The diffusion model (UNet)
            object_tokens: Tokens for early injection (spatial structure)
            style_tokens: Tokens for late injection (style)
        """
        self.object_tokens = object_tokens
        self.style_tokens = style_tokens
        
        # Get embeddings for tokens
        self.object_embeddings = clip_utils.embed_tokens(object_tokens)
        self.style_embeddings = clip_utils.embed_tokens(style_tokens)
        
        # Clear previous hooks
        self.clear_hooks()
        
        # Register hooks for attention blocks
        self._register_attention_hooks(model)
        
    def _register_attention_hooks(self, model):
        """Register hooks for cross-attention blocks."""
        # Check if model has named_modules method (real model)
        if hasattr(model, 'named_modules'):
            for name, module in model.named_modules():
                if "attn2" in name and "to_k" in name:  # Cross-attention key projection
                    hook = self._create_attention_hook(name, module)
                    self.attention_hooks.append(hook)
                    module.register_forward_hook(hook)
        else:
            # For fallback models, skip hook registration
            logging.info("Skipping LPA hooks for fallback model")
                
    def _create_attention_hook(self, name: str, module):
        """Create a hook for attention injection."""
        def hook_fn(module, input, output):
            return self._inject_attention(name, module, input, output)
        return hook_fn
    
    def _inject_attention(self, name: str, module, input, output):
        """
        Inject token embeddings into attention computation.
        
        Args:
            name: Module name
            module: The attention module
            input: Input to the module
            output: Original output
            
        Returns:
            Modified output with injected tokens
        """
        # Determine if this is early or late layer based on name
        is_early = self._is_early_layer(name)
        tokens_to_inject = self.object_tokens if is_early else self.style_tokens
        embeddings_to_inject = self.object_embeddings if is_early else self.style_embeddings
        
        if not tokens_to_inject:
            return output
        
        # Get the attention weights
        attention_weights = self._get_attention_weights(module, input)
        
        # Inject token embeddings
        modified_output = self._apply_token_injection(
            output, attention_weights, embeddings_to_inject, is_early
        )
        
        return modified_output
    
    def _is_early_layer(self, name: str) -> bool:
        """Determine if this is an early or late layer."""
        # Early layers typically have lower numbers in the name
        # This is a heuristic - can be refined based on specific model architecture
        layer_num = self._extract_layer_number(name)
        return layer_num < 8  # Arbitrary threshold
    
    def _extract_layer_number(self, name: str) -> int:
        """Extract layer number from module name."""
        import re
        match = re.search(r'(\d+)', name)
        return int(match.group(1)) if match else 0
    
    def _get_attention_weights(self, module, input):
        """Get attention weights from the module."""
        # This is a simplified version - actual implementation depends on model architecture
        if hasattr(module, 'weight'):
            return module.weight
        return None
    
    def _apply_token_injection(self, output, attention_weights, embeddings_to_inject, is_early):
        """
        Apply token injection to the attention output.
        
        Args:
            output: Original attention output
            attention_weights: Attention weights
            embeddings_to_inject: Token embeddings to inject
            is_early: Whether this is early injection
            
        Returns:
            Modified output with injected tokens
        """
        if not embeddings_to_inject:
            return output
        
        # Convert embeddings to the right format
        token_embeddings = torch.stack(list(embeddings_to_inject.values()))
        
        # Reshape token embeddings to match output dimensions
        batch_size, seq_len, hidden_dim = output.shape
        num_tokens = token_embeddings.shape[0]
        
        # Pad or truncate token embeddings to match sequence length
        if token_embeddings.shape[1] != hidden_dim:
            # Project to correct dimension if needed
            projection = torch.nn.Linear(token_embeddings.shape[1], hidden_dim).to(self.device)
            token_embeddings = projection(token_embeddings)
        
        # Create injection mask based on early/late strategy
        injection_strength = 0.3 if is_early else 0.5  # Style tokens get stronger injection
        
        # Blend original output with injected tokens
        # This is a simplified blending strategy
        token_contribution = token_embeddings.mean(dim=0, keepdim=True)
        token_contribution = token_contribution.expand(batch_size, seq_len, -1)
        
        modified_output = (1 - injection_strength) * output + injection_strength * token_contribution
        
        return modified_output
    
    def clear_hooks(self):
        """Clear all registered hooks."""
        for hook in self.attention_hooks:
            # Remove hooks if they have a remove method
            if hasattr(hook, 'remove'):
                hook.remove()
        self.attention_hooks = []
    
    def set_injection_config(self, config: Dict[str, Any]):
        """
        Set injection configuration parameters.
        
        Args:
            config: Configuration dictionary with injection parameters
        """
        self.injection_config = config
    
    def get_injection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current injection setup."""
        return {
            'object_tokens': self.object_tokens,
            'style_tokens': self.style_tokens,
            'num_object_embeddings': len(self.object_embeddings),
            'num_style_embeddings': len(self.style_embeddings),
            'num_hooks': len(self.attention_hooks),
            'injection_config': self.injection_config
        }

class AttentionVisualizer:
    """Helper class for visualizing attention maps."""
    
    def __init__(self):
        self.attention_maps = {}
    
    def capture_attention_map(self, name: str, attention_weights: torch.Tensor):
        """Capture attention map for visualization."""
        self.attention_maps[name] = attention_weights.detach().cpu()
    
    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Get all captured attention maps."""
        return self.attention_maps
    
    def clear_attention_maps(self):
        """Clear all captured attention maps."""
        self.attention_maps = {}

# Global instances
lpa_injector = LPAInjector()
attention_visualizer = AttentionVisualizer() 