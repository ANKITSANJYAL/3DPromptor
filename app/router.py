from typing import Dict, Any, Optional, Tuple
from PIL import Image
import logging

# Import our core modules
from core.sdxl_wrapper import sdxl_wrapper
from core.inference import zero123_inference
from core.prompt_utils import prompt_parser
from core.clip_utils import clip_utils

class ModeRouter:
    """
    Router for handling different operation modes:
    - Mode 1: Prompt-only input
    - Mode 2: Image + Prompt input
    """
    
    def __init__(self):
        """Initialize the mode router."""
        self.supported_modes = ['prompt_only', 'image_prompt']
    
    def route_mode_1_prompt_only(
        self,
        prompt: str,
        num_views: int = 8,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Route Mode 1: Prompt-only input.
        
        Args:
            prompt: Text prompt for generation
            num_views: Number of views to generate
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing generated content and metadata
        """
        try:
            logging.info(f"Processing Mode 1 (Prompt-only): {prompt[:50]}...")
            
            # Step 1: Generate initial image with SDXL
            sdxl_result = sdxl_wrapper.generate_image(
                prompt=prompt,
                **kwargs
            )
            
            if not sdxl_result['success']:
                return {
                    'success': False,
                    'error': 'Failed to generate initial image with SDXL',
                    'mode': 'prompt_only'
                }
            
            initial_image = sdxl_result['image']
            
            # Step 2: Generate multi-view turntable with Zero123 + LPA
            multi_view_result = zero123_inference.generate_multi_view(
                input_image=initial_image,
                prompt=prompt,
                num_views=num_views,
                **kwargs
            )
            
            return {
                'success': multi_view_result['success'],
                'mode': 'prompt_only',
                'initial_image': initial_image,
                'turntable_gif': multi_view_result.get('turntable_gif'),
                'multi_view_images': multi_view_result.get('images', []),
                'metadata': {
                    'sdxl_metadata': sdxl_result['metadata'],
                    'multi_view_metadata': multi_view_result.get('metadata', []),
                    'prompt_analysis': prompt_parser.parse_prompt(prompt)
                },
                'view_angles': multi_view_result.get('view_angles', []),
                'num_generated': multi_view_result.get('num_generated', 0)
            }
            
        except Exception as e:
            logging.error(f"Error in Mode 1 processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'mode': 'prompt_only'
            }
    
    def route_mode_2_image_prompt(
        self,
        input_image: Image.Image,
        prompt: str,
        num_views: int = 8,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Route Mode 2: Image + Prompt input.
        
        Args:
            input_image: Input image for transformation
            prompt: Style prompt for transformation
            num_views: Number of views to generate
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing generated content and metadata
        """
        try:
            logging.info(f"Processing Mode 2 (Image + Prompt): {prompt[:50]}...")
            
            # Step 1: Generate multi-view turntable with Zero123 + LPA
            multi_view_result = zero123_inference.generate_multi_view(
                input_image=input_image,
                prompt=prompt,
                num_views=num_views,
                **kwargs
            )
            
            # Step 2: Get style analysis
            style_analysis = clip_utils.get_style_confidence(prompt)
            top_style_matches = clip_utils.get_top_style_matches(prompt)
            
            return {
                'success': multi_view_result['success'],
                'mode': 'image_prompt',
                'original_image': input_image,
                'turntable_gif': multi_view_result.get('turntable_gif'),
                'multi_view_images': multi_view_result.get('images', []),
                'metadata': {
                    'multi_view_metadata': multi_view_result.get('metadata', []),
                    'prompt_analysis': prompt_parser.parse_prompt(prompt),
                    'style_analysis': style_analysis,
                    'top_style_matches': top_style_matches
                },
                'view_angles': multi_view_result.get('view_angles', []),
                'num_generated': multi_view_result.get('num_generated', 0)
            }
            
        except Exception as e:
            logging.error(f"Error in Mode 2 processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'mode': 'image_prompt'
            }
    
    def get_mode_info(self, mode: str) -> Dict[str, Any]:
        """
        Get information about a specific mode.
        
        Args:
            mode: Mode identifier ('prompt_only' or 'image_prompt')
            
        Returns:
            Dictionary containing mode information
        """
        mode_info = {
            'prompt_only': {
                'name': 'Prompt Only',
                'description': 'Generate 3D turntable from text prompt using SDXL + Zero123',
                'inputs': ['prompt'],
                'outputs': ['initial_image', 'turntable_gif', 'multi_view_images'],
                'models_used': ['SDXL', 'Zero123-XL', 'CLIP', 'LPA']
            },
            'image_prompt': {
                'name': 'Image + Prompt',
                'description': 'Transform uploaded image with style prompt using Zero123 + LPA',
                'inputs': ['input_image', 'prompt'],
                'outputs': ['turntable_gif', 'multi_view_images'],
                'models_used': ['Zero123-XL', 'CLIP', 'LPA']
            }
        }
        
        return mode_info.get(mode, {})
    
    def validate_inputs(self, mode: str, **inputs) -> Tuple[bool, str]:
        """
        Validate inputs for a specific mode.
        
        Args:
            mode: Mode identifier
            **inputs: Input parameters
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if mode == 'prompt_only':
            if 'prompt' not in inputs or not inputs['prompt'].strip():
                return False, "Prompt is required for Mode 1"
            return True, ""
        
        elif mode == 'image_prompt':
            if 'input_image' not in inputs:
                return False, "Input image is required for Mode 2"
            if 'prompt' not in inputs or not inputs['prompt'].strip():
                return False, "Style prompt is required for Mode 2"
            return True, ""
        
        else:
            return False, f"Unknown mode: {mode}"
    
    def get_processing_pipeline(self, mode: str) -> Dict[str, Any]:
        """
        Get the processing pipeline for a specific mode.
        
        Args:
            mode: Mode identifier
            
        Returns:
            Dictionary describing the processing pipeline
        """
        pipelines = {
            'prompt_only': {
                'steps': [
                    {
                        'name': 'SDXL Generation',
                        'description': 'Generate initial image from text prompt',
                        'model': 'SDXL',
                        'output': 'initial_image'
                    },
                    {
                        'name': 'Zero123 Multi-view',
                        'description': 'Generate multi-view turntable with LPA',
                        'model': 'Zero123-XL',
                        'output': 'turntable_gif'
                    }
                ]
            },
            'image_prompt': {
                'steps': [
                    {
                        'name': 'Zero123 Multi-view',
                        'description': 'Generate multi-view turntable with LPA',
                        'model': 'Zero123-XL',
                        'output': 'turntable_gif'
                    }
                ]
            }
        }
        
        return pipelines.get(mode, {})

# Global instance for easy access
mode_router = ModeRouter() 