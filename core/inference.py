import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from pathlib import Path

# Import our custom modules
from core.lpa_injector import lpa_injector
from core.prompt_utils import prompt_parser
from core.clip_utils import clip_utils

class Zero123Inference:
    """
    Shared logic for Zero123 inference with Local Prompt Adaptation (LPA).
    Handles multi-view synthesis from image input with style preservation.
    """
    
    def __init__(self, model_name: str = "ashawkey/zero123-xl"):
        """
        Initialize Zero123 inference engine.
        
        Args:
            model_name: Hugging Face model identifier for Zero123
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.processor = None
        
        try:
            # Import Zero123 model (this would need to be adapted based on actual model structure)
            self._load_model()
            logging.info(f"Zero123 model loaded successfully on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load Zero123 model: {e}")
            # For now, we'll create a placeholder structure
            self._create_placeholder_model()
    
    def _load_model(self):
        """Load the actual Zero123 model."""
        # This is a placeholder - actual implementation depends on model availability
        # For now, we'll create a mock structure
        self._create_placeholder_model()
    
    def _create_placeholder_model(self):
        """Create a placeholder model structure for development."""
        # This is a mock implementation for development purposes
        # In production, this would load the actual Zero123 model
        class MockZero123Model:
            def __init__(self):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            def eval(self):
                # Mock eval method
                pass
            
            def __call__(self, *args, **kwargs):
                # Mock inference - returns a placeholder image
                return self._mock_inference(*args, **kwargs)
            
            def _mock_inference(self, *args, **kwargs):
                # Create a placeholder image
                image = Image.new('RGB', (512, 512), color='gray')
                return {'images': [image]}
        
        self.model = MockZero123Model()
        self.processor = None  # Mock processor
    
    def setup_lpa_injection(self, prompt: str):
        """
        Setup LPA injection for the current prompt.
        
        Args:
            prompt: Text prompt for style guidance
        """
        # Parse prompt into object and style tokens
        parsed = prompt_parser.parse_prompt(prompt)
        object_tokens = parsed['object_tokens']
        style_tokens = parsed['style_tokens']
        
        # Setup LPA injection
        if self.model:
            lpa_injector.setup_injection(self.model, object_tokens, style_tokens)
        
        logging.info(f"LPA injection setup: {len(object_tokens)} object tokens, {len(style_tokens)} style tokens")
    
    def generate_multi_view(
        self,
        input_image: Image.Image,
        prompt: str,
        num_views: int = 8,
        elevation_range: Tuple[float, float] = (-30, 30),
        azimuth_range: Tuple[float, float] = (0, 360),
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate multi-view images from input image with style preservation.
        
        Args:
            input_image: Input image for multi-view synthesis
            prompt: Style prompt for LPA injection
            num_views: Number of views to generate
            elevation_range: Range of elevation angles (degrees)
            azimuth_range: Range of azimuth angles (degrees)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing generated views and metadata
        """
        try:
            # Setup LPA injection
            self.setup_lpa_injection(prompt)
            
            # Generate view angles
            view_angles = self._generate_view_angles(
                num_views, elevation_range, azimuth_range
            )
            
            # Generate images for each view
            generated_images = []
            metadata_list = []
            
            for i, (elevation, azimuth) in enumerate(view_angles):
                logging.info(f"Generating view {i+1}/{num_views}: elevation={elevation:.1f}°, azimuth={azimuth:.1f}°")
                
                result = self._generate_single_view(
                    input_image, elevation, azimuth, prompt, **kwargs
                )
                
                if result['success']:
                    generated_images.append(result['image'])
                    metadata_list.append(result['metadata'])
                else:
                    logging.warning(f"Failed to generate view {i+1}")
            
            # Create turntable animation
            turntable_gif = self._create_turntable_gif(generated_images)
            
            return {
                'images': generated_images,
                'metadata': metadata_list,
                'turntable_gif': turntable_gif,
                'view_angles': view_angles,
                'success': len(generated_images) > 0,
                'num_generated': len(generated_images)
            }
            
        except Exception as e:
            logging.error(f"Error in multi-view generation: {e}")
            return {
                'images': [],
                'metadata': [],
                'turntable_gif': None,
                'view_angles': [],
                'success': False,
                'error': str(e)
            }
        finally:
            # Clean up LPA hooks
            lpa_injector.clear_hooks()
    
    def _generate_view_angles(
        self,
        num_views: int,
        elevation_range: Tuple[float, float],
        azimuth_range: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        """Generate evenly distributed view angles."""
        elevation_min, elevation_max = elevation_range
        azimuth_min, azimuth_max = azimuth_range
        
        # Generate evenly spaced angles
        elevations = np.linspace(elevation_min, elevation_max, num_views)
        azimuths = np.linspace(azimuth_min, azimuth_max, num_views)
        
        return list(zip(elevations, azimuths))
    
    def _generate_single_view(
        self,
        input_image: Image.Image,
        elevation: float,
        azimuth: float,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a single view with the given angles.
        
        Args:
            input_image: Input image
            elevation: Elevation angle in degrees
            azimuth: Azimuth angle in degrees
            prompt: Style prompt
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing generated image and metadata
        """
        try:
            # Prepare input for model
            model_input = self._prepare_model_input(input_image, elevation, azimuth)
            
            # Generate image with LPA injection
            with torch.no_grad():
                result = self.model(model_input)
            
            # Extract generated image
            generated_image = result['images'][0] if 'images' in result else None
            
            # Prepare metadata
            metadata = {
                'elevation': elevation,
                'azimuth': azimuth,
                'prompt': prompt,
                'model_name': self.model_name
            }
            
            return {
                'image': generated_image,
                'metadata': metadata,
                'success': generated_image is not None
            }
            
        except Exception as e:
            logging.error(f"Error generating single view: {e}")
            return {
                'image': None,
                'metadata': {'error': str(e)},
                'success': False
            }
    
    def _prepare_model_input(self, image: Image.Image, elevation: float, azimuth: float) -> Dict[str, Any]:
        """Prepare input for the Zero123 model."""
        # This is a placeholder - actual implementation depends on model requirements
        return {
            'image': image,
            'elevation': elevation,
            'azimuth': azimuth
        }
    
    def _create_turntable_gif(self, images: List[Image.Image], duration: int = 100) -> Optional[Image.Image]:
        """
        Create a turntable GIF from generated images.
        
        Args:
            images: List of generated images
            duration: Duration per frame in milliseconds
            
        Returns:
            GIF image or None if creation fails
        """
        try:
            if not images:
                return None
            
            # Ensure all images have the same size
            target_size = images[0].size
            resized_images = [img.resize(target_size) for img in images]
            
            # Create GIF
            gif_path = "temp_turntable.gif"
            resized_images[0].save(
                gif_path,
                save_all=True,
                append_images=resized_images[1:],
                duration=duration,
                loop=0
            )
            
            # Load and return the GIF
            gif_image = Image.open(gif_path)
            
            # Clean up temporary file
            if os.path.exists(gif_path):
                os.remove(gif_path)
            
            return gif_image
            
        except Exception as e:
            logging.error(f"Error creating turntable GIF: {e}")
            return None
    
    def get_injection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current LPA injection setup."""
        return lpa_injector.get_injection_stats()
    
    def cleanup(self):
        """Clean up resources."""
        lpa_injector.clear_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Global instance for easy access
zero123_inference = Zero123Inference() 