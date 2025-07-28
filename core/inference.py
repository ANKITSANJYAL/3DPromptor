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
        # Use MPS for macOS, CUDA for Linux/Windows, CPU as fallback
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
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
        # For development, use fallback model for faster startup
        # Set this to False to load the real Zero123 model
        USE_DEVELOPMENT_MODE = True
        
        if USE_DEVELOPMENT_MODE:
            logging.info("Using development mode with fallback model for faster startup")
            self._create_placeholder_model()
            return
            
        try:
            # Import Zero123 from diffusers
            from diffusers import DiffusionPipeline
            from diffusers.utils import load_image
            import torch
            
            # Load Zero123-XL model
            self.model = DiffusionPipeline.from_pretrained(
                "ashawkey/zero123-xl",
                torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
                use_safetensors=True
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Enable memory efficient attention if available
            try:
                if hasattr(self.model, "enable_xformers_memory_efficient_attention"):
                    self.model.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logging.warning(f"Could not enable xformers: {e}")
            
            # Set to evaluation mode
            self.model.eval()
            
            logging.info(f"Zero123-XL model loaded successfully on {self.device}")
            
        except Exception as e:
            logging.error(f"Failed to load Zero123 model: {e}")
            logging.info("Falling back to mock model for development")
            self._create_placeholder_model()
    
    def _create_placeholder_model(self):
        """Create a fallback model for development."""
        class FallbackZero123Model:
            def __init__(self):
                # Use MPS for macOS, CUDA for Linux/Windows, CPU as fallback
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    self.device = torch.device("cpu")
            
            def eval(self):
                pass
            
            def to(self, device):
                self.device = device
                return self
            
            def __call__(self, *args, **kwargs):
                return self._generate_fallback_image(*args, **kwargs)
            
            def _generate_fallback_image(self, *args, **kwargs):
                """Generate a more realistic fallback image."""
                import numpy as np
                from PIL import Image, ImageDraw
                import time
                
                # Create a more interesting placeholder image
                size = (512, 512)
                image = Image.new('RGB', size, color='#2C3E50')
                draw = ImageDraw.Draw(image)
                
                # Use current time to create variation
                seed = int(time.time() * 1000) % 1000
                np.random.seed(seed)
                
                # Draw some geometric shapes to simulate 3D object
                # Center rectangle with rotation effect
                center_x, center_y = size[0] // 2, size[1] // 2
                rect_size = 100 + (seed % 50)
                offset_x = (seed % 20) - 10
                offset_y = ((seed + 100) % 20) - 10
                
                draw.rectangle(
                    [center_x - rect_size//2 + offset_x, center_y - rect_size//2 + offset_y,
                     center_x + rect_size//2 + offset_x, center_y + rect_size//2 + offset_y],
                    fill='#E74C3C', outline='#C0392B', width=3
                )
                
                # Add some decorative elements with variation
                for i in range(3 + (seed % 3)):
                    x = np.random.randint(50, size[0] - 50)
                    y = np.random.randint(50, size[1] - 50)
                    radius = np.random.randint(10, 30)
                    color = f'#{np.random.randint(0, 0xFFFFFF):06x}'
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
                
                # Add text with view information
                draw.text((10, 10), f"Zero123-XL View {seed}", fill='white', font=None)
                draw.text((10, size[1] - 30), f"Mock Output {seed}", fill='white', font=None)
                
                return {'images': [image]}
        
        self.model = FallbackZero123Model()
        self.processor = None
    
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
        
        # Setup LPA injection (only for real models)
        if self.model and hasattr(self.model, 'named_modules'):
            lpa_injector.setup_injection(self.model, object_tokens, style_tokens)
        else:
            logging.info("Skipping LPA injection for fallback model")
        
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
            
            logging.info(f"Generated {len(generated_images)} images, GIF created: {turntable_gif is not None}")
            
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'model_type': type(self.model).__name__,
            'is_fallback': hasattr(self.model, '_generate_fallback_image')
        }
    
    def cleanup(self):
        """Clean up resources."""
        lpa_injector.clear_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Clear MPS cache if available
            try:
                torch.mps.empty_cache()
            except:
                pass

# Global instance for easy access
zero123_inference = Zero123Inference() 