import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any, List
import logging

class SDXLWrapper:
    """
    Wrapper around SDXL pipeline for Mode 1: Prompt-only text-to-image generation.
    """
    
    def __init__(self, model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        """
        Initialize SDXL pipeline.
        
        Args:
            model_name: Hugging Face model identifier for SDXL
        """
        # Use MPS for macOS, CUDA for Linux/Windows, CPU as fallback
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model_name = model_name
        
        try:
            # Load SDXL pipeline with optimizations
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.device.type != "cpu" else None,
                low_cpu_mem_usage=True
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Use DPM++ 2M scheduler for better quality
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Enable memory efficient attention if available
            try:
                if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                    self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logging.warning(f"Could not enable xformers memory efficient attention: {e}")
                logging.info("Continuing without xformers optimization")
            
            logging.info(f"SDXL pipeline loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load SDXL model {model_name}: {e}")
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image from text prompt using SDXL.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt to avoid certain elements
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            width: Image width
            height: Image height
            seed: Random seed for reproducibility
            **kwargs: Additional arguments for the pipeline
            
        Returns:
            Dictionary containing generated image and metadata
        """
        try:
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Generate image
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                **kwargs
            )
            
            # Extract image and metadata
            image = result.images[0] if result.images else None
            
            # Prepare metadata
            metadata = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'width': width,
                'height': height,
                'seed': seed,
                'model_name': self.model_name,
                'device': str(self.device)
            }
            
            return {
                'image': image,
                'metadata': metadata,
                'success': image is not None
            }
            
        except Exception as e:
            logging.error(f"Error generating image with SDXL: {e}")
            return {
                'image': None,
                'metadata': {'error': str(e)},
                'success': False
            }
    
    def generate_with_style_guidance(
        self,
        prompt: str,
        style_prompt: str,
        style_strength: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image with style guidance by combining object and style prompts.
        
        Args:
            prompt: Object description prompt
            style_prompt: Style description prompt
            style_strength: Strength of style influence (0-1)
            **kwargs: Additional arguments for generate_image
            
        Returns:
            Dictionary containing generated image and metadata
        """
        # Combine prompts with style guidance
        combined_prompt = f"{prompt}, {style_prompt}"
        
        # Adjust guidance scale based on style strength
        base_guidance = kwargs.get('guidance_scale', 7.5)
        adjusted_guidance = base_guidance * (1 + style_strength)
        
        return self.generate_image(
            prompt=combined_prompt,
            guidance_scale=adjusted_guidance,
            **kwargs
        )
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple images from a list of prompts.
        
        Args:
            prompts: List of text prompts
            **kwargs: Additional arguments for generate_image
            
        Returns:
            List of generation results
        """
        results = []
        for i, prompt in enumerate(prompts):
            logging.info(f"Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")
            result = self.generate_image(prompt=prompt, **kwargs)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'dtype': str(self.pipeline.unet.dtype),
            'parameters': sum(p.numel() for p in self.pipeline.unet.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.pipeline.unet.parameters() if p.requires_grad)
        }
    
    def optimize_for_inference(self):
        """Optimize the model for faster inference."""
        try:
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                self.pipeline.enable_xformers_memory_efficient_attention()
            
            # Enable model CPU offload for memory optimization
            if hasattr(self.pipeline, "enable_model_cpu_offload"):
                self.pipeline.enable_model_cpu_offload()
            
            # Enable sequential CPU offload
            if hasattr(self.pipeline, "enable_sequential_cpu_offload"):
                self.pipeline.enable_sequential_cpu_offload()
                
            # Set attention processor for better performance
            if hasattr(self.pipeline, "set_use_memory_efficient_attention_xformers"):
                self.pipeline.set_use_memory_efficient_attention_xformers(True)
                
        except Exception as e:
            logging.warning(f"Could not apply all optimizations: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'pipeline'):
            del self.pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Global instance for easy access
sdxl_wrapper = SDXLWrapper() 