import gradio as gr
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import os
from pathlib import Path

# Import our modules
from app.router import mode_router
from core.clip_utils import clip_utils
from core.prompt_utils import prompt_parser

class GradioInterface:
    """
    Gradio interface logic for PromptFusion3D.
    Handles mode switching, inputs, outputs, and UI interactions.
    """
    
    def __init__(self):
        """Initialize the Gradio interface."""
        self.current_mode = "prompt_only"
        self.processing = False
        
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        
        with gr.Blocks(
            title="PromptFusion3D",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .output-image {
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            """
        ) as interface:
            
            # Header
            gr.Markdown(
                """
                # 🎨 PromptFusion3D
                ### Generate stylized, view-consistent 3D turntables with Local Prompt Adaptation (LPA)
                
                Choose your mode and start creating amazing 3D content!
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Mode Selection
                    gr.Markdown("### 🎯 Mode Selection")
                    mode_radio = gr.Radio(
                        choices=["Prompt Only", "Image + Prompt"],
                        value="Prompt Only",
                        label="Select Mode",
                        info="Choose how you want to generate your 3D content"
                    )
                    
                    # Input Section
                    gr.Markdown("### 📝 Input")
                    
                    # Prompt input
                    prompt_input = gr.Textbox(
                        label="Style Prompt",
                        placeholder="e.g., 'cubist style', 'vaporwave aesthetic', 'gothic architecture'",
                        lines=3,
                        max_lines=5
                    )
                    
                    # Image upload (for Mode 2)
                    image_input = gr.Image(
                        label="Upload Image (Mode 2 only)",
                        type="pil",
                        visible=False
                    )
                    
                    # Generation Parameters
                    gr.Markdown("### ⚙️ Parameters")
                    
                    with gr.Row():
                        num_views = gr.Slider(
                            minimum=4,
                            maximum=16,
                            value=8,
                            step=1,
                            label="Number of Views",
                            info="More views = smoother turntable"
                        )
                        
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5,
                            label="Guidance Scale",
                            info="Higher = more prompt adherence"
                        )
                    
                    with gr.Row():
                        seed_input = gr.Number(
                            label="Seed",
                            value=-1,
                            info="Set to -1 for random, or specific number for reproducibility"
                        )
                        
                        style_strength = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Style Strength",
                            info="How strongly to apply the style"
                        )
                    
                    # Generate Button
                    generate_btn = gr.Button(
                        "🚀 Generate 3D Turntable",
                        variant="primary",
                        size="lg"
                    )
                    
                    # Progress
                    progress = gr.Progress()
                
                with gr.Column(scale=2):
                    # Output Section
                    gr.Markdown("### 🎬 Output")
                    
                    # Main output
                    output_gif = gr.Image(
                        label="3D Turntable GIF",
                        type="pil",
                        format="gif"
                    )
                    
                    # Additional outputs
                    with gr.Accordion("📊 Analysis & Details", open=False):
                        # Style analysis
                        style_analysis = gr.JSON(
                            label="Style Analysis",
                            visible=True
                        )
                        
                        # Prompt analysis
                        prompt_analysis = gr.JSON(
                            label="Prompt Analysis",
                            visible=True
                        )
                        
                        # Generation metadata
                        metadata_output = gr.JSON(
                            label="Generation Metadata",
                            visible=True
                        )
                    
                    # Download section
                    with gr.Row():
                        download_btn = gr.Button(
                            "💾 Download GIF",
                            variant="secondary"
                        )
                        
                        download_path = gr.Textbox(
                            label="Download Path",
                            value="turntable.gif",
                            visible=False
                        )
            
            # Event handlers
            mode_radio.change(
                fn=self._on_mode_change,
                inputs=[mode_radio],
                outputs=[image_input]
            )
            
            generate_btn.click(
                fn=self._generate_turntable,
                inputs=[
                    mode_radio,
                    prompt_input,
                    image_input,
                    num_views,
                    guidance_scale,
                    seed_input,
                    style_strength
                ],
                outputs=[
                    output_gif,
                    style_analysis,
                    prompt_analysis,
                    metadata_output
                ],
                show_progress=True
            )
            
            download_btn.click(
                fn=self._download_gif,
                inputs=[output_gif],
                outputs=[download_path]
            )
        
        return interface
    
    def _on_mode_change(self, mode: str) -> Tuple[gr.Image]:
        """Handle mode change."""
        self.current_mode = "image_prompt" if mode == "Image + Prompt" else "prompt_only"
        
        # Show/hide image input based on mode
        image_visible = self.current_mode == "image_prompt"
        
        return (gr.Image(visible=image_visible),)
    
    def _generate_turntable(
        self,
        mode: str,
        prompt: str,
        input_image: Optional[Image.Image],
        num_views: int,
        guidance_scale: float,
        seed: int,
        style_strength: float
    ) -> Tuple[Optional[Image.Image], Dict, Dict, Dict]:
        """
        Generate 3D turntable based on mode and inputs.
        
        Args:
            mode: Selected mode
            prompt: Style prompt
            input_image: Input image (for Mode 2)
            num_views: Number of views to generate
            guidance_scale: Guidance scale for generation
            seed: Random seed
            style_strength: Style application strength
            
        Returns:
            Tuple of (gif_image, style_analysis, prompt_analysis, metadata)
        """
        try:
            # Set seed
            if seed != -1:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Determine actual mode
            actual_mode = "image_prompt" if mode == "Image + Prompt" else "prompt_only"
            
            # Validate inputs
            is_valid, error_msg = mode_router.validate_inputs(
                actual_mode,
                prompt=prompt,
                input_image=input_image
            )
            
            if not is_valid:
                return None, {}, {}, {'error': error_msg}
            
            # Route to appropriate processing
            if actual_mode == "prompt_only":
                result = mode_router.route_mode_1_prompt_only(
                    prompt=prompt,
                    num_views=num_views,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    style_strength=style_strength
                )
            else:
                result = mode_router.route_mode_2_image_prompt(
                    input_image=input_image,
                    prompt=prompt,
                    num_views=num_views,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    style_strength=style_strength
                )
            
            if not result['success']:
                return None, {}, {}, {'error': result.get('error', 'Generation failed')}
            
            # Prepare outputs
            gif_image = result.get('turntable_gif')
            
            # Style analysis
            style_analysis = {}
            if 'metadata' in result and 'style_analysis' in result['metadata']:
                style_analysis = result['metadata']['style_analysis']
            
            # Prompt analysis
            prompt_analysis = {}
            if 'metadata' in result and 'prompt_analysis' in result['metadata']:
                prompt_analysis = result['metadata']['prompt_analysis']
            
            # Metadata
            metadata = {
                'mode': actual_mode,
                'num_views': num_views,
                'guidance_scale': guidance_scale,
                'seed': seed,
                'style_strength': style_strength,
                'num_generated': result.get('num_generated', 0),
                'view_angles': result.get('view_angles', [])
            }
            
            return gif_image, style_analysis, prompt_analysis, metadata
            
        except Exception as e:
            logging.error(f"Error in turntable generation: {e}")
            return None, {}, {}, {'error': str(e)}
    
    def _download_gif(self, gif_image: Optional[Image.Image]) -> str:
        """Handle GIF download."""
        if gif_image is None:
            return "No GIF to download"
        
        try:
            # Save GIF to assets/demo_outputs
            output_dir = Path("assets/demo_outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"turntable_{int(np.random.randint(10000, 99999))}.gif"
            filepath = output_dir / filename
            
            gif_image.save(filepath, save_all=True, loop=0)
            
            return f"Saved to: {filepath}"
            
        except Exception as e:
            logging.error(f"Error saving GIF: {e}")
            return f"Error saving GIF: {str(e)}"
    
    def get_interface_info(self) -> Dict[str, Any]:
        """Get information about the interface."""
        return {
            'title': 'PromptFusion3D',
            'description': 'Generate stylized 3D turntables with LPA',
            'modes': ['prompt_only', 'image_prompt'],
            'features': [
                'Mode switching',
                'Real-time generation',
                'Style analysis',
                'Prompt analysis',
                'GIF download',
                'Progress tracking'
            ]
        }

# Global instance for easy access
gradio_interface = GradioInterface() 