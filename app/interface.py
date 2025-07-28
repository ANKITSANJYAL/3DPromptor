import gradio as gr
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
                ### Generate interactive 3D objects with Local Prompt Adaptation (LPA)
                
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
                        visible=False,
                        height=200
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
                            info="More views = smoother 3D object"
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
                        "🚀 Generate 3D Object",
                        variant="primary",
                        size="lg"
                    )
                    
                    # Progress
                    progress = gr.Progress()
                
                with gr.Column(scale=2):
                    # Output Section
                    gr.Markdown("### 🎬 Output")
                    
                    # 3D Viewer
                    viewer_3d = gr.Plot(
                        label="3D Interactive Viewer",
                        container=True
                    )
                    
                    # Individual views gallery
                    views_gallery = gr.Gallery(
                        label="Generated Views",
                        columns=4,
                        rows=2,
                        height="auto"
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
                            "💾 Download 3D Model",
                            variant="secondary"
                        )
                        
                        download_path = gr.Textbox(
                            label="Download Path",
                            value="3d_model.glb",
                            visible=False
                        )
            
            # Event handlers
            mode_radio.change(
                fn=self._on_mode_change,
                inputs=[mode_radio],
                outputs=[image_input]
            )
            
            generate_btn.click(
                fn=self._generate_3d_object,
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
                    viewer_3d,
                    views_gallery,
                    style_analysis,
                    prompt_analysis,
                    metadata_output
                ],
                show_progress=True
            )
            
            download_btn.click(
                fn=self._download_3d_model,
                inputs=[viewer_3d],
                outputs=[download_path]
            )
        
        return interface
    
    def _on_mode_change(self, mode: str) -> Tuple[gr.Image]:
        """Handle mode change."""
        self.current_mode = "image_prompt" if mode == "Image + Prompt" else "prompt_only"
        
        # Show/hide image input based on mode
        image_visible = self.current_mode == "image_prompt"
        
        return (gr.Image(visible=image_visible, type="pil"),)
    
    def _generate_3d_object(
        self,
        mode: str,
        prompt: str,
        input_image: Optional[Image.Image],
        num_views: int,
        guidance_scale: float,
        seed: int,
        style_strength: float
    ) -> Tuple[go.Figure, List[Image.Image], Dict, Dict, Dict]:
        """
        Generate 3D object based on mode and inputs.
        
        Args:
            mode: Selected mode
            prompt: Style prompt
            input_image: Input image (for Mode 2)
            num_views: Number of views to generate
            guidance_scale: Guidance scale for generation
            seed: Random seed
            style_strength: Style application strength
            
        Returns:
            Tuple of (3d_plot, views_gallery, style_analysis, prompt_analysis, metadata)
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
            
            # Debug input validation
            logging.info(f"Mode: {actual_mode}, Prompt: {prompt}, Image: {input_image is not None}")
            
            # Validate inputs
            is_valid, error_msg = mode_router.validate_inputs(
                actual_mode,
                prompt=prompt,
                input_image=input_image
            )
            
            if not is_valid:
                return self._create_error_plot(error_msg), [], {}, {}, {'error': error_msg}
            
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
                return self._create_error_plot(result.get('error', 'Generation failed')), [], {}, {}, {'error': result.get('error', 'Generation failed')}
            
            # Prepare outputs
            images = result.get('multi_view_images', [])
            
            # Create 3D viewer
            plot_3d = self._create_3d_viewer(images, result.get('view_angles', []))
            
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
            
            return plot_3d, images, style_analysis, prompt_analysis, metadata
            
        except Exception as e:
            logging.error(f"Error in 3D object generation: {e}")
            return self._create_error_plot(str(e)), [], {}, {}, {'error': str(e)}
    
    def _create_3d_viewer(self, images: List[Image.Image], view_angles: List[Tuple[float, float]]) -> go.Figure:
        """Create an interactive 3D viewer from multiple images."""
        if not images:
            return self._create_error_plot("No images generated")
        
        try:
            # Create a 3D scatter plot with images as points
            fig = go.Figure()
            
            # Convert images to base64 for display
            import base64
            from io import BytesIO
            
            for i, (image, (elevation, azimuth)) in enumerate(zip(images, view_angles)):
                # Convert image to base64
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Create 3D point with image
                fig.add_trace(go.Scatter3d(
                    x=[np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation))],
                    y=[np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation))],
                    z=[np.sin(np.radians(elevation))],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        opacity=0.8
                    ),
                    text=[f"View {i+1}: {elevation:.1f}°, {azimuth:.1f}°"],
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    name=f"View {i+1}"
                ))
            
            # Update layout for better 3D viewing
            fig.update_layout(
                title="3D Object Viewer - Click and drag to rotate",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y", 
                    zaxis_title="Z",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                width=600,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating 3D viewer: {e}")
            return self._create_error_plot(f"Error creating 3D viewer: {str(e)}")
    
    def _create_error_plot(self, error_msg: str) -> go.Figure:
        """Create an error plot."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {error_msg}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Generation Error",
            width=600,
            height=500
        )
        return fig
    
    def _download_3d_model(self, plot_3d: go.Figure) -> str:
        """Handle 3D model download."""
        if plot_3d is None:
            return "No 3D model to download"
        
        try:
            # Save 3D model (for now, save as HTML)
            output_dir = Path("assets/demo_outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"3d_model_{int(np.random.randint(10000, 99999))}.html"
            filepath = output_dir / filename
            
            plot_3d.write_html(str(filepath))
            
            return f"Saved to: {filepath}"
            
        except Exception as e:
            logging.error(f"Error saving 3D model: {e}")
            return f"Error saving 3D model: {str(e)}"
    
    def get_interface_info(self) -> Dict[str, Any]:
        """Get information about the interface."""
        return {
            'title': 'PromptFusion3D',
            'description': 'Generate interactive 3D objects with LPA',
            'modes': ['prompt_only', 'image_prompt'],
            'features': [
                'Mode switching',
                'Real-time generation',
                'Interactive 3D viewer',
                'Style analysis',
                'Prompt analysis',
                '3D model download',
                'Progress tracking'
            ]
        }

# Global instance for easy access
gradio_interface = GradioInterface() 