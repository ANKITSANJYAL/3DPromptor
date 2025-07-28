#!/usr/bin/env python3
"""
PromptFusion3D - Main Application Entry Point
=============================================

A Gradio-based app that generates stylized, view-consistent 3D turntables
using Local Prompt Adaptation (LPA) to preserve spatial structure and style.

Author: AI Scientist
"""

import logging
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from app.interface import gradio_interface
from core.sdxl_wrapper import sdxl_wrapper
from core.inference import zero123_inference
from core.clip_utils import clip_utils
from core.prompt_utils import prompt_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('promptfusion3d.log')
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Setup the environment and check dependencies."""
    logger.info("Setting up PromptFusion3D environment...")
    
    # Check if spaCy model is available
    try:
        import spacy
        spacy.load("en_core_web_sm")
        logger.info("✓ spaCy English model loaded")
    except OSError:
        logger.error("✗ spaCy English model not found. Please run: python -m spacy download en_core_web_sm")
        return False
    
    # Check device availability
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("✓ MPS (Metal Performance Shaders) available for macOS")
        else:
            logger.warning("⚠ Using CPU - consider GPU for better performance")
    except ImportError:
        logger.error("✗ PyTorch not installed")
        return False
    
    # Create necessary directories
    dirs_to_create = [
        "assets/demo_inputs",
        "assets/demo_outputs", 
        "assets/lpa_diagrams",
        "examples"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("✓ Environment setup complete")
    return True

def initialize_models():
    """Initialize all required models."""
    logger.info("Initializing models...")
    
    try:
        # Initialize CLIP utilities
        logger.info("Loading CLIP model...")
        clip_utils.get_text_embedding("test")  # Test CLIP loading
        logger.info("✓ CLIP model loaded")
        
        # Initialize SDXL wrapper
        logger.info("Loading SDXL model...")
        sdxl_wrapper.optimize_for_inference()
        sdxl_info = sdxl_wrapper.get_model_info()
        logger.info(f"✓ SDXL model loaded: {sdxl_info['parameters']:,} parameters")
        
        # Initialize Zero123 inference
        logger.info("Loading Zero123 model...")
        zero123_info = zero123_inference.get_model_info()
        logger.info(f"✓ Zero123 model loaded: {zero123_info.get('model_name', 'Fallback')}")
        
        # Test prompt parser
        logger.info("Testing prompt parser...")
        test_result = prompt_parser.parse_prompt("cubist dragon statue")
        logger.info(f"✓ Prompt parser working: {len(test_result['object_tokens'])} object tokens, {len(test_result['style_tokens'])} style tokens")
        
        logger.info("✓ All models initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model initialization failed: {e}")
        return False

def create_demo_content():
    """Create demo content for the application."""
    logger.info("Creating demo content...")
    
    # Create sample prompts
    demo_prompts = [
        "cubist dragon statue",
        "vaporwave aesthetic building",
        "gothic cathedral architecture",
        "cyberpunk robot design",
        "art deco furniture piece"
    ]
    
    # Save demo prompts
    demo_file = Path("assets/demo_inputs/sample_prompts.txt")
    demo_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(demo_file, 'w') as f:
        f.write("Sample Prompts for PromptFusion3D\n")
        f.write("=" * 40 + "\n\n")
        for i, prompt in enumerate(demo_prompts, 1):
            f.write(f"{i}. {prompt}\n")
    
    logger.info(f"✓ Demo content created: {demo_file}")

def main():
    """Main application entry point."""
    logger.info("🚀 Starting PromptFusion3D...")
    
    # Setup environment
    if not setup_environment():
        logger.error("Environment setup failed. Exiting.")
        return 1
    
    # Initialize models
    if not initialize_models():
        logger.error("Model initialization failed. Exiting.")
        return 1
    
    # Create demo content
    create_demo_content()
    
    # Create Gradio interface
    logger.info("Creating Gradio interface...")
    interface = gradio_interface.create_interface()
    
    # Launch the app
    logger.info("Launching PromptFusion3D web interface...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        debug=True,
        show_error=True,
        inbrowser=True
    )
    
    return 0

def cleanup():
    """Cleanup resources on exit."""
    logger.info("Cleaning up resources...")
    
    try:
        # Cleanup model resources
        sdxl_wrapper.cleanup()
        zero123_inference.cleanup()
        
        # Clear CUDA cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✓ Cleanup complete")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Shutting down...")
        cleanup()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        cleanup()
        sys.exit(1) 