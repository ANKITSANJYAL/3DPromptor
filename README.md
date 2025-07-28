# 🎨 PromptFusion3D

Generate stylized, view-consistent 3D turntables with **Local Prompt Adaptation (LPA)** to preserve spatial structure and style across different viewing angles.

## 🚀 Features

- **Two Operating Modes:**
  - 🔵 **Mode 1: Prompt-only input** - Generate 3D objects from text descriptions
  - 🟢 **Mode 2: Image + Prompt** - Upload an image and apply style transformations

- **Advanced Style Control:**
  - Local Prompt Adaptation (LPA) for consistent style across views
  - CLIP-based style analysis and scoring
  - Intelligent prompt parsing (object vs. style tokens)

- **High-Quality Output:**
  - Smooth 3D turntable GIFs
  - Interactive viewer options
  - Attention map visualization
  - Download functionality

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio UI     │    │   Mode Router   │    │   Core Models   │
│                 │    │                 │    │                 │
│ • Mode Selector │───▶│ • Input Validation│───▶│ • SDXL         │
│ • Parameter UI  │    │ • Pipeline Route│    │ • Zero123-XL    │
│ • Output Display│    │ • Error Handling│    │ • CLIP          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   LPA Injector  │
                       │                 │
                       │ • Token Parsing │
                       │ • Attention Hook│
                       │ • Style Injection│
                       └─────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/promptfusion3d.git
   cd promptfusion3d
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Verify installation:**
   ```bash
   python -c "from core.prompt_utils import prompt_parser; print('✓ Setup complete')"
   ```

## 🎯 Usage

### Quick Start

1. **Launch the application:**
   ```bash
   python app/main.py
   ```

2. **Open your browser** to `http://localhost:7860`

3. **Choose your mode:**
   - **Prompt Only**: Enter a description like "cubist dragon statue"
   - **Image + Prompt**: Upload an image and add style like "vaporwave aesthetic"

4. **Adjust parameters** and click "Generate 3D Turntable"

### Mode 1: Prompt Only

Generate 3D objects directly from text descriptions:

```
Input: "A vaporwave cyberpunk building with neon lights"
Output: 3D turntable showing the building from multiple angles
```

**Pipeline:**
1. SDXL generates initial image from prompt
2. Zero123-XL creates multi-view with LPA injection
3. Stitched into smooth turntable GIF

### Mode 2: Image + Prompt

Transform uploaded images with style prompts:

```
Input: [Upload building image] + "gothic architecture style"
Output: 3D turntable of building in gothic style
```

**Pipeline:**
1. Parse prompt into object vs. style tokens
2. Zero123-XL generates multi-view with LPA
3. Style tokens injected in late layers, object tokens in early layers

## 🔧 Configuration

### Model Parameters

```python
# In app/interface.py
generation_params = {
    'num_views': 8,           # Number of views (4-16)
    'guidance_scale': 7.5,    # Prompt adherence (1.0-20.0)
    'style_strength': 0.7,    # Style application (0.0-1.0)
    'seed': -1               # Random seed (-1 for random)
}
```

### LPA Configuration

```python
# In core/lpa_injector.py
lpa_config = {
    'early_injection_strength': 0.3,  # Object tokens
    'late_injection_strength': 0.5,   # Style tokens
    'attention_layers': 'cross_attn'   # Injection target
}
```

## 📁 Project Structure

```
promptfusion3d/
├── app/
│   ├── main.py            # Gradio app entry point
│   ├── interface.py       # UI logic (mode switch, inputs, outputs)
│   └── router.py          # Mode routing logic
├── core/
│   ├── lpa_injector.py    # Injects prompt tokens into UNet attention
│   ├── inference.py       # Shared logic for Zero123 inference
│   ├── sdxl_wrapper.py    # For Mode 1: SDXL text-to-image
│   ├── clip_utils.py      # CLIP token embedding + style scoring
│   └── prompt_utils.py    # spaCy-based object/style token splitter
├── assets/
│   ├── demo_inputs/       # Sample prompts + images
│   ├── demo_outputs/      # Stylized turntables
│   └── lpa_diagrams/      # For attention viz (optional)
├── examples/
│   └── test_notebooks.ipynb
├── README.md
├── requirements.txt
└── LICENSE
```

## 🧠 Technical Details

### Local Prompt Adaptation (LPA)

LPA injects prompt tokens into specific attention layers:

- **Early Layers (1-8)**: Object tokens for spatial structure
- **Late Layers (9-16)**: Style tokens for aesthetic control

```python
# Token injection example
object_tokens = ["dragon", "statue"]      # Early injection
style_tokens = ["cubist", "style"]        # Late injection

# Attention modification
modified_attention = (1 - strength) * original + strength * token_embedding
```

### Style Analysis

CLIP-based style scoring provides confidence metrics:

```python
style_confidence = {
    'artistic': 0.85,      # Cubist, impressionist, etc.
    'modern': 0.72,        # Modern, minimalist, etc.
    'colorful': 0.63,      # Vibrant, bright, etc.
    'textured': 0.41       # Metallic, wooden, etc.
}
```

### Prompt Parsing

Intelligent separation of object vs. style tokens:

```python
prompt = "cubist dragon statue"
parsed = {
    'object_tokens': ['dragon', 'statue'],
    'style_tokens': ['cubist'],
    'original_prompt': 'cubist dragon statue'
}
```

## 🎨 Style Examples

### Available Styles

| Category | Examples |
|----------|----------|
| **Artistic** | cubist, impressionist, abstract, surreal, pop art |
| **Modern** | modern, minimalist, futuristic, cyberpunk |
| **Classical** | baroque, gothic, renaissance, classical |
| **Colorful** | colorful, vibrant, bright, shimmering |
| **Monochrome** | monochrome, black and white, sepia |
| **Textured** | metallic, wooden, crystal, rough, smooth |

### Prompt Examples

```
✅ Good prompts:
- "cubist dragon statue"
- "vaporwave aesthetic building"
- "gothic cathedral architecture"
- "cyberpunk robot design"
- "art deco furniture piece"

❌ Avoid:
- "make it look good" (too vague)
- "very very very detailed" (redundant)
- "trending on artstation" (not style-specific)
```

## 🔍 Troubleshooting

### Common Issues

1. **"spaCy model not found"**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **CUDA out of memory**
   - Reduce `num_views` parameter
   - Use CPU mode (slower but works)
   - Close other GPU applications

3. **Generation fails**
   - Check prompt length (keep under 200 chars)
   - Try different seeds
   - Adjust guidance scale

4. **Slow generation**
   - Enable xformers: `pip install xformers`
   - Use smaller image sizes
   - Reduce number of views

### Performance Tips

- **GPU Memory**: 8GB+ recommended for smooth operation
- **Batch Processing**: Generate multiple turntables in sequence
- **Caching**: Models are cached after first load
- **Optimization**: Enable memory efficient attention

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest tests/

# Format code
black .

# Lint code
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stability AI** for SDXL
- **Zero123** team for multi-view synthesis
- **OpenAI** for CLIP
- **Gradio** for the web interface
- **spaCy** for NLP processing

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/promptfusion3d/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/promptfusion3d/discussions)
- **Email**: your.email@example.com

---

**Made with ❤️ by AI Scientists**

*Generate amazing 3D content with style and precision!*