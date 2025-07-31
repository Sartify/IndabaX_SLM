# PAWA-MIN-ALPHA Language Model

A lightweight language model implementation with ChatML template support and GPU memory monitoring.

## 🚀 Quick Start

### Installation

```bash
pip install --upgrade "transformers>=4.52.0" "accelerate" torch
```

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "sartifyllc/pawa-min-alpha"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
```

## 📋 Features

- **ChatML Template Support**: Properly formatted conversation handling
- **GPU Memory Monitoring**: Track memory usage during inference
- **Multi-language Support**: Demonstrated with English-Swahili translation
- **Efficient Memory Usage**: Uses bfloat16 precision for optimal performance
- **Automatic Device Placement**: Handles GPU/CPU allocation automatically

## 🔧 Configuration

### Model Parameters
- **Model**: `sartifyllc/pawa-min-alpha`
- **Max Sequence Length**: 2048 tokens
- **Precision**: bfloat16
- **Temperature**: 0.2 (adjustable)
- **Top-p**: 0.9 (nucleus sampling)

### Generation Parameters
```python
generation_config = {
    "max_new_tokens": 128,
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.9,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "use_cache": True
}
```

## 💬 Chat Template Format

The model uses ChatML format for conversations:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"}
]
```

Supported roles:
- `system` - System instructions
- `user` - User messages  
- `assistant` - Model responses

## 📊 Memory Monitoring

The implementation includes comprehensive GPU memory tracking:

```python
# Before inference
gpu_stats = torch.cuda.get_device_properties(0)
start_reserved = torch.cuda.max_memory_reserved() / 1024**3

# After inference  
end_reserved = torch.cuda.max_memory_reserved() / 1024**3
delta_reserved = end_reserved - start_reserved
```

## 🌍 Example: Translation Task

```python
def translate_swahili_to_english(swahili_text):
    messages = [
        {
            "role": "user", 
            "content": f"Translate to English from swahili: '{swahili_text}'"
        }
    ]
    
    response = generate_response(messages, max_new_tokens=128)
    return response

# Example usage
swahili_text = "Baba yangu ni shujaa na anaishi Marekani"
translation = translate_swahili_to_english(swahili_text)
print(f"Original: {swahili_text}")
print(f"Translation: {translation}")
```

## 🛠️ Functions

### `apply_chatml_template(messages)`
Converts conversation messages into ChatML format.

**Parameters:**
- `messages` (list): List of message dictionaries with `role` and `content`

**Returns:**
- `str`: Formatted ChatML template string

### `generate_response(messages, max_new_tokens=64)`
Generates model response for given conversation.

**Parameters:**
- `messages` (list|str): Conversation messages or single string
- `max_new_tokens` (int): Maximum tokens to generate

**Returns:**
- `list`: Generated response(s)

## 📈 Performance Monitoring

The script provides detailed memory usage statistics:

```
🖥️ GPU: NVIDIA GeForce RTX 4090
📊 Max memory: 24.0 GB
🔹 Reserved before Inference: 2.1 GB
📈 Peak reserved memory after Inference: 3.8 GB
📉 Additional memory used for Inference: 1.7 GB
💯 Total memory used (%): 15.8 %
🧠 Inference memory usage (%): 7.1 %
```

## ⚠️ Requirements

- Python 3.8+
- PyTorch with CUDA support
- Transformers >= 4.52.0
- Accelerate library
- CUDA-compatible GPU (recommended)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Model Information

- **Model Hub**: [sartifyllc/pawa-min-alpha](https://huggingface.co/sartifyllc/pawa-min-alpha)
- **Architecture**: Causal Language Model
- **Use Case**: Multi-language text generation and translation

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `max_seq_length` or use CPU inference
2. **Model Loading Errors**: Ensure `trust_remote_code=True` is set
3. **Tokenizer Issues**: Verify pad_token is properly configured

### Memory Optimization Tips

- Use `torch.bfloat16` or `torch.float16` for reduced memory usage
- Enable `use_cache=True` for faster inference
- Consider gradient checkpointing for training scenarios

---

**Note**: This model is designed for research and educational purposes. Please review the model's capabilities and limitations before production use.
