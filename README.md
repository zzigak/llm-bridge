# LLMBridge

A unified interface that bridges different LLM providers (OpenAI, HuggingFace, OpenRouter, etc.) with a common API, featuring cost tracking and caching capabilities.

## Features

- **Unified API**: Use the same interface for different LLM providers
- **Cost Tracking**: Monitor your API usage and costs in real-time
- **Caching**: Save money and improve response times with disk-based caching
- **Easy Model Management**: Add new models with simple configuration updates
- **OpenRouter Support**: Full integration with OpenRouter's model marketplace

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd llm-bridge

# Install the package in editable mode
uv pip install -e LLMBridge/

# Activate the virtual environment
source .venv/bin/activate
```

## Quick Start with OpenRouter

### 1. Set up your API key

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

### 2. Basic Usage

```python
from llmbridge import create_llm, choose_provider

# Choose OpenRouter as your provider
choose_provider('openrouter')

# Create a model instance
model = create_llm('google/gemini-2.5-pro-preview')

# ⚠️ IMPORTANT: Always pass prompts as a LIST, not a string!
prompt = "Hello, how are you today?"
response = model.prompt([prompt], max_tokens=500, temperature=0.7)[0]
print(response)
```

## ⚠️ Critical: Prompt Format

**ALWAYS pass prompts as a list, not a string!**

```python
# ❌ WRONG - This will break your string into characters!
response = model.prompt("Hello world", max_tokens=100)

# ✅ CORRECT - Pass as a list
response = model.prompt(["Hello world"], max_tokens=100)[0]

# ✅ CORRECT - Multiple prompts
responses = model.prompt(["Hello", "How are you?"], max_tokens=100)
```

## Cost Tracking

LLMBridge provides detailed cost tracking for all your API calls:

```python
from llmbridge import create_llm, choose_provider

choose_provider('openrouter')
model = create_llm('google/gemini-2.5-pro-preview')

# Make some API calls
response1 = model.prompt(["What is 2+2?"], max_tokens=50)[0]
response2 = model.prompt(["Explain quantum physics"], max_tokens=200)[0]

# Get cost information
info = model.get_info()
print(f"Total calls: {info['calls']}")
print(f"Input tokens: {info['input_tokens']}")
print(f"Output tokens: {info['output_tokens']}")
print(f"Total cost: ${info['actual_cost']:.6f}")
```

### Cost Information Available

- `calls`: Number of API calls made
- `input_tokens`: Total input tokens used
- `output_tokens`: Total output tokens generated
- `actual_cost`: Real cost in USD
- `cost`: Interface cost (may have bugs, use `actual_cost`)

## Caching

Save money and improve response times with disk-based caching:

```python
from llmbridge import create_llm, choose_provider

choose_provider('openrouter')
model = create_llm('google/gemini-2.5-pro-preview')

# Set up disk caching
model.setup_cache('disk', database_path='my_cache.db')

# First call - will be slow and cache the result
response1 = model.prompt(["What is the capital of France?"], max_tokens=100)[0]

# Second call with same parameters - will be fast and use cache
response2 = model.prompt(["What is the capital of France?"], max_tokens=100)[0]

# Different parameters - will not use cache
response3 = model.prompt(["What is the capital of France?"], max_tokens=200)[0]
```

### Cache Behavior

- **Same prompt + same parameters**: Uses cache (fast, no cost)
- **Same prompt + different parameters**: New API call (slow, costs money)
- **Different prompt**: New API call (slow, costs money)

## Adding New Models

To add new models to the cost tracking system:

### 1. Update the model list in `LLMBridge/llmbridge/utils.py`

```python
# Find this section in utils.py
model_list = [
    'gpt-3.5',
    'gpt-4',
    'o1',
    'claude',
    'gemini',
    'llama',
    'your-new-model'  # Add your new model here
]
```

### 2. Update cost information in `LLMBridge/llmbridge/openai.py`

Find the `get_cost` function and add pricing for your model:

```python
def get_cost(self, model_name, input_tokens, output_tokens):
    # Add your model's pricing here
    if 'your-new-model' in model_name:
        input_cost_per_1k = 0.001  # $0.001 per 1k input tokens
        output_cost_per_1k = 0.002  # $0.002 per 1k output tokens
    elif 'gpt-4' in model_name:
        input_cost_per_1k = 0.03
        output_cost_per_1k = 0.06
    # ... other models
    
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    return input_cost + output_cost
```

### 3. Reinstall the package

```bash
# Uninstall the old version
uv pip uninstall llmbridge

# Reinstall with your changes
uv pip install -e LLMBridge/
```

## Complete Example

```python
import os
import time
from llmbridge import create_llm, choose_provider

# Set up API key
api_key = os.environ.get('OPENROUTER_API_KEY')
if not api_key:
    raise ValueError("Please set the OPENROUTER_API_KEY environment variable")

# Configure the interface
choose_provider('openrouter')
model = create_llm('google/gemini-2.5-pro-preview')

# Set up caching
model.setup_cache('disk', database_path='cache.db')

# Test caching with proper list format
prompt = "Explain the benefits of caching in AI applications."

print("1️⃣ First call (will cache):")
start_time = time.time()
response1 = model.prompt([prompt], max_tokens=500, temperature=0.5)[0]
time1 = time.time() - start_time
print(f"   Time: {time1:.3f} seconds")

print("\n2️⃣ Second call (should use cache):")
start_time = time.time()
response2 = model.prompt([prompt], max_tokens=500, temperature=0.5)[0]
time2 = time.time() - start_time
print(f"   Time: {time2:.3f} seconds")

# Get final stats
info = model.get_info()
print(f"\n📊 Final Stats:")
print(f"   Total calls: {info['calls']}")
print(f"   Total cost: ${info['actual_cost']:.6f}")
```

## Supported Models

LLMBridge supports all models available through OpenRouter, including:

- **Google**: gemini-2.5-pro-preview, gemini-2.0-flash-exp
- **OpenAI**: gpt-4, gpt-3.5-turbo, gpt-4o
- **Anthropic**: claude-3-opus, claude-3-sonnet, claude-3-haiku
- **Meta**: llama-3.1-8b-instruct, llama-3.1-70b-instruct
- **And many more** available through OpenRouter's marketplace

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Make sure you've activated the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. **API Key Error**: Ensure your OpenRouter API key is set:
   ```bash
   export OPENROUTER_API_KEY="your_key_here"
   ```

3. **String instead of list error**: Always pass prompts as lists:
   ```python
   # ❌ Wrong
   model.prompt("Hello", max_tokens=100)
   
   # ✅ Correct
   model.prompt(["Hello"], max_tokens=100)[0]
   ```

4. **Cost tracking not working**: Check that your model is in the `model_list` in `utils.py`

## Contributing

To add new features or fix bugs:

1. Make your changes in the `LLMBridge/` directory
2. Reinstall the package: `uv pip install -e LLMBridge/`
3. Test your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
