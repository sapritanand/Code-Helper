# Code-Helper 🤖

A complete guide to fine-tuning the TinyLlama language model using LoRA (Low-Rank Adaptation) for code-related tasks. This project demonstrates efficient model training on GPUs with quantization techniques to minimize memory usage while maintaining quality results.

## Project Purpose
The Code-Helper repository is designed to assist developers by providing tools and libraries that simplify coding tasks, enhance productivity, and facilitate learning. Whether you are debugging, learning a new programming language, or searching for coding best practices, Code-Helper has something for everyone.

## Features
- **Code Snippets**: A collection of reusable code snippets for various programming languages.
- **Documentation**: Comprehensive guides and tutorials for beginners and experienced developers.
- **Integration**: Easy integration with popular code editors and IDEs.
- **Examples**: Practical usage examples to help you leverage the tools effectively.

## Overview
Install dependencies by running `pip install transformers datasets peft bitsandbytes accelerate trl torch`.  
Set environment variable to prevent tokenizer warnings: `export TOKENIZERS_PARALLELISM=false`.  
Verify GPU setup by running Python code:  
`import torch`  
`print(f"CUDA available: {torch.cuda.is_available()}")`  
`if torch.cuda.is_available(): print(f"GPU Name: {torch.cuda.get_device_name(0)}")`

### Quick Start
1. **Loading the Model**: Use the TinyLlama model with 4-bit quantization by importing `AutoModelForCausalLM`, `AutoTokenizer`, `BitsAndBytesConfig` from transformers. Configure `BitsAndBytesConfig` with `load_in_4bit=True`, `bnb_4bit_quant_type='nf4'`, `bnb_4bit_use_double_quant=True`, `bnb_4bit_compute_dtype=torch.bfloat16`. Load the tokenizer with `AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)` and the model with `AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map='auto', trust_remote_code=True)`.

2. **Configure LoRA**: Import `LoraConfig` and `get_peft_model` from peft. Define LoRA configuration with `r=16`, `lora_alpha=32`, `target_modules=["q_proj","v_proj"]`, `lora_dropout=0.05`, `bias="none"`, `task_type="CAUSAL_LM"`. Wrap the model with `get_peft_model(model, lora_config)`.

3. **Prepare Data and Train**: The notebook includes complete data loading and training loops for fine-tuning on custom datasets.

### Model Specifications
- Base Model: TinyLlama-1.1B-Chat-v0.6  
- Architecture: Transformer (1.1B parameters)  
- Quantization: 4-bit (NF4)  
- Training Method: LoRA (LoRA Rank: 16)  
- Max Context Length: 2048 tokens  
- Training Hardware: Google Colab T4 GPU  

### Usage Examples
- **Basic Inference**:  
Set `prompt = "What is machine learning?"` and tokenize it with `inputs = tokenizer(prompt, return_tensors="pt").to(model.device)`. Generate output with `outputs = model.generate(**inputs, max_length=100)` and decode with `response = tokenizer.decode(outputs[0], skip_special_tokens=True)`.

- **Fine-tuning on Custom Data**: Follow the notebook for dataset loading, training configuration (learning rate, batch size, epochs), evaluation metrics, and model saving/loading.

- **Python Snippet Example**:  
Define `def reverse_string(s): return s[::-1]` and run `print(reverse_string("Hello, World!"))`.

- **Integrating with IDE**: Follow `docs/integration.md` to set up Code-Helper with your preferred IDE.

### Educational Value
Learn model quantization techniques, parameter-efficient fine-tuning (LoRA), transformer architecture and attention mechanisms, PyTorch and Hugging Face ecosystem, GPU optimization techniques, and production ML workflows.

### Performance Metrics
- Model Size: ~1.1B parameters  
- Quantized Size: ~550MB (4-bit)  
- Inference Speed: Fast on T4 GPUs  
- Training Time: Hours for medium datasets  
- Memory Usage: Optimized for <10GB VRAM  

### Key Code Sections
- Environment Setup – Library installation, CUDA verification  
- Model Loading – BitsAndBytes quantization, tokenizer initialization  
- LoRA Configuration – Parameter definition, model wrapping with PEFT  
- Data Processing – Dataset loading, tokenization, train/validation split  
- Training – Training loop, loss tracking, checkpointing  

### Troubleshooting
- **Out of Memory (OOM) Errors**: Reduce batch size, enable gradient accumulation, use smaller datasets  
- **Slow Training**: Ensure GPU usage, adjust batch size/learning rate, enable mixed precision  
- **Model Loading Issues**: Set `trust_remote_code=True`, check internet connection, verify Hugging Face Hub access  

### Resources
Hugging Face Transformers Documentation, PEFT GitHub Repository, TinyLlama Model Card, BitsAndBytes Documentation  

### Contributing
Contributions are welcome! Report issues, suggest improvements, submit pull requests, and share fine-tuning results.  

### License
MIT License  

### Author
Saprit Anand  
GitHub: [@sapritanand](https://github.com/sapritanand)  
Project: Code-Helper Repository  

### Acknowledgments
Hugging Face for the Transformers library, TinyLlama team for the efficient model, and the open-source ML community.  

> Note: Uses free GPU resources from Google Colab for demo. For production, consider dedicated GPU servers or cloud services.  

