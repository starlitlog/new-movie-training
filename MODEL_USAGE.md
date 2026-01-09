# Persona Model Usage Guide

**Author: Dr Ylli Prifti**

This guide explains how to use the fine-tuned persona model across different platforms and tools.

## Overview

The model has been fine-tuned to role-play as specific personas from a documentary/film:

| Persona | Description | Key Topics |
|---------|-------------|------------|
| **Tilda** | Actress, runs Drumduan school in Scotland | Education philosophy, exams, childhood development, Steiner methods |
| **Ahsan** | Dhaka Lit Festival director, poet | Literature, festival organizing, Bangladesh culture |
| **Anis** | Tea plantation owner | Sustainable farming, organic agriculture, community cooperatives |

## Available Model Formats

| Format | File | Use Case |
|--------|------|----------|
| **SafeTensors** | `model.safetensors` | Transformers, Python inference |
| **GGUF F16** | [`model-f16.gguf`](https://huggingface.co/ylliprifti/documentary-personas/blob/main/model-f16.gguf) | Ollama, llama.cpp, full precision |

---

## Prompt Format

The model was trained with this prompt structure:

    You are {PERSONA_NAME}, {persona_description}.

    Human: {user_question}

    {PERSONA_NAME}:

**Example:**

    You are Tilda, an acclaimed actress who runs Drumduan school in Scotland. You speak thoughtfully about education, childhood development, and your philosophy of exam-free learning.

    Human: What do you think about traditional exams?

    Tilda:

---

## 1. Jupyter Notebook / Google Colab

### Quick Start (Colab)

Open a new Colab notebook and run:

```python
# Install dependencies
!pip install transformers torch accelerate -q

# Load model from HuggingFace
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_REPO = "ylliprifti/documentary-personas"  # Replace with actual repo

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### Chat Function

```python
def chat_as_persona(persona: str, description: str, question: str) -> str:
    """Generate a response as a specific persona."""

    prompt = f"You are {persona}, {description}.\n\nHuman: {question}\n\n{persona}:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()
```

### Example Usage

```python
# Define personas
PERSONAS = {
    "Tilda": "an acclaimed actress who runs Drumduan school in Scotland, passionate about exam-free education",
    "Anis": "a tea plantation owner who transformed sandy soil into thriving organic farms",
    "Ahsan": "the director of Dhaka Literary Festival and an accomplished poet"
}

# Chat with Tilda
response = chat_as_persona(
    "Tilda",
    PERSONAS["Tilda"],
    "What do you think about traditional exams?"
)
print(response)

# Chat with Anis
response = chat_as_persona(
    "Anis",
    PERSONAS["Anis"],
    "How did you transform the sandy soil?"
)
print(response)
```

### Memory-Efficient Loading (for free Colab)

```python
# Use 4-bit quantization for limited GPU memory
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    quantization_config=bnb_config,
    device_map="auto"
)
```

---

## 2. Ollama (Local Computer)

Ollama provides the easiest way to run GGUF models locally with a simple CLI.

### Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

### Create Modelfile

Create a file named `Modelfile`:

```
FROM ./model-f16.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048

SYSTEM "You are a persona from a documentary about education and sustainable farming. You can embody Tilda (actress and educator), Anis (tea farmer), or Ahsan (literary festival director). Respond as the persona specified in the prompt."
```

### Build and Run

```bash
# Download the GGUF file from HuggingFace
huggingface-cli download ylliprifti/documentary-personas model-f16.gguf --local-dir .

# Create the Ollama model
ollama create persona-model -f Modelfile

# Run interactively
ollama run persona-model

# Or use the API
curl http://localhost:11434/api/generate -d '{
  "model": "persona-model",
  "prompt": "You are Tilda, an actress who runs Drumduan school.\n\nHuman: Why no exams?\n\nTilda:",
  "stream": false
}'
```

### Ollama with Python

```python
import requests

def chat_ollama(persona: str, description: str, question: str) -> str:
    prompt = f"You are {persona}, {description}.\n\nHuman: {question}\n\n{persona}:"

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "persona-model",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# Example
print(chat_ollama("Tilda", "an educator", "What makes your school different?"))
```

---

## 3. Open WebUI (Chat Interface)

Open WebUI provides a ChatGPT-like interface for local models.

### Install with Docker

```bash
# Start Open WebUI with Ollama support
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

### Configure

1. Open http://localhost:3000 in your browser
2. Create an account (local only)
3. Go to Settings > Connections
4. Ollama URL: `http://host.docker.internal:11434`
5. Select your `persona-model` from the model dropdown

### Custom System Prompts

In Open WebUI, set custom system prompts for each persona:

**Tilda Prompt:**
```
You are Tilda, an acclaimed actress who runs Drumduan school in Scotland. You speak thoughtfully about education philosophy, childhood development, and your belief in exam-free learning. Your children went to university without ever taking traditional exams.
```

**Anis Prompt:**
```
You are Anis, a tea plantation owner who transformed 400 acres of sandy soil into organic farms. You speak passionately about sustainable agriculture, composting, and building soil through leguminous crops and nitrogen fixation.
```

---

## 4. llama.cpp / vLLM / Low-Level Tools

### llama.cpp (CLI)

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j

# Run inference
./build/bin/llama-cli \
  -m /path/to/model-f16.gguf \
  -p "You are Tilda, an educator.\n\nHuman: Why no exams?\n\nTilda:" \
  -n 256 \
  --temp 0.7 \
  --top-p 0.9
```

### llama.cpp Server (OpenAI-compatible API)

```bash
# Start server
./build/bin/llama-server \
  -m /path/to/model-f16.gguf \
  --host 0.0.0.0 \
  --port 8080

# Use with OpenAI client
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "You are Tilda...\n\nHuman: Question?\n\nTilda:",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### llama-cpp-python

```python
from llama_cpp import Llama

# Load model
llm = Llama(
    model_path="./model-f16.gguf",
    n_ctx=2048,
    n_gpu_layers=-1  # Use all GPU layers
)

# Generate
output = llm(
    "You are Tilda, an educator.\n\nHuman: Why no exams?\n\nTilda:",
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    stop=["\nHuman:", "\n\n"]
)

print(output["choices"][0]["text"])
```

### vLLM (High-Performance Serving)

```python
from vllm import LLM, SamplingParams

# Load model (safetensors format)
llm = LLM(model="ylliprifti/documentary-personas", dtype="float16")

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

prompts = [
    "You are Tilda, an educator.\n\nHuman: Why no exams?\n\nTilda:",
    "You are Anis, a farmer.\n\nHuman: How do you build soil?\n\nAnis:"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### vLLM Server

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model ylliprifti/documentary-personas \
  --dtype float16 \
  --port 8000

# Compatible with OpenAI API
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ylliprifti/documentary-personas",
    "prompt": "You are Tilda...",
    "max_tokens": 256
  }'
```

---

## 5. Bare Metal GPU + Python

For maximum control over inference without frameworks.

### Direct PyTorch Inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model_path = "ylliprifti/documentary-personas"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16
).cuda()

model.eval()

def generate(prompt: str, max_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Use
prompt = "You are Tilda, an educator.\n\nHuman: Why no exams?\n\nTilda:"
response = generate(prompt)
print(response[len(prompt):])
```

### Batch Inference

```python
def batch_generate(prompts: list[str], max_tokens: int = 256) -> list[str]:
    """Generate responses for multiple prompts efficiently."""

    # Tokenize all prompts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [r[len(p):].strip() for r, p in zip(responses, prompts)]

# Batch example
prompts = [
    "You are Tilda...\n\nHuman: About exams?\n\nTilda:",
    "You are Anis...\n\nHuman: About soil?\n\nAnis:",
    "You are Ahsan...\n\nHuman: About literature?\n\nAhsan:"
]

responses = batch_generate(prompts)
for p, r in zip(["Tilda", "Anis", "Ahsan"], responses):
    print(f"[{p}]: {r}\n")
```

### Memory Optimization

```python
# For GPUs with limited VRAM

# Option 1: 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map="auto"
)

# Option 2: 4-bit quantization
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

# Option 3: CPU offloading for very large models
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload"
)
```

---

## 6. Additional Tools and Integrations

### LangChain

```python
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import torch

# Create pipeline
pipe = pipeline(
    "text-generation",
    model="ylliprifti/documentary-personas",
    torch_dtype=torch.float16,
    device_map="auto"
)

llm = HuggingFacePipeline(pipeline=pipe)

# Create prompt template
template = PromptTemplate(
    input_variables=["persona", "description", "question"],
    template="You are {persona}, {description}.\n\nHuman: {question}\n\n{persona}:"
)

# Create chain
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=template)

# Run
response = chain.run(
    persona="Tilda",
    description="an educator who runs Drumduan school",
    question="What do you think about exams?"
)
print(response)
```

### Gradio Web Interface

```python
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained("ylliprifti/documentary-personas")
model = AutoModelForCausalLM.from_pretrained(
    "ylliprifti/documentary-personas",
    torch_dtype=torch.float16,
    device_map="auto"
)

PERSONAS = {
    "Tilda": "an acclaimed actress who runs Drumduan school in Scotland",
    "Anis": "a tea plantation owner who practices organic farming",
    "Ahsan": "the director of Dhaka Literary Festival"
}

def chat(persona, question, temperature, max_tokens):
    description = PERSONAS[persona]
    prompt = f"You are {persona}, {description}.\n\nHuman: {question}\n\n{persona}:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# Create interface
demo = gr.Interface(
    fn=chat,
    inputs=[
        gr.Dropdown(choices=list(PERSONAS.keys()), label="Persona", value="Tilda"),
        gr.Textbox(label="Your Question", lines=3),
        gr.Slider(0.1, 1.0, value=0.7, label="Temperature"),
        gr.Slider(50, 512, value=256, step=50, label="Max Tokens")
    ],
    outputs=gr.Textbox(label="Response", lines=10),
    title="Persona Chat",
    description="Chat with personas from the documentary"
)

demo.launch()
```

### Text Generation Inference (TGI)

```bash
# Run TGI container
docker run --gpus all -p 8080:80 \
  -v /path/to/model:/model \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id /model \
  --dtype float16

# Query
curl http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "You are Tilda...\n\nHuman: Why no exams?\n\nTilda:",
    "parameters": {"max_new_tokens": 256, "temperature": 0.7}
  }'
```

### OpenAI-Compatible Clients

Since llama.cpp server and vLLM provide OpenAI-compatible APIs:

```python
from openai import OpenAI

# Point to local server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.completions.create(
    model="persona-model",
    prompt="You are Tilda...\n\nHuman: About exams?\n\nTilda:",
    max_tokens=256,
    temperature=0.7
)

print(response.choices[0].text)
```

---

## 7. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Out of memory | Use 4-bit quantization or smaller GGUF quantization |
| Slow generation | Enable GPU layers in llama.cpp (`-ngl 99`) |
| Responses off-topic | Use full prompt format with system context |
| Model breaks character | Lower temperature (0.5-0.7) |
| Repetitive output | Increase temperature or add repetition penalty |

### Recommended Settings

| Parameter | Recommended | Range |
|-----------|-------------|-------|
| temperature | 0.7 | 0.5 - 0.9 |
| top_p | 0.9 | 0.8 - 0.95 |
| max_tokens | 256 | 128 - 512 |
| repetition_penalty | 1.1 | 1.0 - 1.2 |

---

## Model Card

- **Base Model**: meta-llama/Meta-Llama-3-8B (or variant)
- **Fine-tuning Method**: LoRA (r=64, alpha=128)
- **Training Data**: Documentary transcripts (2020, 2023)
- **Training Framework**: llm-training-workshop
- **Quantization**: F16 (GGUF)
- **Use Case**: Persona role-play, educational dialogue
- **License**: [Check base model license]

---

## Quick Reference

```bash
# Ollama (easiest)
ollama run persona-model "You are Tilda...\n\nHuman: About exams?\n\nTilda:"

# llama.cpp
./llama-cli -m model.gguf -p "prompt" -n 256

# Python (transformers)
from transformers import pipeline
pipe = pipeline("text-generation", model="ylliprifti/documentary-personas")
pipe("You are Tilda...")
```