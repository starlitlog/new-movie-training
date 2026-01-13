# Persona-Based LLM Training Guide

**Author: Dr Ylli Prifti**

Fine-tune an LLM to embody documentary/film personas - responding with their knowledge and speaking style.

## Quick Links

- [MODEL_USAGE.md](MODEL_USAGE.md) - How to use the trained model
- [TRAINING_LOG.md](TRAINING_LOG.md) - Detailed training logs and per-model results
- [HuggingFace Model](https://huggingface.co/ylliprifti/documentary-personas/tree/main/llama31-8b-instruct) - Download the winning model

---

## Results Summary

### Winner: Llama 3.1 8B Instruct

| Metric | Value | Target | Achievement |
|--------|-------|--------|-------------|
| ROUGE-1 | 0.345 | 0.45-0.50 | 70% |
| ROUGE-2 | 0.149 | 0.20-0.25 | 65% |
| BLEU | 0.135 | 0.18-0.22 | 68% |

9 models tested across 2 phases. Key findings:

1. **Model size doesn't guarantee better results** - 8B model outperformed 14B, 27B, and 32B models
2. **Instruct models work well for persona learning** - Contrary to initial hypothesis
3. **Data quality > model size** - Architecture efficiency matters more than parameter count
4. **Contrastive data helps** - Phase 2 improved results by ~5%

### Limitation

Training data volume (171 samples) limited final metrics. With ~1,500 quality samples, target metrics should be achievable.

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (12GB+ VRAM for 8B models)
- Git

### Setup

```bash
git clone https://github.com/starlitlog/new-movie-training.git
cd new-movie-training/pipeline

# Create and activate environment
make venv
source .venv/bin/activate

# Install dependencies
make install

# Verify
make help
```

### Directory Structure

```
new-movie-training/
├── pipeline/
│   ├── data/              # Training data (encrypted)
│   ├── configs/           # Training configurations
│   ├── src/               # Pipeline source code
│   ├── scripts/           # Utility scripts
│   └── outputs/           # Model outputs
├── README.md              # This file
├── MODEL_USAGE.md         # End-user documentation
└── TRAINING_LOG.md        # Detailed training logs
```

---

## Methodology

### 1. Data Preparation

**Personas trained:**

| Persona | Description | Key Topics |
|---------|-------------|------------|
| **Tilda** | Actress, runs Drumduan school | Education, exams, childhood development |
| **Ahsan** | Dhaka Lit Festival director | Literature, festivals, Bangladesh culture |
| **Anis** | Tea plantation owner | Sustainable farming, organic agriculture |

**Data format:** `prompt/completion` pairs for cross-model compatibility.

```json
{
  "prompt": "You are Tilda, an actress who runs Drumduan school...\n\nHuman: What do you think about exams?\n\nTilda:",
  "completion": " This is a school which employs the use of no exams at all..."
}
```

**Data generation types:**
- **Extracted** - Direct quotes from transcripts
- **Transformed** - Same content, rephrased
- **Hypothetical** - New scenarios matching persona tone

Generate data:
```bash
cd pipeline
python3 generate_persona_data.py
```

### 2. Training Configuration

**LoRA settings (all models):**
- `lora_r`: 64
- `lora_alpha`: 128
- `lora_dropout`: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Epochs: 6 (reduced from 10 after observing convergence)
- `max_length`: 512

**Available configs:**

| Config | Model | Batch | Notes |
|--------|-------|-------|-------|
| `train_llama31_8b_instruct.yaml` | Llama 3.1 8B Instruct | 4 | **Recommended** |
| `train_llama3_8b.yaml` | Llama 3 8B | 4 | Base model |
| `train_mistral_7b.yaml` | Mistral 7B | 4 | - |
| `train_qwen2_14b.yaml` | Qwen2.5 14B | 2 | Gradient checkpointing |
| `train_gemma2_27b.yaml` | Gemma 2 27B | 1 | QLoRA (4-bit) |

### 3. Training Pipeline

```
Tokenize → Train (LoRA) → Merge Weights → Evaluate → Publish
```

---

## How to Repeat

### Full Pipeline

```bash
cd pipeline

# 1. Tokenize (optional, speeds up training)
make tokenize TRAIN_CONFIG=train_llama31_8b_instruct

# 2. Train
make train TRAIN_CONFIG=train_llama31_8b_instruct

# 3. Merge LoRA weights into base model
make merge

# 4. Evaluate
make eval EVAL_CONFIG=eval_llama31_8b_instruct

# 5. Compare with baseline
make eval EVAL_CONFIG=eval_llama31_8b_instruct_baseline
```

### Background Training

```bash
# Run in tmux session
make train-bg TRAIN_CONFIG=train_llama31_8b_instruct
```

### Monitor Training

```bash
tensorboard --logdir outputs/runs/latest/logs
```

### Export to GGUF

```bash
make gguf TRAIN_CONFIG=train_llama31_8b_instruct QUANT=Q5_K_M
```

### Publish to HuggingFace

```bash
export HF_REPO_ID="username/repo"
make publish-hf TRAIN_CONFIG=train_llama31_8b_instruct
```

---

## Model Comparison

### All Models Tested

| Rank | Model | Size | ROUGE-1 | Notes |
|------|-------|------|---------|-------|
| 1 | **Llama 3.1 8B Instruct** | 8B | **0.345** | Winner (Phase 2) |
| 2 | Llama 3 8B Instruct | 8B | 0.328 | Phase 1 winner |
| 3 | Mistral 7B | 7B | 0.321 | Best base model |
| 4 | Qwen3 32B | 32B | 0.320 | Large model test |
| 5 | Gemma 3 27B | 27B | 0.316 | Multimodal architecture |
| 6 | Llama 3 8B | 8B | 0.296 | - |
| 7 | Mistral 7B Instruct | 7B | 0.295 | - |
| 8 | Qwen2.5 14B | 14B | 0.289 | - |
| 9 | Qwen2.5 14B Instruct | 14B | 0.256 | Worst performer |

### Base vs Instruct

| Model Family | Winner | Margin |
|--------------|--------|--------|
| Llama | Instruct | +10% |
| Mistral | Base | +10% |
| Qwen | Base | +13% |

No universal pattern - model family matters more than base vs instruct.

---

## Using the Trained Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load from HuggingFace
model_id = "ylliprifti/documentary-personas"
subfolder = "llama31-8b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=subfolder)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    subfolder=subfolder,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Chat with a persona
prompt = "You are Tilda, an actress who runs Drumduan school.\n\nHuman: Why no exams?\n\nTilda:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response[len(prompt):])
```

For detailed usage examples, see [MODEL_USAGE.md](MODEL_USAGE.md).

---

## Future Work

To reach target metrics (ROUGE-1: 0.45+):

1. **Interview the creators** - Capture authentic voice directly from Tilda, Ahsan, and Anis
2. **Expand to ~1,500 training samples** - Currently 171, need ~10x more
3. **Add negative examples** - Prevent persona blending (Persona A knows X, Persona B does not)
4. **Human validation** - Review AI-generated training data for quality

---

## License

See base model licenses for usage terms.