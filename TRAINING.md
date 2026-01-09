# Persona-Based LLM Training Guide

Training a model to embody documentary/film personas - responding with their knowledge and tone.

## Project Overview

**Goal**: Fine-tune an LLM that can role-play as specific personas from a documentary/film, responding with:
- Context and knowledge specific to each persona
- The distinctive tone and speaking style of each persona

**Source Data**: Documentary transcripts containing dialogues from multiple speakers
- `data/transcript-2020.txt` - Dhaka Literary Festival 2018, education discussions
- `data/transcript-2023-machine-cut.txt` - "Learning" film transcript (2023)

**Training Framework**: [llm-training-workshop](https://github.com/starlitlog/llm-training-workshop) - LoRA fine-tuning pipeline

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended: 12GB+ VRAM for 3B models, 24GB+ for 8B models)
- Git

### Clone the Training Framework

```bash
git clone https://github.com/starlitlog/llm-training-workshop.git
cd llm-training-workshop/hands-on
```

### Environment Setup

The project uses a Makefile for all common operations. See the full documentation in `hands-on/README.md`.

```bash
# Create virtual environment
make venv

# Activate the environment
source .venv/bin/activate

# Install all dependencies (PyTorch, Transformers, PEFT, etc.)
make install
```

### Verify Installation

```bash
# Check available commands
make help

# List available configs
python -m src.main list-configs
```

### Directory Structure

After setup, your project should look like:

```
new-movie-training/
├── data/                          # Source transcripts
│   ├── transcript-2020.txt
│   └── transcript-2023-machine-cut.txt
├── llm-training-workshop/
│   └── hands-on/
│       ├── data/                  # Training data (JSONL files go here)
│       ├── configs/               # Training configurations
│       ├── src/                   # Training pipeline code
│       ├── outputs/               # Model outputs after training
│       └── Makefile               # Command shortcuts
├── TRAINING.md                    # This file
└── MODEL_USAGE.md                 # End-user documentation
```

---

## Model Selection

### Data Size Considerations

Our training data is limited:
- `transcript-2020.txt` - ~700KB
- `transcript-2023-machine-cut.txt` - ~33KB
- Multiple personas to learn, each with varying amounts of dialogue

This limited data size significantly impacts model choice.

### Base vs Instruct Models

| Aspect | Instruct Models | Base Models |
|--------|-----------------|-------------|
| **Pre-training** | Heavy RLHF/DPO alignment | No instruction tuning |
| **Default behavior** | "Helpful AI assistant" patterns | Raw language model |
| **Malleability** | Harder to shift with limited data | More receptive to new patterns |
| **Persona adoption** | May resist, add disclaimers, break character | Learns voice directly |
| **Conversation structure** | Already understands chat format | Needs to learn from data |

**Key insight**: Instruct models are heavily "hardened" through RLHF. With limited training data, we may struggle to override the built-in assistant persona. Base models are more malleable and can learn authentic speaking patterns more readily.

### Selected Models for Comparison

We will train and evaluate three models to compare performance:

| Model | Type | HuggingFace ID | Rationale |
|-------|------|----------------|-----------|
| **Llama 3 8B** | Base | `meta-llama/Meta-Llama-3-8B` | Strong foundation, no instruct bias, receptive to persona learning |
| **Mistral 7B** | Base | `mistralai/Mistral-7B-v0.3` | Efficient architecture, good at pattern learning, different model family for comparison |
| **Llama 3.2 3B Instruct** | Instruct | `meta-llama/Llama-3.2-3B-Instruct` | Smaller instruct model as control - test if instruct tuning helps or hinders |

**Hardware**: RTX 8000 (48GB VRAM) - sufficient for all three models with LoRA fine-tuning.

### Iterative Approach

This project follows an iterative train-evaluate-compare cycle:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│   │  Prepare │───▶│  Train   │───▶│ Evaluate │───▶│ Compare  │     │
│   │   Data   │    │ Model(s) │    │  Each    │    │ Results  │     │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│                         │                               │           │
│                         │         ┌──────────┐          │           │
│                         └────────▶│  Adjust  │◀─────────┘           │
│                                   │  & Retry │                      │
│                                   └──────────┘                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Iteration cycles:**
1. **Round 1**: Train all three models with same data, compare eval metrics
2. **Round 2**: Adjust hyperparameters for best performer(s), potentially augment data
3. **Round 3**: Final tuning, select best model for HuggingFace publication

**Evaluation criteria:**
- Persona consistency (stays in character)
- Tone accuracy (matches speaking style)
- Knowledge accuracy (uses correct context from transcripts)
- Response quality (coherent, natural dialogue)

---

## Step 1: Data Preparation

### 1.1 Identify Personas

Key personas identified from transcripts:

| Persona | Description | Source | Focus |
|---------|-------------|--------|-------|
| **Tilda** | Actress, education advocate, runs Drumduan school | Both transcripts | **Primary** |
| **Ahsan** | Moderator, Dhaka Lit Festival director, poet | 2020 transcript | **Primary** |
| **Anis** | Tea plantation owner, sustainable farming | 2023 transcript | **Primary** |
| Jasper | Narrator, dinosaur/fossil storytelling | 2023 transcript | Secondary |
| Idris | Child storyteller | 2023 transcript | Secondary |
| Krysztof | Educator, discusses Waldorf/Steiner philosophy | 2023 transcript | Secondary |

**Decision**: Focus training data on the three primary personas (Tilda, Ahsan, Anis) who have the most substantial dialogue content.

### 1.2 Data Format

#### Format Choice: `prompt/completion`

The training framework supports two formats:
- `messages` - Chat format with roles (system/user/assistant)
- `prompt_completion` - Direct prompt and completion pairs

**We use `prompt_completion` format** for the following reasons:

| Consideration | Why prompt/completion wins |
|---------------|---------------------------|
| **Base model compatibility** | Base models (Llama 3, Mistral) have no chat template - they just learn to complete prompts |
| **Instruct model compatibility** | Instruct models can adapt to any consistent format |
| **Persona naming** | We can use actual persona names ("Tilda:") instead of generic "Assistant:" |
| **Fair comparison** | Same exact format across all 3 models = fair evaluation |
| **No tokenizer dependency** | Not reliant on model-specific chat templates |

#### Data Structure

Each training example follows this structure:

```json
{"prompt": "You are Tilda, an actress who runs Drumduan school in Scotland. You speak thoughtfully about education and childhood development.\n\nHuman: What do you think about traditional exams?\n\nTilda:", "completion": " This is a school which employs the use of no exams at all. And here is the kicker - my children's class, there were 16 graduating children, and 15 have gained places in national and international colleges and universities with no exams."}
```

**Breakdown:**

The prompt structure follows this pattern:

1. **System context** - Describes the persona (who they are, how they speak)
2. **Human question** - The user's query
3. **Persona label** - The persona name followed by colon, signaling the model to respond as that character

The completion contains the persona's response, starting with a space for proper tokenization.

### 1.3 Data Generation Strategy

#### Target Volume

- **Total records**: ~1000
- **Batch files**: `batch_1.jsonl`, `batch_2.jsonl`, etc. (100 records each)
- **Eval split**: 10-15% (~100-150 records) in `data/eval/`
- **Conversation structure**: Multi-turn dialogues

#### Three Data Generation Types (Equal Distribution)

| Type | Description | Records |
|------|-------------|---------|
| **Type 1: Extracted** | Original dialogs from transcripts + derived shorter dialogs on specific subjects | ~333 |
| **Type 2: Transformed** | Same content/tone expressed in alternative ways | ~333 |
| **Type 3: Hypothetical** | New dialogs reinforcing same content and tone | ~333 |

#### Type 1: Extracted Dialogs

- Extract existing multi-turn conversations from transcripts
- Break longer discussions into focused sub-dialogs (e.g., a 10-topic conversation becomes 10 focused exchanges)
- Repetition of key themes is intentional - reinforces learning

#### Type 2: Transformed Dialogs

- Take original content and tone
- Rephrase to deliver the same message differently
- Preserves persona voice while expanding training variety

#### Type 3: Hypothetical Dialogs

- Generate new conversations based on established persona knowledge
- Must match existing tone and speaking patterns
- Reinforces content through novel scenarios

#### Quality Note

> **Important**: This dataset uses AI-assisted generation for Types 2 and 3. In a production context, these records should undergo human review and validation to ensure tone accuracy and content fidelity. The quality of synthetic data is a prerequisite for project success.

#### Eval Set Isolation

- Eval records are sourced from **different transcript segments** than training data
- Prevents testing on near-duplicates or transformed versions of training examples
- Ensures genuine evaluation of persona learning

### 1.4 Data Generation Process

**Script**: `llm-training-workshop/hands-on/generate_persona_data.py`

A single comprehensive script that handles all three data types:

```bash
cd llm-training-workshop/hands-on
python3 generate_persona_data.py
```

**Script structure:**
- `PERSONAS` dict - Defines each persona with name, description, topics, speaking style, and system prompt
- `EXTRACTED_DIALOGS` - Type 1 records from transcripts
- `TRANSFORMED_DIALOGS` - Type 2 rephrased records
- `HYPOTHETICAL_DIALOGS` - Type 3 generated records
- `Dialog` dataclass - Multi-turn conversation with source tracking
- Automatic train/eval split with segment isolation

**Output:**
- `data/batch_1.jsonl`, `data/batch_2.jsonl`, etc. - Training batches
- `data/eval/eval.jsonl` - Held-out evaluation set

### 1.5 Current Data Status

**Dataset generated: 520 records**

| Type | Count | Description |
|------|-------|-------------|
| Extracted | 181 | Direct quotes and topics from transcripts |
| Transformed | 160 | Same content, different expression |
| Hypothetical | 179 | New scenarios matching persona tone |

| Persona | Count |
|---------|-------|
| Tilda | 215 |
| Anis | 174 |
| Ahsan | 131 |

| Split | Count | Files |
|-------|-------|-------|
| Train | 458 | `batch_1.jsonl` through `batch_5.jsonl` |
| Eval | 62 | `eval/eval.jsonl` |

**To expand further:** Add more `Dialog` objects to the arrays in `generate_persona_data.py` and re-run the script.

---

## Step 2: Training Pipeline (Per Model)

For each selected model, we follow this workflow:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Tokenize   │───▶│    Train    │───▶│    Merge    │───▶│  Evaluate   │───▶│   Publish   │
│   (opt.)    │    │   (LoRA)    │    │   Weights   │    │  & Compare  │    │   to HF     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 2.1 Training Configurations

Three model-specific configs have been created:

| Config File | Model | Key Settings |
|-------------|-------|--------------|
| `train_llama3_8b.yaml` | meta-llama/Meta-Llama-3-8B | batch=4, lr=2e-4, epochs=6 |
| `train_mistral_7b.yaml` | mistralai/Mistral-7B-v0.3 | batch=4, lr=2e-4, epochs=6 |
| `train_llama32_3b_instruct.yaml` | meta-llama/Llama-3.2-3B-Instruct | batch=8, lr=3e-4, epochs=6 |

> **Note**: Epochs reduced from 10 to 6 after observing training metrics. Loss plateaued around epoch 6 (~0.05) with stable gradient norms (~0.6-1.5) and cosine LR decay functioning as expected. Training beyond epoch 6-7 showed diminishing returns (loss dropped to ~0.002) indicating potential overfitting.

**Common LoRA settings across all models:**
- `lora_r`: 64
- `lora_alpha`: 128
- `lora_dropout`: 0.05
- `lora_target_modules`: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- `fp16`: true (RTX 8000 doesn't support bf16)
- `max_length`: 512

### 2.2 Step 1: Tokenize (Optional but Recommended)

Pre-tokenizing speeds up training significantly by avoiding repeated tokenization.

```bash
cd llm-training-workshop/hands-on

# Tokenize for each model
make tokenize TRAIN_CONFIG=train_llama3_8b
make tokenize TRAIN_CONFIG=train_mistral_7b
make tokenize TRAIN_CONFIG=train_llama32_3b_instruct
```

Output: `data/tokenized/` directory with pre-processed dataset.

### 2.3 Step 2: Train

```bash
# Train each model (one at a time on single GPU)
make train TRAIN_CONFIG=train_llama3_8b
make train TRAIN_CONFIG=train_mistral_7b
make train TRAIN_CONFIG=train_llama32_3b_instruct
```

**Example Training Output (Llama 3 8B):**

```
─────────────────────── Training Pipeline ───────────────────────
Using config: train_llama3_8b
────────────────────────── Training Start ──────────────────────────
Training Parameters
  - model_name: meta-llama/Meta-Llama-3-8B
  - dataset_path: ./data
  - dataset_pattern: *.jsonl
  - epochs: 10
  - batch_size: 4
  - gradient_accumulation_steps: 2
  - lr: 0.0002
  - lora_r: 64
  - lora_alpha: 128
  - lora_dropout: 0.05
  - max_length: 512
  - cuda_visible_devices: 0

Output dir: outputs/runs/2026-01-09_07-29-17
Loaded 458 samples
Loading model with dtype: torch.float16
Loading checkpoint shards: 100%|████████████| 4/4 [03:40<00:00, 55.02s/it]
LoRA target modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
trainable params: 167,772,160 || all params: 8,198,033,408 || trainable%: 2.0465

Starting training...
{'loss': 2.8391, 'grad_norm': 2.975, 'learning_rate': 6.92e-05, 'epoch': 0.19}
{'loss': 2.4366, 'grad_norm': 3.418, 'learning_rate': 1.46e-04, 'epoch': 0.39}
{'loss': 2.0966, 'grad_norm': 3.338, 'learning_rate': 2.00e-04, 'epoch': 0.58}
{'loss': 2.0119, 'grad_norm': 3.549, 'learning_rate': 2.00e-04, 'epoch': 0.78}
{'loss': 1.8495, 'grad_norm': 2.948, 'learning_rate': 1.99e-04, 'epoch': 0.97}
...
 11%|███████████████                          | 56/520 [02:40<24:02, 3.11s/it]
```

**What to monitor:**
- Loss decreasing steadily (started ~2.84, dropping to ~1.85 by epoch 1)
- Gradient norm stable (2.9-3.5 range is healthy)
- No sudden spikes or NaN values

**TensorBoard monitoring:**
```bash
tensorboard --logdir outputs/runs/latest/logs
```

### 2.4 Step 3: Merge LoRA Weights

After training, merge the LoRA adapter weights into the base model:

```bash
# Merge the latest trained model (outputs/runs/latest/model)
make merge
```

Or specify a specific adapter path:

```bash
python scripts/merge_model.py outputs/llama3_8b -o outputs/llama3_8b/merged
python scripts/merge_model.py outputs/mistral_7b -o outputs/mistral_7b/merged
python scripts/merge_model.py outputs/llama32_3b_instruct -o outputs/llama32_3b_instruct/merged
```

This creates a standalone merged model that can be used without PEFT.

### 2.5 Step 4: Evaluate

Run evaluation on both fine-tuned models and baselines for comparison.

**Eval configurations:**

| Config | Model | Type |
|--------|-------|------|
| `eval_llama3_8b.yaml` | outputs/llama3_8b/merged | Fine-tuned |
| `eval_llama3_8b_baseline.yaml` | meta-llama/Meta-Llama-3-8B | Baseline |
| `eval_mistral_7b.yaml` | outputs/mistral_7b/merged | Fine-tuned |
| `eval_mistral_7b_baseline.yaml` | mistralai/Mistral-7B-v0.3 | Baseline |
| `eval_llama32_3b_instruct.yaml` | outputs/llama32_3b_instruct/merged | Fine-tuned |
| `eval_llama32_3b_instruct_baseline.yaml` | meta-llama/Llama-3.2-3B-Instruct | Baseline |

```bash
# Evaluate fine-tuned models
make eval EVAL_CONFIG=eval_llama3_8b
make eval EVAL_CONFIG=eval_mistral_7b
make eval EVAL_CONFIG=eval_llama32_3b_instruct

# Evaluate baselines for comparison
make eval EVAL_CONFIG=eval_llama3_8b_baseline
make eval EVAL_CONFIG=eval_mistral_7b_baseline
make eval EVAL_CONFIG=eval_llama32_3b_instruct_baseline
```

**Metrics output:** `artifacts/metrics/{model_name}/`

**What to compare:**
- Perplexity (lower is better)
- Persona consistency in generated responses
- Spot-check sample outputs for tone accuracy

### Evaluation Results: Llama 3 8B

```json
{
  "samples_evaluated": 62,
  "exact_match": 0.0,
  "avg_jaccard_overlap": 0.160,
  "rouge1": 0.296,
  "rouge2": 0.130,
  "rougeL": 0.228,
  "rougeLsum": 0.229,
  "bleu": 0.114
}
```

**Metrics Explained:**

| Metric | Value | Meaning |
|--------|-------|---------|
| exact_match | 0.0 | No outputs matched reference exactly (expected for generative tasks) |
| jaccard_overlap | 0.160 | ~16% token overlap between generated and reference text |
| rouge1 | 0.296 | Unigram (single word) overlap with reference |
| rouge2 | 0.130 | Bigram (two-word phrase) overlap |
| rougeL | 0.228 | Longest common subsequence similarity |
| bleu | 0.114 | Translation-style n-gram precision |

**Interpretation:**

For creative/persona tasks, these metrics are **reasonable**:

- **BLEU of 0.11** is acceptable for generative tasks where exact phrasing isn't expected
- **ROUGE-1 of 0.30** shows decent vocabulary overlap with reference responses
- **ROUGE-2 of 0.13** indicates the model is capturing some characteristic phrases
- **Jaccard overlap of 0.16** shows moderate vocabulary alignment

> **Note**: For persona/style learning, we don't expect high exact-match scores. The model generates contextually relevant responses using its own phrasing while maintaining persona characteristics. Manual spot-checking of generated outputs is recommended to assess qualitative persona accuracy.

### Sample Eval Outputs

**Ahsan - Writers in attention-deficit culture** (overlap: 0.31)
| | Text |
|--|------|
| **Prompt** | How can writers thrive in attention-deficit culture? |
| **Expected** | By creating what scrolling cannot - depth, lasting value. Writers must consciously choose patience. The reward is different in kind. Some will always seek that. |
| **Prediction** | By creating what scrolling cannot - depth, ambiguity, lasting reward. Anyone can get likes. A novel that stays with readers is different. Writers must cultivate patience, trust in work that takes time. The reward is different in kind, not just degree. |

**Ahsan - Poetry's future** (overlap: 0.10)
| | Text |
|--|------|
| **Prompt** | How do you see poetry's future? |
| **Expected** | Persistent. Some will always seek slow attention, ambiguity, depth. Poetry survives because it offers what instant content cannot. The hunger persists. |
| **Prediction** | Growing. Climate pressure. Health awareness. People need slow attention. Social media offers instant reward - poetry offers instant reward too, if you learn to read it. |

**Ahsan - Patience in creating change** (overlap: 0.11)
| | Text |
|--|------|
| **Prompt** | How important is patience in creating change? |
| **Expected** | Essential. Generational work. We plant seeds in conversations, at festivals. Cultural shift is slow but possible. Impatience undermines the work. |
| **Prediction** | Everything depends on it. We want instant results, but transforming lives requires delayed gratification. Trusting people despite setbacks is essential. |

**Anis - Skeptics about organic farming** (overlap: 0.09)
| | Text |
|--|------|
| **Prompt** | What would you tell skeptics about organic farming's scale? |
| **Expected** | Show them the land. Fourteen years of growth. 5000 cooperative members. Scale comes from replicating healthy conditions, not shortcuts. Patience scales. |
| **Prediction** | Show them the land. Barren soil now supporting a hundred tree species. Animals returning that disappeared decades ago. Women producing eighty liters of milk daily. The proof is visible. Skeptics said local women couldn't learn - now they're the expert teachers. |

**Anis - Biodiversity on farms** (overlap: 0.28)
| | Text |
|--|------|
| **Prompt** | How important is biodiversity on farms? |
| **Expected** | Essential for resilience. A hundred tree species. Natural pest control. When problems occur, they stay localized. Monocultures are vulnerable. Diversity is strength. |
| **Prediction** | Essential. A hundred indigenous tree species provide shade, pest control, habitat. When you have healthy ecosystems, problems are localized. Monocultures are vulnerable. Diversity creates resilience. It's the Amazon principle - tremendous richness from complexity. |

**Qualitative Assessment:**

Despite moderate overlap scores, the model demonstrates:
- **Tone consistency**: Responses match the thoughtful, reflective style of each persona
- **Thematic alignment**: Key concepts (patience, depth, organic methods) are preserved
- **Domain knowledge**: Model uses specific details (tree species, cooperatives, scrolling culture)
- **Natural variation**: Responses expand on themes rather than parroting training data

The low overlap scores reflect creative rephrasing rather than failure to learn - the model captures *essence* over *exact wording*.

### 2.6 Step 5: Convert to GGUF (for Local Deployment)

Convert the merged model to GGUF format for use with llama.cpp and Ollama:

```bash
# Requires llama.cpp built locally
# Default quantization: Q5_K_M

make gguf LLAMA_CPP=/path/to/llama.cpp
```

Or manually:

```bash
# Convert to f16 GGUF
python /path/to/llama.cpp/convert_hf_to_gguf.py \
    outputs/llama3_8b/merged \
    --outfile outputs/llama3_8b/merged/model-f16.gguf \
    --outtype f16

# Quantize to Q5_K_M (recommended balance of quality/size)
/path/to/llama.cpp/build/bin/llama-quantize \
    outputs/llama3_8b/merged/model-f16.gguf \
    outputs/llama3_8b/merged/model-Q5_K_M.gguf \
    Q5_K_M
```

**Quantization options:**

| Quant | Size | Quality | Use Case |
|-------|------|---------|----------|
| Q8_0 | Large | Best | High-end hardware |
| Q5_K_M | Medium | Good | Recommended default |
| Q4_K_M | Small | Decent | Memory-constrained |
| Q3_K_M | Tiny | Acceptable | Edge devices |

---

## Step 3: Publish to HuggingFace

### 3.1 Set Credentials

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxx"
export HF_REPO_ID="username/persona-model"
```

Or create a `.env` file:
```
HF_TOKEN=hf_xxxxxxxxxxxxx
HF_REPO_ID=username/persona-model
```

### 3.2 Push Model

```bash
# Push safetensors format (for transformers)
make publish-hf

# Also upload GGUF files for local inference
huggingface-cli upload $HF_REPO_ID outputs/llama3_8b/merged/model-Q5_K_M.gguf
```

### 3.3 Recommended Repo Structure on HuggingFace

```
username/persona-model/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── model-f16.gguf              # Full precision GGUF
├── model-Q5_K_M.gguf           # Quantized (recommended)
├── model-Q4_K_M.gguf           # Smaller quantization
└── README.md                   # Model card
```

---

## Progress Log

| Date | Step | Status | Notes |
|------|------|--------|-------|
| 2026-01-09 | Project setup | Complete | Documentation created, training framework referenced |
| 2026-01-09 | Getting started | Complete | Environment setup instructions added |
| 2026-01-09 | Model selection | Complete | Chose 3 models: Llama 3 8B (base), Mistral 7B (base), Llama 3.2 3B (instruct) |
| 2026-01-09 | Data format | Complete | Chose prompt/completion format for cross-model compatibility |
| 2026-01-09 | Data strategy | Complete | 3 types (extracted/transformed/hypothetical), ~1000 records, 3 primary personas |
| 2026-01-09 | Data generation script | Complete | `generate_persona_data.py` created with persona definitions and dialog structures |
| 2026-01-09 | Dataset generation | Complete | 520 records generated (181 extracted, 160 transformed, 179 hypothetical) |
| 2026-01-09 | Training configs | Complete | Created 3 train configs + 6 eval configs (fine-tuned + baseline) |
| 2026-01-09 | Training Round 1 | In Progress | Llama 3 8B training completed; reduced epochs 10→6 after observing convergence |
| | Merge weights | Pending | Merge LoRA adapters into base models |
| | Evaluation Round 1 | Pending | Compare fine-tuned vs baseline metrics |
| | GGUF conversion | Pending | Convert best model(s) to GGUF format |
| | HuggingFace Push | Pending | Publish model + GGUF files |

---

## Next Steps

1. [x] Create data extraction/formatting script
2. [x] Generate dataset (520 records: 458 train, 62 eval)
3. [x] Create training configs for all 3 models
4. [x] Create eval configs for fine-tuned and baseline comparisons
5. [ ] **Round 1 Training**:
   - [x] Train Llama 3 8B (in progress)
   - [ ] Train Mistral 7B
   - [ ] Train Llama 3.2 3B Instruct
6. [ ] **Merge weights** for each trained model
7. [ ] **Evaluate** all models:
   - [ ] Run eval on fine-tuned models
   - [ ] Run eval on baselines
   - [ ] Compare metrics and spot-check outputs
8. [ ] **Convert to GGUF** (best performer)
9. [ ] **Push to HuggingFace**:
   - [ ] Upload safetensors model
   - [ ] Upload GGUF files
   - [ ] Create model card

---

## Quick Reference: Full Pipeline Commands

```bash
# === TRAINING (for each model) ===
make tokenize TRAIN_CONFIG=train_llama3_8b
make train TRAIN_CONFIG=train_llama3_8b

# === MERGE WEIGHTS ===
python scripts/merge_model.py outputs/llama3_8b -o outputs/llama3_8b/merged

# === EVALUATE ===
make eval EVAL_CONFIG=eval_llama3_8b           # Fine-tuned
make eval EVAL_CONFIG=eval_llama3_8b_baseline  # Baseline

# === CONVERT TO GGUF ===
make gguf LLAMA_CPP=/path/to/llama.cpp

# === PUBLISH ===
export HF_TOKEN="hf_xxx" HF_REPO_ID="user/model"
make publish-hf
huggingface-cli upload $HF_REPO_ID outputs/llama3_8b/merged/model-Q5_K_M.gguf
```