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

## Step 2: Training

### 2.1 Configure Training

Edit `configs/train.yaml` or create model-specific configs:

- `train_llama3_8b.yaml` - For Llama 3 8B base
- `train_mistral_7b.yaml` - For Mistral 7B base
- `train_llama32_3b.yaml` - For Llama 3.2 3B Instruct

Key settings:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `data_format` | `prompt_completion` | As discussed above |
| `epochs` | 8-16 | Start with 8, increase if needed |
| `batch_size` | 4-8 | Adjust based on VRAM |
| `lr` | 0.001-0.003 | LoRA learning rate |
| `max_length` | 512 | Adjust based on response lengths |

### 2.2 Run Training

Local training with each model:

    make train TRAIN_CONFIG=train_llama3_8b
    make train TRAIN_CONFIG=train_mistral_7b
    make train TRAIN_CONFIG=train_llama32_3b

### 2.3 Monitor Training

    tensorboard --logdir outputs/runs/latest/logs

**What to watch for:**
- Loss decreasing steadily
- No sudden spikes in gradient norm
- Stop early if loss plateaus

---

## Step 3: Evaluation

### 3.1 Create Evaluation Data

Create `data/eval/test.jsonl` with persona-based test cases:

    {"prompt": "You are Tilda...\n\nHuman: What is your view on traditional exams?\n\nTilda:", "completion": "..."}
    {"prompt": "You are Anis...\n\nHuman: How did you transform the sandy soil?\n\nAnis:", "completion": "..."}

### 3.2 Run Evaluation

    # Evaluate each fine-tuned model
    make eval

    # Compare with baseline (unfine-tuned model)
    make eval-baseline

---

## Step 4: Publish to HuggingFace

### 4.1 Set Credentials

    export HF_TOKEN="hf_xxxxxxxxxxxxx"
    export HF_REPO_ID="username/persona-model"

### 4.2 Push Model

    make publish-hf

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
| | Training Round 1 | Pending | Train all 3 models |
| | Evaluation Round 1 | Pending | Compare metrics across models |
| | Iteration/Refinement | Pending | Adjust based on results |
| | HuggingFace Push | Pending | Publish best performer |

---

## Next Steps

1. [x] Create data extraction/formatting script
2. [x] Generate dataset (520 records: 458 train, 62 eval)
3. [ ] Create training configs for all 3 models
4. [ ] **Round 1**: Train Llama 3 8B, Mistral 7B, Llama 3.2 3B Instruct
5. [ ] Evaluate all models, compare metrics
6. [ ] **Round 2**: Refine best performer(s), adjust hyperparameters
7. [ ] **Round 3**: Final tuning if needed
9. [ ] Push best model to HuggingFace