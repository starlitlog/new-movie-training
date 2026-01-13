# Training Log

**Author: Dr Ylli Prifti**

Detailed training logs, per-model evaluation results, and experimental findings. For methodology and quick start, see [README.md](README.md).

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
| 2026-01-09 | Training configs | Complete | Created 7 train configs + 14 eval configs; reduced epochs 10→6 after observing convergence |
| 2026-01-09 | **Llama 3 8B** | Complete | Trained, merged, evaluated (ROUGE-1: 0.296, BLEU: 0.114) |
| 2026-01-09 | GGUF conversion | Complete | Created model-f16.gguf |
| 2026-01-09 | HuggingFace Push | Complete | Published to [ylliprifti/documentary-personas](https://huggingface.co/ylliprifti/documentary-personas) |
| 2026-01-10 | **Mistral 7B** | Complete | Trained, merged, evaluated (ROUGE-1: 0.321, BLEU: 0.126) |
| 2026-01-10 | **Llama 3 8B Instruct** | Complete | ROUGE-1: 0.328, BLEU: 0.122 - **LEADER**, instruct > base |
| 2026-01-10 | **Mistral 7B Instruct** | Complete | ROUGE-1: 0.295, BLEU: 0.113 - base > instruct (opposite pattern) |
| 2026-01-10 | **Qwen2.5 14B** | Complete | ROUGE-1: 0.289, BLEU: 0.105 - size ≠ quality |
| 2026-01-11 | **Qwen2.5 14B Instruct** | Complete | ROUGE-1: 0.256, BLEU: 0.078 - worst overall, base > instruct |
| 2026-01-11 | **Phase 1 Complete** | Inconclusive | 6 models tested, persona blending issue identified, moving to Phase 2 |
| 2026-01-11 | Data augmentation | Complete | Added contrastive examples, 171 training samples |
| 2026-01-12 | **Gemma 3 27B** | Complete | ROUGE-1: 0.316 - required multimodal handling (Gemma3ForConditionalGeneration) |
| 2026-01-12 | **Qwen3 32B** | Complete | ROUGE-1: 0.320 - standard LoRA with gradient checkpointing, 21 min training |
| 2026-01-13 | **Llama 3.1 8B Instruct** | Complete | ROUGE-1: 0.345 - **PHASE 2 WINNER**, best overall |
| 2026-01-13 | **Phase 2 Complete** | Complete | Winner: Llama 3.1 8B Instruct, 70% of target metrics achieved |
| | **Future Work** | Pending | Interview creators, expand to ~1,500 samples with negative examples |

---

## Phase 1: Initial Model Comparison

### Hardware

- **GPU**: NVIDIA RTX 8000 (48GB VRAM)
- **Training**: LoRA with fp16 (RTX 8000 doesn't support bf16)

### Llama 3 8B Base

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

### Mistral 7B Base

```json
{
  "samples_evaluated": 62,
  "exact_match": 0.0,
  "avg_jaccard_overlap": 0.177,
  "rouge1": 0.321,
  "rouge2": 0.141,
  "rougeL": 0.259,
  "rougeLsum": 0.258,
  "bleu": 0.126
}
```

**Mistral 7B vs Llama 3 8B:**

| Metric | Llama 3 8B | Mistral 7B | Difference |
|--------|------------|------------|------------|
| ROUGE-1 | 0.296 | **0.321** | +8.4% |
| ROUGE-2 | 0.130 | **0.141** | +8.5% |
| ROUGE-L | 0.228 | **0.259** | +13.6% |
| BLEU | 0.114 | **0.126** | +10.5% |
| Jaccard | 0.160 | **0.177** | +10.6% |

### Llama 3 8B Instruct

```json
{
  "samples_evaluated": 62,
  "exact_match": 0.0,
  "avg_jaccard_overlap": 0.185,
  "rouge1": 0.328,
  "rouge2": 0.148,
  "rougeL": 0.256,
  "rougeLsum": 0.259,
  "bleu": 0.122
}
```

**Base vs Instruct (Llama 3 8B):**

| Metric | Llama 3 8B Base | Llama 3 8B Instruct | Diff |
|--------|-----------------|---------------------|------|
| ROUGE-1 | 0.296 | **0.328** | +10.8% |
| ROUGE-2 | 0.130 | **0.148** | +13.8% |
| ROUGE-L | 0.228 | **0.256** | +12.3% |
| BLEU | 0.114 | **0.122** | +7.0% |
| Jaccard | 0.160 | **0.185** | +15.6% |

**Surprising Finding:** The instruct model outperforms the base model on all metrics! This contradicts our initial hypothesis that base models would be more malleable for persona learning.

### Mistral 7B Instruct

```json
{
  "samples_evaluated": 62,
  "exact_match": 0.0,
  "avg_jaccard_overlap": 0.168,
  "rouge1": 0.295,
  "rouge2": 0.130,
  "rougeL": 0.232,
  "rougeLsum": 0.231,
  "bleu": 0.113
}
```

**Base vs Instruct (Mistral 7B):**

| Metric | Mistral 7B Base | Mistral 7B Instruct | Diff |
|--------|-----------------|---------------------|------|
| ROUGE-1 | **0.321** | 0.295 | -8.1% |
| ROUGE-2 | **0.141** | 0.130 | -7.8% |
| ROUGE-L | **0.259** | 0.232 | -10.4% |
| BLEU | **0.126** | 0.113 | -10.3% |
| Jaccard | **0.177** | 0.168 | -5.1% |

**Result:** For Mistral, base outperforms instruct (opposite of Llama pattern).

### Qwen2.5 14B Base

```json
{
  "samples_evaluated": 62,
  "exact_match": 0.0,
  "avg_jaccard_overlap": 0.159,
  "rouge1": 0.289,
  "rouge2": 0.121,
  "rougeL": 0.223,
  "rougeLsum": 0.224,
  "bleu": 0.105
}
```

**Result:** Qwen2.5 14B Base performs worse than all 7-8B models despite being nearly 2x the size.

### Qwen2.5 14B Instruct

```json
{
  "samples_evaluated": 62,
  "exact_match": 0.0,
  "avg_jaccard_overlap": 0.134,
  "rouge1": 0.256,
  "rouge2": 0.084,
  "rougeL": 0.194,
  "rougeLsum": 0.193,
  "bleu": 0.078
}
```

**Result:** Qwen2.5 14B Instruct is the worst performer overall.

### Phase 1 Leaderboard

| Rank | Model | Size | Type | ROUGE-1 | BLEU |
|------|-------|------|------|---------|------|
| 1 | **Llama 3 8B Instruct** | 8B | Instruct | **0.328** | 0.122 |
| 2 | Mistral 7B | 7B | Base | 0.321 | **0.126** |
| 3 | Llama 3 8B | 8B | Base | 0.296 | 0.114 |
| 4 | Mistral 7B Instruct | 7B | Instruct | 0.295 | 0.113 |
| 5 | Qwen2.5 14B | 14B | Base | 0.289 | 0.105 |
| 6 | Qwen2.5 14B Instruct | 14B | Instruct | 0.256 | 0.078 |

### Base vs Instruct Summary

| Model Family | Winner | Margin |
|--------------|--------|--------|
| **Llama 3 8B** | Instruct | +10% |
| **Mistral 7B** | Base | +10% |
| **Qwen2.5 14B** | Base | +13% |

### Sample Eval Outputs (Llama 3 8B)

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

---

## Phase 1 Conclusion

**Status: Inconclusive - learnings captured, moving to Phase 2**

Phase 1 tested 6 model configurations with the initial dataset. While we gathered useful data, the results are inconclusive due to:

1. **Persona blending issue**: All personas respond similarly regardless of identity context
2. **Mixed base/instruct results**: No clear pattern across model families
3. **Unexpected scale results**: 14B Qwen underperformed 7-8B Llama/Mistral

### Learnings for Phase 2

| Learning | Action |
|----------|--------|
| Personas blend together | Add contrastive training examples (same Q, different persona → different A) |
| Model family matters | Test latest models: Gemma 3, Llama 3.1, Mistral Nemo, Qwen3 |
| Base vs Instruct unclear | Continue testing both variants |
| Size alone doesn't help | Focus on architecture + data quality over scaling |

---

## Phase 2: Data Augmentation + Latest Models

### Hardware

- **GPU**: NVIDIA A100 80GB
- **Training**: Standard LoRA (no quantization) for better quality

### Data Augmentation Strategy

To fix persona blending, added contrastive examples:

1. **Cross-persona answers**: Same question answered differently by each persona
2. **Persona-specific markers**: Embed identity in responses ("At Drumduan, we...", "On my plantation...")
3. **Domain overlap questions**: Force differentiation on shared topics

Dataset: 171 training samples (after quality filtering)

### Gemma 3 27B

- **ROUGE-1**: 0.316
- **Training Time**: ~25 min
- **Notes**: Required multimodal handling (Gemma3ForConditionalGeneration)

### Qwen3 32B

- **ROUGE-1**: 0.320
- **Training Time**: ~21 min
- **Notes**: Standard LoRA with gradient checkpointing

### Llama 3.1 8B Instruct

- **ROUGE-1**: 0.345
- **ROUGE-2**: 0.149
- **BLEU**: 0.135
- **Training Time**: ~5 min
- **Notes**: **PHASE 2 WINNER** - Best overall performance

### Phase 2 Results Summary

| Model | Size | Type | ROUGE-1 | ROUGE-2 | BLEU | Training Time |
|-------|------|------|---------|---------|------|---------------|
| **Llama 3.1 8B Instruct** | 8B | Instruct | **0.345** | **0.149** | **0.135** | ~5 min |
| Qwen3 32B | 32B | Base | 0.320 | - | - | ~21 min |
| Gemma 3 27B | 27B | Base | 0.316 | - | - | ~25 min |

---

## Final Combined Leaderboard (All Phases)

| Rank | Model | Size | Phase | ROUGE-1 | Notes |
|------|-------|------|-------|---------|-------|
| 1 | **Llama 3.1 8B Instruct** | 8B | 2 | **0.345** | Phase 2 winner, contrastive data |
| 2 | Llama 3 8B Instruct | 8B | 1 | 0.328 | Phase 1 winner |
| 3 | Mistral 7B | 7B | 1 | 0.321 | Best base model |
| 4 | Qwen3 32B | 32B | 2 | 0.320 | Larger model, similar results |
| 5 | Gemma 3 27B | 27B | 2 | 0.316 | Multimodal architecture |
| 6 | Llama 3 8B | 8B | 1 | 0.296 | - |
| 7 | Mistral 7B Instruct | 7B | 1 | 0.295 | - |
| 8 | Qwen2.5 14B | 14B | 1 | 0.289 | - |
| 9 | Qwen2.5 14B Instruct | 14B | 1 | 0.256 | Worst performer |

---

## Example Training Output

**Llama 3 8B training log:**

```
Training Pipeline
Using config: train_llama3_8b
Training Start
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
```

**What to monitor:**
- Loss decreasing steadily (started ~2.84, dropping to ~1.85 by epoch 1)
- Gradient norm stable (2.9-3.5 range is healthy)
- No sudden spikes or NaN values

**TensorBoard monitoring:**
```bash
tensorboard --logdir outputs/runs/latest/logs
```
