# Persona Model Usage Guide

This guide explains how to use the fine-tuned persona model to role-play as characters from the documentary/film.

## Overview

The model has been trained on documentary transcripts and can embody different personas, responding with:
- Knowledge and context specific to each character
- The distinctive tone and speaking style of each persona

## Available Personas

| Persona | Description | Key Topics |
|---------|-------------|------------|
| **Tilda** | Actress, runs Drumduan school in Scotland | Education philosophy, exams, childhood development, Steiner methods |
| **Ahsan** | Dhaka Lit Festival director, poet | Literature, festival organizing, Bangladesh culture |
| **Anis** | Tea plantation owner | Sustainable farming, organic agriculture, community cooperatives |
| **Jasper** | Documentary narrator | Fossils, dinosaurs, storytelling |
| **Krysztof** | Educator | Waldorf/Steiner education, child development, classroom philosophy |

---

## Quick Start

### Installation

```bash
pip install transformers torch peft
```

### Load the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Replace with actual HuggingFace repo
MODEL_REPO = "username/persona-model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForCausalLM.from_pretrained(MODEL_REPO)
```

### Generate Responses

```python
def chat_as_persona(persona: str, question: str) -> str:
    """Chat with the model as a specific persona."""

    prompt = f"Act as {persona}. {question}"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# Example usage
response = chat_as_persona("Tilda", "What do you think about traditional exams?")
print(response)
```

---

## Usage Examples

### Example 1: Talking to Tilda about Education

```python
response = chat_as_persona(
    "Tilda",
    "Why did you start a school without exams?"
)
# Expected response reflects Tilda's views on exam-free education,
# her experience with Drumduan school, and her children's outcomes
```

### Example 2: Asking Anis about Sustainable Farming

```python
response = chat_as_persona(
    "Anis",
    "How did you transform arid soil into a tea plantation?"
)
# Expected response discusses organic matter, leguminous crops,
# nitrogen fixation, and the ecosystem that developed
```

### Example 3: Exploring with Jasper

```python
response = chat_as_persona(
    "Jasper",
    "Tell me about dinosaur fossils."
)
# Expected response in Jasper's narrative storytelling style
```

---

## Advanced Usage

### Using System Prompts

For more nuanced persona embodiment, use system prompts:

```python
from transformers import pipeline

generator = pipeline("text-generation", model=MODEL_REPO)

messages = [
    {
        "role": "system",
        "content": "You are Tilda, an acclaimed actress who runs Drumduan school in Scotland. You are passionate about alternative education and speak thoughtfully about childhood development."
    },
    {
        "role": "user",
        "content": "What makes your school different?"
    }
]

response = generator(messages, max_new_tokens=256)
print(response[0]["generated_text"])
```

### Batch Processing

```python
personas_and_questions = [
    ("Tilda", "What is your view on creativity in education?"),
    ("Anis", "How do cooperatives work on your farm?"),
    ("Krysztof", "Why do younger children need different classroom colors?"),
]

for persona, question in personas_and_questions:
    response = chat_as_persona(persona, question)
    print(f"[{persona}]: {response}\n")
```

---

## Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `max_new_tokens` | 256-512 | Response length |
| `temperature` | 0.7-0.9 | Higher = more creative, lower = more consistent |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `do_sample` | True | Enable sampling for varied responses |

---

## Best Practices

1. **Be specific with persona names** - Use exact names as trained
2. **Provide context** - The more context in your question, the better the response
3. **Stay within persona knowledge** - Each persona knows only what they discussed in the source material
4. **Adjust temperature** - Lower for factual responses, higher for creative conversations

---

## Limitations

- Personas can only respond based on topics covered in the training transcripts
- The model may occasionally blend personas if prompts are ambiguous
- Complex multi-turn conversations may require careful context management

---

## Troubleshooting

**Model generates off-topic responses:**
- Use a more explicit system prompt
- Lower the temperature parameter

**Responses don't match persona style:**
- Ensure you're using the exact persona name
- Try providing more context about what the persona would know

**Out of memory errors:**
- Use `torch.float16` or `bfloat16` for inference
- Reduce `max_new_tokens`

---

## Model Card

- **Base Model**: Meta-Llama-3-8B (or specified variant)
- **Fine-tuning Method**: LoRA
- **Training Data**: Documentary transcripts (2020, 2023)
- **Use Case**: Persona role-play and educational dialogue
