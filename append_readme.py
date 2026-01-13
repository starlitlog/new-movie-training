#!/usr/bin/env python3
import os

readme_path = '/home/yprift01/Dev/new-movie-training/README.md'

# Content to append (the missing Python code completion)
lines = [
    "",
    "",
    "Human: What do you think about traditional exams?",
    "",
    'Tilda:"""',
    "",
    'inputs = tokenizer(prompt, return_tensors="pt").to(model.device)',
    "outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)",
    "response = tokenizer.decode(outputs[0], skip_special_tokens=True)",
    "print(response[len(prompt):])",
    "```",
    "",
]

with open(readme_path, 'a') as f:
    f.write('\n'.join(lines))

print("Appended content to README.md")
