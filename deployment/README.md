# Smart MCQ Solver

## Overview

This project uses a fine-tuned DeBERTa-v3-base model to rank the five answer options (A–E) for a multiple-choice question and return the Top-3 most likely answers.

## Model

- Base Model: microsoft/deberta-v3-base
- Fine-tuned for binary classification
- Hosted on Hugging Face:
  https://huggingface.co/vvenkata/smart-mcq-solver-deberta-base

## Input

- Question
- Option A
- Option B
- Option C
- Option D
- Option E

## Output

Top-3 ranked answer options with confidence scores.

## Author

Venkata Ganapathi Subramanian