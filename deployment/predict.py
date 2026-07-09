import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_NAME = "vvenkata/smart-mcq-solver-deberta-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()

OPTION_LABELS = ["A", "B", "C", "D", "E"]


def predict(prompt, A, B, C, D, E):

    options = [A, B, C, D, E]

    scores = []

    with torch.no_grad():

        for option in options:

            inputs = tokenizer(
                prompt,
                option,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            outputs = model(**inputs)

            probs = F.softmax(outputs.logits, dim=1)

            positive_prob = probs[0][1].item()

            scores.append(positive_prob)

    ranked = sorted(
        zip(OPTION_LABELS, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:3]