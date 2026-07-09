import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------------------------------
# Hugging Face Model Repository
# ----------------------------------------------------
MODEL_NAME = "vvenkata/smart-mcq-solver-deberta-base"

# ----------------------------------------------------
# Device
# ----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------
# Load Tokenizer and Model (only once)
# ----------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()

OPTION_LABELS = ["A", "B", "C", "D", "E"]


# ----------------------------------------------------
# Predict Function
# ----------------------------------------------------
def predict(prompt, A, B, C, D, E):

    options = [A, B, C, D, E]

    scores = []

    with torch.no_grad():

        for option in options:

            encoding = tokenizer(
                prompt,
                option,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            encoding = {
                key: value.to(device)
                for key, value in encoding.items()
            }

            outputs = model(**encoding)

            probabilities = F.softmax(outputs.logits, dim=1)

            positive_probability = probabilities[0][1].item()

            scores.append(positive_probability)

    ranked = sorted(
        zip(OPTION_LABELS, options, scores),
        key=lambda x: x[2],
        reverse=True
    )

    return ranked[:3]