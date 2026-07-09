import gradio as gr
from predict import predict


def run_prediction(prompt, A, B, C, D, E):

    results = predict(prompt, A, B, C, D, E)

    output = "Top 3 Predictions\n\n"

    for rank, (label, score) in enumerate(results, start=1):
        output += f"{rank}. Option {label} (Score: {score:.4f})\n"

    return output


demo = gr.Interface(
    fn=run_prediction,

    inputs=[
        gr.Textbox(lines=5, label="Question"),
        gr.Textbox(label="Option A"),
        gr.Textbox(label="Option B"),
        gr.Textbox(label="Option C"),
        gr.Textbox(label="Option D"),
        gr.Textbox(label="Option E"),
    ],

    outputs=gr.Textbox(label="Predicted Top 3 Answers"),

    title="Smart MCQ Solver",
    description="DeBERTa-v3-base model fine-tuned for Top-3 MCQ Answer Prediction",
)

if __name__ == "__main__":
    demo.launch()