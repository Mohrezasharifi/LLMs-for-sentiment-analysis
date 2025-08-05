import torch
import gradio as gr
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Loading the pre-trained models and tokenizers
sentiment_model = RobertaForSequenceClassification.from_pretrained("/content/drive/MyDrive/AI_Models/./fine_tuned_roberta_weighted")
sentiment_tokenizer = RobertaTokenizer.from_pretrained("/content/drive/MyDrive/AI_Models/./fine_tuned_roberta_weighted")
emotion_model = RobertaForSequenceClassification.from_pretrained("/content/drive/MyDrive/AI_Models/./emotion_model")
emotion_tokenizer = RobertaTokenizer.from_pretrained("/content/drive/MyDrive/AI_Models/./emotion_model")
intention_model = RobertaForSequenceClassification.from_pretrained("/content/drive/MyDrive/AI_Models/./intention_model")
intention_tokenizer = RobertaTokenizer.from_pretrained("/content/drive/MyDrive/AI_Models/./intention_model")

sentiment_model.eval()
emotion_model.eval()
intention_model.eval()

def predict_label(text, model, tokenizer, id2label):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        top_idx = torch.argmax(probs, dim=1).item()
        label = id2label[top_idx] if isinstance(list(id2label.keys())[0], int) else id2label[str(top_idx)]
        confidence = probs[0][top_idx].item()
        return label, round(confidence * 100, 1)


def generate_recommendations(sentiment, emotion, intention):
    recommendations = []

    if emotion == 'Frustration':
        recommendations.append("Improve charging station reliability and maintenance schedules.")
    if emotion == 'Confusion':
        recommendations.append("Simplify charging app interface and provide better user guidance.")
    if emotion == 'Anxiety':
        recommendations.append("Enhance customer communication and provide reassurance.")
    if emotion == 'Satisfaction':
        recommendations.append("Identify and replicate successful practices across the network.")

    if intention == 'Reporting Issue':
        recommendations.append("Implement proactive monitoring and faster response times.")
    if intention == 'Comparison':
        recommendations.append("Analyze competitor features and pricing strategies.")
    if intention == 'Praise':
        recommendations.append("Identify and replicate successful practices across the network.")
    if intention == 'Suggestion':
        recommendations.append("Evaluate user suggestions for feature improvements and implementation.")
    if intention == 'Request':
        recommendations.append("Improve response times and support accessibility.")

    if sentiment in ['Very Negative', 'Negative']:
        recommendations.append("Implement immediate damage control and customer retention strategies.")

    if not recommendations:
        recommendations.append("Immediate action required.")

    return "\n".join(recommendations)


def classify_feedback(text):
    sentiment, s_conf = predict_label(text, sentiment_model, sentiment_tokenizer, sentiment_model.config.id2label)
    emotion, e_conf = predict_label(text, emotion_model, emotion_tokenizer, emotion_model.config.id2label)
    intention, i_conf = predict_label(text, intention_model, intention_tokenizer, intention_model.config.id2label)

    recommendations = generate_recommendations(sentiment, emotion, intention)

    return (sentiment, s_conf,
            emotion, e_conf,
            intention, i_conf,
            recommendations)


with gr.Blocks(theme=gr.themes.Soft(), css=".gr-box {background-color: #f9f9f9;text-align:center;}") as demo:
    gr.Markdown(
        """
        # LLMs for Sentiment Analysis of EV Driver Feedback
        ### (final MVP) Sentiment → Emotion → Intention Analysis + Actionable Recommendations
        Enter EV feedback and analyze it using LLM-powered insights.
        """
    )

    with gr.Row():
        input_text = gr.Textbox(
            lines=4,
            placeholder="Write/Paste EV charging feedback here...",
            label="User Feedback"
        )
    classify_btn = gr.Button("Analyze Feedback")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Sentiment (1-Very Negative, 2-Negative, 3-Neutral, 4-Positive, 5-Very Positive)")
            sentiment_out = gr.Label(label="Sentiment")
            sentiment_conf = gr.Label(label="Confidence %")

        with gr.Column():
            gr.Markdown("### Emotion (GoEmotions)")
            emotion_out = gr.Label(label="Emotion")
            emotion_conf = gr.Label(label="Confidence %")

        with gr.Column():
            gr.Markdown("### Intention")
            intention_out = gr.Label(label="Intention")
            intention_conf = gr.Label(label="Confidence %")

    gr.Markdown("### Recommended Actions")
    recommendations_out = gr.Textbox(
        lines=5,
        label="Recommendations",
        interactive=False
    )

    classify_btn.click(
        fn=classify_feedback,
        inputs=input_text,
        outputs=[
            sentiment_out, sentiment_conf,
            emotion_out, emotion_conf,
            intention_out, intention_conf,
            recommendations_out
        ]
    )

    gr.Markdown("---")

    gr.HTML(
        """
        <div style="text-align:center; font-size:0.9rem; color:#555;">
            Developed by <strong>Mohammad Sharifi</strong> (230533851), School of Computing, Newcastle University<br>
            View the project on <a href="https://github.com/" target="_blank">GitHub</a><br><br>
        </div>
        """
    )

demo.launch()
