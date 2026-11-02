import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# On Hugging Face, models are placed in the same repository folder
CUSTOM_MODEL_PATH = "custom_cnn_rooster_pigeon.keras"
TL_MODEL_PATH = "transfer_learning_mobilenetv2.keras"

# Placeholders for dynamic loading to prevent HF Space startup timeout errors
custom_model = None
tl_model = None

def predict_image(img, model_choice):
    global custom_model, tl_model
    if img is None:
        return {"Error: Please upload an image": 1.0}, "No image provided."

    # Standardize input image to 224x224 RGB
    img_resized = img.convert('RGB').resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)

    results = {}
    summary_text = ""

    if model_choice in ["Custom Self-Designed CNN", "Compare Both Models"]:
        # Load dynamically when needed
        if custom_model is None and os.path.exists(CUSTOM_MODEL_PATH):
            try:
                custom_model = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
            except Exception as e:
                return {"Error loading custom model": 1.0}, f"Failed to load custom model: {str(e)}"

        if custom_model is not None:
            prob_rooster = float(custom_model.predict(img_batch, verbose=0)[0][0])
            prob_pigeon = 1.0 - prob_rooster
            pred_label = "Rooster" if prob_rooster >= 0.5 else "Pigeon"
            conf = max(prob_rooster, prob_pigeon) * 100
            summary_text += f"**Custom CNN (.keras) Prediction:** {pred_label} ({conf:.2f}% confidence)\n\n"
            if model_choice == "Custom Self-Designed CNN":
                results = {"Rooster": prob_rooster, "Pigeon": prob_pigeon}
        else:
            summary_text += "**Custom CNN:** Weights not found on disk. Please upload custom_cnn_rooster_pigeon.keras\n\n"

    if model_choice in ["Transfer Learning (MobileNetV2)", "Compare Both Models"]:
        # Load dynamically when needed
        if tl_model is None and os.path.exists(TL_MODEL_PATH):
            try:
                tl_model = tf.keras.models.load_model(TL_MODEL_PATH)
            except Exception as e:
                return {"Error loading Transfer Learning model": 1.0}, f"Failed to load Transfer Learning model: {str(e)}"

        if tl_model is not None:
            prob_rooster_tl = float(tl_model.predict(img_batch, verbose=0)[0][0])
            prob_pigeon_tl = 1.0 - prob_rooster_tl
            pred_label_tl = "Rooster" if prob_rooster_tl >= 0.5 else "Pigeon"
            conf_tl = max(prob_rooster_tl, prob_pigeon_tl) * 100
            summary_text += f"**Transfer Learning (.keras) Prediction:** {pred_label_tl} ({conf_tl:.2f}% confidence)"
            if model_choice == "Transfer Learning (MobileNetV2)" or len(results) == 0:
                results = {"Rooster": prob_rooster_tl, "Pigeon": prob_pigeon_tl}
        else:
            summary_text += "**Transfer Learning (MobileNetV2):** Weights not found on disk. Please upload transfer_learning_mobilenetv2.keras"

    return results, summary_text

readme_content = "No Readme.txt found in the Space repository."
if os.path.exists("Readme.txt"):
    with open("Readme.txt", "r", encoding="utf-8") as f:
        readme_content = f.read()

with gr.Blocks(title="Rooster vs Pigeon CNN Classification") as demo:
    gr.Markdown("# Rooster vs. Pigeon Image Classification")
    gr.Markdown("Permanent Deployment on Hugging Face Spaces.")

    with gr.Tabs():
        with gr.TabItem("1. Live Model Testing (Classifier)"):
            gr.Markdown("### Upload an Image of a Rooster or Pigeon to Classify")
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Input Image")
                    model_selector = gr.Radio(
                        choices=["Compare Both Models", "Custom Self-Designed CNN", "Transfer Learning (MobileNetV2)"],
                        value="Compare Both Models",
                        label="Select Model Architecture"
                    )
                    submit_btn = gr.Button("Classify Image", variant="primary")
                with gr.Column():
                    label_output = gr.Label(num_top_classes=2, label="Class Probabilities")
                    text_output = gr.Markdown(label="Detailed Model Output")

            submit_btn.click(
                fn=predict_image,
                inputs=[image_input, model_selector],
                outputs=[label_output, text_output]
            )

        with gr.TabItem("2. Performance & Visualizations"):
            gr.Markdown("## Network Architecture Diagrams & Training Curves")
            with gr.Row():
                if os.path.exists("custom_cnn_diagram.png"):
                    gr.Image("custom_cnn_diagram.png", label="Custom Self-Designed CNN Diagram")
                if os.path.exists("transfer_learning_diagram.png"):
                    gr.Image("transfer_learning_diagram.png", label="Transfer Learning (MobileNetV2) Diagram")
            with gr.Row():
                if os.path.exists("custom_cnn_metrics.png"):
                    gr.Image("custom_cnn_metrics.png", label="Custom CNN Training Metrics")
                if os.path.exists("tl_metrics.png"):
                    gr.Image("tl_metrics.png", label="Transfer Learning Training Metrics")

        with gr.TabItem("3. Readme.txt"):
            gr.Code(value=readme_content, language="markdown", label="Readme.txt")

if __name__ == '__main__':
    # Explicitly listen on port 7860 and interface 0.0.0.0 for Hugging Face Space stability
    demo.launch(server_name='0.0.0.0', server_port=7860)