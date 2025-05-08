import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("generated_email_dataset.csv")  # Replace with the correct path
df["text"] = df["subject"] + " " + df["body"]
X = df["text"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization and training
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# Prediction logic
def classify_email(subject, body):
    text = subject + " " + body
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return "✅ Important - Open" if prediction == 1 else "🚫 Not Important - Ignore"

# Clear inputs function
def clear_inputs():
    return "", "", ""  # Clear both subject, body, and output

# UI
with gr.Blocks(theme=gr.themes.Base(primary_hue="blue", secondary_hue="green")) as iface:
    gr.Markdown(
        """
        <h1 style='text-align: center; color: #2e7d32;'>Automated Inbox Manager</h1>
        <p style='text-align: center;'>Simplify your email management with automated prioritization and sorting.</p>
        """,
        elem_id="header"
    )

    with gr.Row():
        with gr.Column():
            subject = gr.Textbox(label="Email Subject", placeholder="Type here")
            body = gr.Textbox(label="Email Body", lines=5, placeholder="Type here")
            btn_submit = gr.Button("Classify")
            btn_clear = gr.Button("Clear")
        with gr.Column():
            output = gr.Textbox(label="Classification Result", interactive=False)

    btn_submit.click(fn=classify_email, inputs=[subject, body], outputs=output)
    btn_clear.click(fn=clear_inputs, inputs=None, outputs=[subject, body, output])

iface.launch(share=True)