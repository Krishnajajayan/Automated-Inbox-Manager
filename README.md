#  Automated Inbox Manager

**Automated Inbox Manager** is a lightweight machine learning-based application that classifies emails as either **Important** or **Not Important**, helping users prioritize their inbox more efficiently.

It uses a trained `RandomForestClassifier` on TF-IDF features extracted from email subject and body text. The UI is built with **Gradio**, offering an easy and intuitive web interface to interact with the model.

---

##  Objective

> To simplify email management using machine learning by automatically identifying and highlighting important emails.

---

##  How It Works

1. **Dataset Preparation**:
   - A labeled dataset `generated_email_dataset.csv` is used, which contains:
     - `subject` – the email subject line
     - `body` – the content of the email
     - `label` – binary classification label (1 = Important, 0 = Not Important)

2. **Text Preprocessing**:
   - The `subject` and `body` are concatenated into a single text field.
   - A TF-IDF vectorizer transforms the combined text into numerical features.

3. **Model Training**:
   - A `RandomForestClassifier` is trained on the vectorized email texts to predict importance.

4. **Prediction**:
   - The model predicts if a new email is important based on the subject and body entered in the UI.

---

##  User Interface (Gradio)

### Fields:
- **Email Subject** – Text input for the subject line.
- **Email Body** – Multiline input for the body of the email.
- **Classification Result** – Shows the prediction:
  - ✅ Important – Open
  - 🚫 Not Important – Ignore

---

##  Sample Emails to Try

| Subject | Body | Expected Output |
|--------|------|-----------------|
| `Your Amazon order has shipped` | `Track your package for expected delivery on Friday.` | ✅ Important – Open |
| `Win a brand new iPhone now!` | `Just click this link to claim your prize.` | 🚫 Not Important – Ignore |
| `Project Meeting Today at 3 PM` | `Please join the Zoom link provided.` | ✅ Important – Open |
| `Flash Sale: 90% OFF only for 3 hours!` | `Unbelievable discounts just for you.` | 🚫 Not Important – Ignore |

---



