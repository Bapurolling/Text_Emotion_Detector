import streamlit as st
import torch
import torch.nn as nn
import re
from transformers import BertTokenizer, BertModel
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Define the CustomBERTModel class
class CustomBERTModel(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        x = self.dropout(pooled_output)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.linear2(x)
        return logits

# Initialize the model
model = CustomBERTModel('bert-base-uncased', num_labels=6)

# Load the model with error handling
try:
    model.load_state_dict(torch.load("bert_emotion_model.pth", map_location=torch.device('cpu')))
    model.eval()
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    st.error(f"Failed to load the model: {e}")
    # Optionally exit or handle the error

# Load the tokenizer from the local directory
try:
    tokenizer = BertTokenizer.from_pretrained("./")
except Exception as e:
    logging.error(f"Error loading tokenizer: {e}")
    st.error(f"Failed to load the tokenizer: {e}")
    # Optionally exit or handle the error

# Function to clean input text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Define the function to predict emotion
def predict_emotion(text, model, tokenizer):
    cleaned_text = clean_text(text)

    if len(cleaned_text) < 5:
        return "Please enter a meaningful sentence."

    inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {key: val for key, val in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs
            probabilities = torch.nn.functional.softmax(logits, dim=1)
    except Exception as e:
        logging.error(f"Error during model prediction: {e}")
        return "Error during prediction."

    predicted_label = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, predicted_label].item()

    confidence_threshold = 0.6
    if confidence < confidence_threshold:
        return "Confidence too low. Please enter a clearer sentence."

    label_to_emotion = {0: "sadness", 1: "anger", 2: "love", 3: "surprise", 4: "fear", 5: "joy"}
    predicted_emotion = label_to_emotion[predicted_label]

    return predicted_emotion

# Streamlit UI
st.set_page_config(page_title="Emotion Detection", page_icon="😊")
st.title("Text Emotion Detector")
st.markdown("Enter a sentence below to predict the emotion!")

st.subheader("Emotions Detected:")
emotion_emoji = {
    "sadness": "😢",
    "anger": "😡",
    "love": "❤️",
    "surprise": "😮",
    "fear": "😨",
    "joy": "😊"
}
emotion_list = ", ".join([f"{emotion}: {emoji}" for emotion, emoji in emotion_emoji.items()])
st.markdown(emotion_list)

text = st.text_input("Enter a sentence:")
if st.button("Predict Emotion"):
    if text:
        emotion = predict_emotion(text, model, tokenizer)
        if emotion == "Please enter a meaningful sentence." or emotion == "Confidence too low. Please enter a clearer sentence.":
            st.warning(emotion)
        else:
            st.success(f"The predicted emotion is: {emotion} {emotion_emoji[emotion]}")
    else:
        st.warning("Please enter a sentence!")
