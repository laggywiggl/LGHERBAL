import streamlit as st
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dense
import numpy as np
import os
import pandas as pd
from keras.models import Sequential
import nltk
import re
import emoji
import matplotlib.pyplot as plt
import time
from transformers import AutoTokenizer, AutoModel
import joblib
train_df = pd.read_csv('clean_data.csv')

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(train_df['clean_tweet'].values,
                                                                    pd.get_dummies(train_df['label']).values,
                                                                    test_size=0.2, random_state=42,shuffle=False)

# Tokenize and pad sequences for training data
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(train_data)

# Load the pre-trained model
loaded_model1 = load_model('Interface/lstm2.h5')
loaded_model2 = load_model('Interface/lstm.h5')


def replace_emoji(sent):
    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', sent)

def is_emoji(s):
    return s in UNICODE_EMOJI['en']
# add space near your emoji
def add_space(text):
    return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()

# Load the stop words from the file into a set
with open('stop_words.txt', 'r', encoding='utf-8') as file:
    stop_words = set(file.read().split())

def remove_stop_words(text):
    # Remove stop words
    return ' '.join([word for word in text.split() if word not in stop_words])


def remove_special_characters(text):
    # Define a pattern to match special characters (excluding alphanumeric, Arabic characters, and whitespace)
    pattern = re.compile('[^A-Za-z0-9\s\u0600-\u06FF]+', flags=re.UNICODE)

    # Use the pattern to replace special characters with an empty string
    clean_text = pattern.sub('', text)

    return clean_text
def preprocess_input(input_text):
    # Apply preprocessing steps
    text_without_emoji = replace_emoji(input_text)
    lowercase_text = text_without_emoji.lower()
    text_without_special_chars = remove_special_characters(lowercase_text)
    text_without_stop_words = remove_stop_words(text_without_special_chars)
    return text_without_stop_words

############################################OUR APP##########################################



# Title and sidebar
st.image(image='Logo.png', caption="Lgherbal", use_column_width=True)

st.title("Arabic Sentiment Analysis App")

st.sidebar.title("Our systems")
# Sidebar with two buttons
selected_page = st.sidebar.selectbox(
    "Select a System",
    ["Reviews/Comments sentiment analysis","Sentence sentiment analysis"],
    key="select a system",
    help="Choose a system according to your preference.",
    #index=0,
    format_func=lambda page: f"📄 {page}",
)
# Main content based on the selected page
if selected_page == "Reviews/Comments sentiment analysis":
    st.title("Reviews/Comments sentiment analysis")

    st.header('Put your file of reviews :blue[here] :point_down:')
    # File upload
    uploaded_file = st.file_uploader("Upload a file", type=['txt'])

    # Model selection
    selected_models = st.multiselect("Select models for predictions", ['Paid model', 'Free model'])

    # Function to calculate average probabilities and generate pie chart
    def generate_pie_chart(model_name, class_probabilities):
        data_for_pie_chart = {}

        for class_name, total_prob in class_probabilities.items():
            average_prob = total_prob / total_lines
            data_for_pie_chart[class_name] = average_prob

        # Create a pie chart with a similar style
        explode = [0.1] * len(data_for_pie_chart)
        colors = ['#66b3ff', '#99ff99', '#ffcc99']  # Choose your own colors

        fig, ax = plt.subplots()
        ax.pie(data_for_pie_chart.values(), labels=data_for_pie_chart.keys(), autopct='%1.1f%%',
            startangle=90, explode=explode, colors=colors, shadow=True)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Add a title
        plt.title(f"{model_name} Average Probabilities", fontsize=16, fontweight='bold')

        return fig

    if uploaded_file is not None:
        # Read the content of the file
        class_names = {0: "Positive", 1: "Neutral", 2: "Negative"}

        file_contents = uploaded_file.getvalue().decode("utf-8")
        #st.write(file_contents)

        # Accumulate probabilities for each class and each model
        total_probabilities = {model_name: {class_name: 0 for class_name in class_names.values()} for model_name in selected_models}
        total_lines = len(file_contents.split('\n'))
        
        lines = file_contents.split('\n')
        for i, line in enumerate(lines):
            #st.write(f"\nLine {i+1}: {line}")

            # Preprocess the current line
            preprocessed_line = preprocess_input(line)

            # Tokenize and pad sequences for the current line
            sequences = tokenizer.texts_to_sequences([preprocessed_line])
            padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

            for model_name in selected_models:
        

                if model_name == "Paid model":
                    predictions = loaded_model1.predict(np.array(padded_sequence))
                elif model_name == "Free model":
                    predictions = loaded_model2.predict(np.array(padded_sequence))

                # Accumulate probabilities for each class
                for j, prob in enumerate(predictions[0]):
                    total_probabilities[model_name][class_names[j]] += prob

        # Calculate average probabilities for each class and each model
        for model_name, class_probabilities in total_probabilities.items():
            st.subheader(f"{model_name} Average Probabilities:")
            for class_name, total_prob in class_probabilities.items():
                average_prob = total_prob / total_lines
                st.write(f"{class_name}: {average_prob * 100:.2f}%")



        # Calculate average probabilities for each class and each model
        for model_name, class_probabilities in total_probabilities.items():
            # Generate pie chart
            fig = generate_pie_chart(model_name, class_probabilities)

            # Display the pie chart using Streamlit
            st.pyplot(fig)

            # Create a flag to check if the download button is clicked
            button_key = f"{model_name}_download_button"
            download_clicked = st.button(f"Download {model_name} Pie Chart", key=button_key)

            # Check if the download button is clicked
            if download_clicked:
                # Save the figure to a file
                file_path = f"{model_name}_average_probabilities.png"
                plt.savefig(file_path, bbox_inches='tight')

                # Trigger the download
                #st.download_button(label=f"Download {model_name} Pie Chart", data=file_path, mime="image/png")
                
                st.success(f"Downloaded successfully: {model_name} Pie Chart")



if selected_page == "Sentence sentiment analysis":
    sentiment_labels = {0: "Positive", 1: "Neutral", 2: "Negative"}


    # Load the DarijaBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")
    model = AutoModel.from_pretrained("SI2M-Lab/DarijaBERT")

    # Load the Persistent Activation Classifier (PAC) model from the file
    pac_model = joblib.load('Interface/pac.sav')

    def predict_sentiment(sentence):
        # Example on a single sentence
        encoded_input = tokenizer(sentence, padding="max_length", max_length=128, truncation=True, return_tensors='pt')
        output = model(**encoded_input)
        sentence_embedding = output.last_hidden_state.detach().numpy().reshape(1, -1)

        # Use the loaded PAC model to get decision function scores
        decision_scores = pac_model.decision_function(sentence_embedding)

        # Apply softmax to obtain class probabilities
        class_probabilities = np.exp(decision_scores) / np.sum(np.exp(decision_scores), axis=1, keepdims=True)

        # Get the predicted class (index with the highest probability)
        predicted_class = np.argmax(class_probabilities)
        
        return predicted_class, class_probabilities

    # Streamlit app
    st.title("DarijaBERT Sentiment Analysis")

    user_input = st.text_area("Enter a sentence in Arabic:")

    if st.button("Predict"):
        if user_input:

            # Make a prediction
            predicted_class, class_probabilities = predict_sentiment(user_input)

            # Get the sentiment label based on the predicted class
            predicted_label = sentiment_labels[predicted_class]

            st.write(f"Predicted sentiment: {predicted_label}")
            st.write("Probabilities:")
            for i, prob in enumerate(class_probabilities[0]):
                st.write(f"{sentiment_labels[i]}: {prob * 100:.2f}%")
        else:
            st.warning("Please enter a sentence before predicting.")



