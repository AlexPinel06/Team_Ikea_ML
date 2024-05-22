import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the model, vectorizer, and label encoder
model = joblib.load('model2/logistic_regression_model.pkl')
vectorizer = joblib.load('model2/tfidf_vectorizer.pkl')
label_encoder = joblib.load('model2/label_encoder.pkl')

# Function to reset the input
def reset_input():
    st.session_state["scores"] = []

# Function to calculate the score
def calculate_score(user_answers, correct_answers):
    return sum(1 for user, correct in zip(user_answers, correct_answers) if user == correct)

# Function to update scores
def update_scores(new_score):
    if "scores" not in st.session_state:
        st.session_state["scores"] = []
    st.session_state["scores"].append(new_score)
    if len(st.session_state["scores"]) > 3:
        st.session_state["scores"] = st.session_state["scores"][-3:]

# Initialize session state
if "scores" not in st.session_state:
    st.session_state["scores"] = []

# App title and description
st.title("Sentence Difficulty Prediction")
st.markdown("""
Welcome to the Sentence Difficulty Prediction app. 
This tool predicts the difficulty level of a given French sentence using a pre-trained model.
""")

# Display the last three scores
st.sidebar.title("Your Last 3 Scores")
for score in st.session_state["scores"]:
    st.sidebar.write(f"Score: {score}/10")

# Questions and answers (for simplicity, using predefined questions and correct answers)
given_sentences = [
    "Je suis étudiant.",
    "Il fait beau aujourd'hui.",
    "La théorie de la relativité est complexe.",
    "Les ordinateurs quantiques sont l'avenir.",
    "Il a couru un marathon hier."
]
correct_levels = ['A1', 'A2', 'B1', 'B2', 'C1']
difficulty_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

# User answers
user_answers = []
correct_answers = correct_levels + correct_levels[:5]

# First 5 questions: Select difficulty level for given sentences
st.subheader("Select the difficulty level for the given sentences:")
for i, sentence in enumerate(given_sentences):
    user_answers.append(st.selectbox(f"Sentence {i+1}: {sentence}", options=difficulty_levels))

# Last 5 questions: Write sentences for given difficulty levels
st.subheader("Write a sentence for each given difficulty level:")
for level in difficulty_levels[:5]:
    user_sentence = st.text_input(f"Write a sentence for level {level}:")
    if user_sentence:
        X_tfidf = vectorizer.transform([user_sentence])
        prediction = model.predict(X_tfidf)
        predicted_level = label_encoder.inverse_transform(prediction)[0]
        user_answers.append(predicted_level)

# Calculate score
if st.button("Submit"):
    score = calculate_score(user_answers, correct_answers)
    update_scores(score)
    st.write(f"Your score is: {score}/10")
    
    st.subheader("Your Answers")
    for i, (user_answer, correct_answer) in enumerate(zip(user_answers, correct_answers)):
        if user_answer == correct_answer:
            st.write(f"Question {i+1}: Correct! {user_answer} ✅")
        else:
            st.write(f"Question {i+1}: Incorrect! {user_answer} ❌ (Correct: {correct_answer})")

# Button to reset the input
if st.button("Reset"):
    st.session_state.clear()
    st.experimental_rerun()

# Add a sidebar with additional information
st.sidebar.title("About")
st.sidebar.info("""
This app uses a pre-trained model to predict the difficulty level of sentences.
The model was trained on a dataset of sentences labeled with difficulty levels.
""")
st.sidebar.title("Instructions")
st.sidebar.info("""
1. Select the difficulty level for each of the first 5 sentences.
2. Write a sentence for each of the last 5 difficulty levels.
3. Submit your answers to see your score.
""")

# Add extra information in the sidebar
st.sidebar.title("Fun Facts")
st.sidebar.info("""
- A1 is the beginner level, while C2 is the mastery level.
- Difficulty prediction can help in language learning by tailoring content to your level.
- Natural Language Processing (NLP) techniques are used to analyze and understand human language.
- The model is a state-of-the-art model for sequence classification tasks.
""")

# Footer
st.markdown("""
---
Developed by Igor Dallemagne and Alex Pinel.
""")
