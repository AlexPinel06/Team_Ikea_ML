import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np
import random

# Load the model, vectorizer, and label encoder
model = joblib.load('model2/logistic_regression_model.pkl')
vectorizer = joblib.load('model2/tfidf_vectorizer.pkl')
label_encoder = joblib.load('model2/label_encoder.pkl')

# Function to reset the input
def reset_input():
    st.session_state["scores"] = []
    st.session_state["random_questions"] = []

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

if "random_questions" not in st.session_state:
    st.session_state["random_questions"] = []

# Questions and correct answers
given_sentences = [
    "Je suis étudiant.",
    "Il fait beau aujourd'hui.",
    "La théorie de la relativité est complexe.",
    "Les ordinateurs quantiques sont l'avenir.",
    "Il a couru un marathon hier.",
    "Elle aime les animaux.",
    "Ils ont déménagé en France.",
    "Nous allons à la plage.",
    "Je préfère le café au thé.",
    "Le réchauffement climatique est une menace."
]
correct_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'A1', 'A2', 'B1', 'B2', 'C1']
difficulty_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

# Example sentences for each level
example_sentences = {
    'A1': "Je m'appelle Marie.",
    'A2': "J'aime le chocolat.",
    'B1': "Je vais à l'université en bus.",
    'B2': "La technologie moderne révolutionne notre quotidien.",
    'C1': "L'innovation est essentielle pour la compétitivité des entreprises.",
    'C2': "La théorie des cordes est une perspective fascinante de la physique moderne."
}

# Randomize questions if not already randomized
if not st.session_state["random_questions"]:
    combined = list(zip(given_sentences, correct_levels))
    random.shuffle(combined)
    st.session_state["random_questions"] = combined[:6]  # Select 6 random questions

random_questions = st.session_state["random_questions"]
given_sentences = [q[0] for q in random_questions]
correct_levels = [q[1] for q in random_questions]

# User answers
user_answers = []
correct_answers = correct_levels + correct_levels[:6]

# First 6 questions: Select difficulty level for given sentences
st.subheader("Select the difficulty level for the given sentences:")
for i, sentence in enumerate(given_sentences):
    user_answers.append(st.selectbox(f"Sentence {i+1}: {sentence}", options=difficulty_levels))

# Last 6 questions: Write sentences for given difficulty levels
st.subheader("Write a sentence for each given difficulty level:")
user_written_sentences = []
for level in difficulty_levels[:6]:
    user_sentence = st.text_input(f"Write a sentence for level {level}:")
    user_written_sentences.append(user_sentence)
    if user_sentence:
        X_tfidf = vectorizer.transform([user_sentence])
        prediction = model.predict(X_tfidf)
        predicted_level = label_encoder.inverse_transform(prediction)[0]
        user_answers.append(predicted_level)

# Calculate score
if st.button("Submit"):
    score = calculate_score(user_answers, correct_answers)
    update_scores(score)
    st.write(f"Your score is: {score}/12")
    
    st.subheader("Your Answers")
    for i, (user_answer, correct_answer) in enumerate(zip(user_answers[:6], correct_answers[:6])):
        if user_answer == correct_answer:
            st.write(f"Question {i+1}: Correct! {user_answer} ✅")
        else:
            st.write(f"Question {i+1}: Incorrect! {user_answer} ❌ (Correct: {correct_answer})")

    st.subheader("Analysis of Your Written Sentences")
    for i, (user_sentence, level) in enumerate(zip(user_written_sentences, difficulty_levels[:6])):
        if user_sentence:
            X_tfidf = vectorizer.transform([user_sentence])
            prediction = model.predict(X_tfidf)
            predicted_level = label_encoder.inverse_transform(prediction)[0]
            example_sentence = example_sentences[level]
            st.write(f"Sentence {i+7}: Predicted Level: {predicted_level}")
            st.write(f"Example sentence for level {level}: {example_sentence}")

# Button to reset the input
if st.button("Reset"):
    st.session_state.clear()
    st.experimental_rerun()

# Display the last three scores
st.sidebar.title("Your Last 3 Scores")
for score in st.session_state["scores"]:
    st.sidebar.write(f"Score: {score}/12")

# Add a sidebar with additional information
st.sidebar.title("About")
st.sidebar.info("""
This app uses a pre-trained model to predict the difficulty level of sentences.
The model was trained on a dataset of sentences labeled with difficulty levels.
""")
st.sidebar.title("Instructions")
st.sidebar.info("""
1. Select the difficulty level for each of the first 6 sentences.
2. Write a sentence for each of the last 6 difficulty levels.
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
