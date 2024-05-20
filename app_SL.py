import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the model, vectorizer, and label encoder
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# App title and description
st.title("Sentence Difficulty Prediction")
st.markdown("""
Welcome to the Sentence Difficulty Prediction app. 
This tool predicts the difficulty level of a given sentence using a pre-trained machine learning model.
""")

# Input sentence
sentence = st.text_input("Enter a sentence to predict its difficulty level:")

# Perform prediction
if sentence:
    X_tfidf = vectorizer.transform([sentence])
    prediction = model.predict(X_tfidf)
    difficulty = label_encoder.inverse_transform(prediction)[0]

    # Display the predicted difficulty
    st.subheader("Prediction Result")
    st.write(f"The predicted difficulty level for the sentence is: **{difficulty}**")

    # Sample data for illustration (replace with actual data if available)
    difficulty_levels = ['Easy', 'Medium', 'Hard']
    counts = [50, 30, 20]  # Example counts for each difficulty level

    # Display a pie chart of difficulty levels
    st.subheader("Difficulty Level Distribution")
    fig, ax = plt.subplots()
    ax.pie(counts, labels=difficulty_levels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107', '#F44336'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # Display ranking information
    total_sentences = sum(counts)
    user_position = counts[difficulty_levels.index(difficulty)]
    user_percentile = (user_position / total_sentences) * 100
    st.subheader("Your Ranking")
    st.write(f"You are in the top **{user_percentile:.1f}%** of all users based on sentence difficulty.")

# Add a sidebar with additional information
st.sidebar.title("About")
st.sidebar.info("""
This app uses a Logistic Regression model to predict the difficulty level of sentences.
The model was trained on a dataset of sentences labeled with difficulty levels.
""")
st.sidebar.title("Instructions")
st.sidebar.info("""
1. Enter a sentence in the text box.
2. The app will predict the difficulty level of the sentence.
3. View the distribution of difficulty levels and see where you rank.
""")

# Footer
st.markdown("""
---
Developed by Igor Dallemagne & Alex Pinel**.
""")
