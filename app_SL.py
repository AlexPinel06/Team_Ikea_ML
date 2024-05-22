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
    st.session_state["sentence"] = ""

# App title and description
st.title("Sentence Difficulty Prediction")
st.markdown("""
Welcome to the Sentence Difficulty Prediction app. 
This tool predicts the difficulty level of a given French sentence using a pre-trained Logistic Regression model.
""")

# Check if 'sentence' exists in session_state
if "sentence" not in st.session_state:
    st.session_state["sentence"] = ""

# Create a form for the sentence input
with st.form(key='sentence_form'):
    sentence = st.text_input("Enter a sentence to predict its difficulty level:", key="sentence")
    submit_button = st.form_submit_button(label='Submit')

# Button to reset the input
if st.button("Reset"):
    st.session_state.clear()
    st.experimental_rerun()

# Perform prediction
if submit_button and st.session_state["sentence"]:
    sentence = st.session_state["sentence"]
    X_tfidf = vectorizer.transform([sentence])
    prediction = model.predict(X_tfidf)
    difficulty = label_encoder.inverse_transform(prediction)[0]

    # Display the predicted difficulty
    st.subheader("Prediction Result")
    st.write(f"The predicted difficulty level for the sentence is: **{difficulty}**")

    # Estimated data for illustration
    difficulty_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    counts = [30, 25, 20, 15, 7, 3]  # Estimated percentage for each difficulty level

    # Display a pie chart of difficulty levels
    st.subheader("Estimated Distribution of French Learners by Competency Level")
    fig, ax = plt.subplots(facecolor='#0e1117')  # Set the background color of the figure
    wedges, texts, autotexts = ax.pie(counts, labels=difficulty_levels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Customize the appearance
    for text in texts:
        text.set_color('white')
    for autotext in autotexts:
        autotext.set_color('white')
    fig.patch.set_facecolor('#0e1117')  # Set the background color of the plot area

    st.pyplot(fig)

    # Calculate the ranking
    difficulty_index = difficulty_levels.index(difficulty)
    better_than_percentage = sum(counts[:difficulty_index])
    top_percent = 100 - better_than_percentage
    st.subheader("Your Ranking")
    st.write(f"You belong to the top **{top_percent}%** of all users with this difficulty level.")

    # Fun facts for each level
    fun_facts = {
        'A1': "Fun fact: Even Einstein had to start somewhere!",
        'A2': "Fun fact: You're now better than most tourists!",
        'B1': "Fun fact: You're officially conversational!",
        'B2': "Fun fact: You can enjoy French movies without subtitles!",
        'C1': "Fun fact: Your French is better than most expats!",
        'C2': "Fun fact: You're at the mastery level, like a true Parisian!"
    }
    st.write(fun_facts[difficulty])

    # Add source information
    st.markdown("""
    **Sources:**
    - Portal (CECR)
    - French Together – Learn French
    - Service-Public
    - Kwiziq French
    - FluentU
    **Note:** These percentages are based on general estimates and information available on the distribution of language proficiency levels according to the CEFR in various educational and linguistic sources.
    """)

    # YouTube videos for each level
    youtube_videos = {
        'A1': "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Example link, replace with a funny video
        'A2': "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Example link, replace with a funny video
        'B1': "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Example link, replace with a funny video
        'B2': "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Example link, replace with a funny video
        'C1': "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Example link, replace with a funny video
        'C2': "https://www.youtube.com/watch?v=dQw4w9WgXcQ"   # Example link, replace with a funny video
    }

    st.markdown("""
    ---
    For a comparison, here's an English speaker at your level:
    """)

    st.video(youtube_videos[difficulty])

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

# Add extra information in the sidebar
st.sidebar.title("Fun Facts")
st.sidebar.info("""
- A1 is the beginner level, while C2 is the mastery level.
- Difficulty prediction can help in language learning by tailoring content to your level.
- Natural Language Processing (NLP) techniques are used to analyze and understand human language.
- Logistic Regression is a simple yet powerful model for classification tasks.
""")

# Footer
st.markdown("""
---
Developed by Igor Dallemagne and Alex Pinel.
""")
