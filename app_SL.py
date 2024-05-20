import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the model, vectorizer, and label encoder
model = joblib.load('model2/logistic_regression_model.pkl')
vectorizer = joblib.load('model2/tfidf_vectorizer.pkl')
label_encoder = joblib.load('model2/label_encoder.pkl')

# App title and description
st.title("Prédiction de la Difficulté des Phrases")
st.markdown("""
Bienvenue sur l'application de prédiction de la difficulté des phrases.
Cet outil prédit le niveau de difficulté d'une phrase donnée en utilisant un modèle d'apprentissage automatique pré-entraîné.
""")

# Input sentence
sentence = st.text_input("Entrez une phrase pour prédire son niveau de difficulté :")

# Perform prediction
if sentence:
    X_tfidf = vectorizer.transform([sentence])
    prediction = model.predict(X_tfidf)
    difficulty = label_encoder.inverse_transform(prediction)[0]

    # Display the predicted difficulty
    st.subheader("Résultat de la Prédiction")
    st.write(f"Le niveau de difficulté prédit pour la phrase est : **{difficulty}**")

    # Sample data for illustration (replace with actual data if available)
    difficulty_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    counts = [15, 20, 25, 20, 10, 10]  # Example counts for each difficulty level

    # Display a pie chart of difficulty levels
    st.subheader("Répartition des Niveaux de Difficulté")
    fig, ax = plt.subplots()
    ax.pie(counts, labels=difficulty_levels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # Display ranking information
    total_sentences = sum(counts)
    user_count = counts[difficulty_levels.index(difficulty)]
    user_percentile = (user_count / total_sentences) * 100
    st.subheader("Votre Classement")
    st.write(f"Vous faites partie des **{user_percentile:.1f}%** des utilisateurs ayant ce niveau de difficulté.")

# Add a sidebar with additional information
st.sidebar.title("À Propos")
st.sidebar.info("""
Cette application utilise un modèle de régression logistique pour prédire le niveau de difficulté des phrases.
Le modèle a été entraîné sur un ensemble de données de phrases étiquetées par niveaux de difficulté.
""")
st.sidebar.title("Instructions")
st.sidebar.info("""
1. Entrez une phrase dans la zone de texte.
2. L'application prédira le niveau de difficulté de la phrase.
3. Consultez la répartition des niveaux de difficulté et voyez où vous vous situez.
""")

# Footer
st.markdown("""
---
Développé par **[Votre Nom]**.
""")
