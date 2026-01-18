import streamlit as st
import requests
import json

st.title(" Recipe Generator Chatbot")
st.write("Enter ingredients you have, and I'll suggest recipes!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if ingredients := st.chat_input("What ingredients do you have? (e.g., eggs, onions, tomatoes)"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": ingredients})
    with st.chat_message("user"):
        st.markdown(ingredients)
    
    # Get recipe from backend
    with st.chat_message("assistant"):
        with st.spinner("Finding recipes..."):
            try:
                response = requests.post(
                    "http://localhost:8000/recipe",
                    json={"ingredients": ingredients}
                )
                if response.status_code == 200:
                    recipe = response.json()["recipe"]
                    st.markdown(recipe)
                    st.session_state.messages.append({"role": "assistant", "content": recipe})
                else:
                    st.error("Failed to get recipe")
            except:
                st.error("Backend server not running. Start it with: uvicorn main:app --reload")

# Sidebar info
with st.sidebar:
    st.header("How to Use")
    st.write("1. Enter ingredients (comma-separated)")
    st.write("2. Press Enter")
    st.write("3. Get recipe suggestions!")
    st.divider()
    st.write("**Example inputs:**")
    st.write("- eggs, onions, cheese")
    st.write("- chicken, rice, vegetables")
    st.write("- bananas, chocolate, milk")
