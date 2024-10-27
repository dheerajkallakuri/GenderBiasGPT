import requests
import streamlit as st

def model_reply(question, language):
    # Send the user question to the Flask backend API
    url = "https://genderbiasgpt.onrender.com/api/message"  # Flask API endpoint
    data = {
        'message': question,
        'language': language
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
        
        result = response.json()
        
        return result['response']

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("GenderBiasGPT")

# Language selection dropdown
language = st.selectbox("Select Language", options=["Select Language","Vietnamese 🇻🇳", "Hindi 🇮🇳"], index=0)
if language == 'Select Language':
    st.write("Please select a language.")

# User question input
user_question = st.text_input("Enter your message")
flags=["🇻🇳","🇮🇳"]


# Button to trigger user question analysis
if st.button("Submit Question"):
    if user_question.strip() != "":
        with st.spinner("Fetching the response..."):
            response = model_reply(user_question, language[:-3])
            if isinstance(response, dict):  # Check if the response is a dictionary
                # Add the user input and response to the session state messages if they don't already exist
                if not any(msg["content"] == user_question and msg["role"] == "user" for msg in st.session_state.messages):
                    st.session_state.messages.append({"role": "user", "content": user_question})

                if not any(msg["content"] == response and msg["role"] == "assistant" for msg in st.session_state.messages):
                    # Format the response for consistency
                    response_formatted = (
                        f"- {language[len(language)-2:]}**Translation**: {response.get('translated', 'No translation found')}\n" 
                        f"- **Female Bias Value**: {response.get('f_val', 'N/A')}/1\n" 
                        f"- **Male Bias Value**: {response.get('m_val', 'N/A')}/1\n" 
                    )

                    st.session_state.messages.append({"role": "assistant", "content": response_formatted})
            else:
                # Handle the case where response is not as expected (i.e., not a dictionary)
                error_message = "Error: Expected a structured response but received a string."
                print(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
# Keep the chat visible all the time
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").markdown(f"**You:** {message['content']}")
    else:
        st.chat_message("assistant").markdown(f"**Bot:** {message['content']}")

# Add an "About" section at the bottom of the app
st.markdown("---")
st.markdown("### About")
st.markdown("Checking the Gender Bias in Regional Langugaes")
