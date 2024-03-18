import openai
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from pandasai import SmartDataframe
import pandas as pd
import os

st.title("🚀📊TabularConverse: Chat with Your CSV Data")

st.write("TabularConverse allows you to chat with your CSV data! 🚀 Upload your CSV files and start conversing to explore insights and gain deeper understanding effortlessly. 💬💡 It's your intuitive interface for interactive data analysis!")

st.markdown("""---""")
# Set up LangChain Google Generative AI
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

headers ={
    'GOOGLE_API_KEY' : st.secrets["GOOGLE_API_KEY"]
}

llm = ChatGoogleGenerativeAI(model="gemini-pro")

with st.sidebar:
    st.title('🤖TabularConverse')
    data = st.file_uploader("Upload a CSV file", type="csv")
    if data is not None:
        st.success('CSV file loaded Successfully!', icon='✅')
        data = pd.read_csv(data)
        df = SmartDataframe(data, config={"llm": llm})

# Chat loop
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if data is not None:
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Get responses from LangChain Google Generative AI
            responses = df.chat(prompt)
            if isinstance(responses, list):
                for response in responses:
                    full_response += str(response)  # Convert to string before concatenating
                    message_placeholder.markdown(full_response + "▌")
            else:
                full_response += str(responses)  # Convert to string before concatenating
                message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


# import pandas as pd
# data = pd.read_csv("reviews.csv")
# df = SmartDataframe(data, config={"llm": llm})
# ans = df.chat(query)

