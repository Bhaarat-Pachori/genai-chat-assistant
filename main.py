import os
from functools import partial


import openai
import streamlit as st
from dotenv import load_dotenv


from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_core.messages import SystemMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Track the project through LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"

prompt = ChatPromptTemplate(
    [
        ("system", "{act_like_a}"),
        ("user", "{question}")
    ]
)


def get_response(context, key, temp, token, llm_model, question):
    openai.api_key = key
    llm = ChatOpenAI(model=llm_model, temperature=temp, max_completion_tokens=token)
    parser = StrOutputParser()
    chain = prompt|llm|parser
    answer = chain.invoke({"act_like_a":act_like_a, "question": question})
    return answer

# create streamlit ui
st.title("Smart Q&A assistant")

st.sidebar.title("Settings")
act_like_a = st.sidebar.text_input("Configure your assistant. Describe what you want to use the chatbot for?",
                      help="""E.g. Behave as a Python expert and\nanswer the coding questions a\nuser might have""", 
                      key="config",
                      value="Act like an uptight mom and respond to questions with pure sarcasm.")
key = st.sidebar.text_input("OPENAI API-KEY", type="password", key="api_key",)
temp = st.sidebar.slider("Temperature",min_value=0.0, max_value=1.0, value=0.6, key="temp")
token = st.sidebar.slider("Tokens length", min_value=50, max_value=300, value=150, key="token")

llm_model = st.sidebar.selectbox("Select LLM models to use", ["gpt-4o", "gpt-4-turbo", "gpt-4"], key="llm")

if st.session_state.get("config") and st.session_state.get("api_key") and st.session_state.get("token") and st.session_state.get("llm"):
    st.session_state.ready_to_chat = True

# main chat window
if st.session_state.get("ready_to_chat"):
    st.write("Ask you questions below")
    user_ip = st.text_input("User:")
    if user_ip:
        response = get_response(act_like_a, key, temp, token, llm_model, user_ip)
        st.markdown(f'<div style=padding: 10px; border-radius: 5px;">{response}</div>',
        unsafe_allow_html=True,)
