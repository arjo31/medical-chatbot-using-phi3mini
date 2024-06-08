import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from src.helper import *
from src.model_path import *
from src.prompt import *

load_dotenv()

data_path = r"data"
vector_store_path = r"faiss_db"

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return hugging_face_embedding_model()

embeddings = load_embedding_model()

@st.cache_resource(show_spinner=False)
def load_and_store_data():
    if os.path.exists(vector_store_path):
        vector_db = load_db(path=vector_store_path, embedding=embeddings)
    else:
        document = load_pdf(data_path=data_path)
        doc_chunks = text_splitter(documents=document)
        vector_db = vector_store(text_chunks=doc_chunks, embedding=embeddings)
        save_db(vector_db=vector_db, path=vector_store_path)
    return vector_db

vector_db = load_and_store_data()
model_path = model_path
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": prompt}

llm = Ollama(
    model="phi3:mini",
    temperature=0.8,
    num_predict=1024,
    top_k=90,
    top_p=0.95,
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type_kwargs=chain_type_kwargs,
    chain_type="stuff",
    return_source_documents=True,
    retriever=vector_db.as_retriever(search_kwargs={"k": 2}),
)

col1, col2 = st.columns([4, 1])

with col1:
    st.header('Medical Chatbot')

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{'role':'ai', 'content':'Hello! How may I assist you today?'}]

st.markdown(
    """
    <style>
    .centered-button {
        display: flex;
        align-items: center;
        justify-content: center;
        position: absolute;
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with col2:
    with st.container():
        st.write('<div class="centered-button">', unsafe_allow_html=True)
        if st.button('Clear Chat', type='primary'):
            st.session_state.chat_history = [{'role':'ai', 'content':'Hello! How may I assist you today?'}]
        st.write('</div>', unsafe_allow_html=True)

def add_message(role, message):
    st.session_state.chat_history.append({'role': role, "content": message})

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.write(message['content'])

if user_input:= st.chat_input("Ask any Question!"):
    add_message('user', user_input)
    with st.chat_message('user'):
        st.write(user_input)
    if vector_db:
        response = qa({'query' : user_input})
        ans = response['result']
        add_message('ai', ans)
        with st.chat_message('ai'):
            st.write(ans)