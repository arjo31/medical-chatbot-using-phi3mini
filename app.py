import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
from src.model_path import *
from src.helper import *
import os

load_dotenv()

data_path = r"D:\My_Coding_Files\medical-chatbot-using-llama2\data"
vector_store_path = r"D:\My_Coding_Files\medical-chatbot-using-llama2\faiss_db"

embeddings = hugging_face_embedding_model()

if os.path.exists(vector_store_path):
    vector_db = load_db(vector_store_path, embedding=embeddings)
else:
    document = load_pdf(data_path=data_path)
    doc_chunks = text_splitter(documents=document)
    vector_db = vector_store(doc_chunks, embeddings)
    save_db(vector_db=vector_db, path=vector_store_path)

model_path = model_path
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
chain_type_kwargs = {"prompt" : prompt}

llm = CTransformers(model=model_path, model_type='llama2', config={'max_new_tokens' : 512, 'temperature' : 0.8})

qa = RetrievalQA.from_chain_type(llm = llm, chain_type_kwargs = chain_type_kwargs, chain_type='stuff', return_source_documents = True, retriever = vector_db.as_retriever(search_kwargs={'k' : 2}))

st.title('Medical Chatbot')

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def add_message(message, is_user = True):
    st.session_state.chat_history.append({'message' : message, 'is_user' : is_user})

user_input = st.text_input('You : ', key='input')

if (st.button("Send")):
    if user_input:
        print(user_input)
        add_message(user_input, is_user=True)
        answer = qa({'query' : user_input})
        print(answer['result'])
        add_message(answer['result'], is_user=False)
    
for entry in st.session_state.chat_history:
    if entry['is_user']:
        st.write(f"You :  {entry['message']}")
    else:
        st.write(f"Bot : {entry['message']}")