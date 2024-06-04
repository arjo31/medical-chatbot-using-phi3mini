from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# Extract data from the PDF
def load_pdf(data_path):
    loader = DirectoryLoader(path=data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


# Create Text chunks
def text_splitter(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    docs = text_splitter.split_documents(documents)
    return docs


# Create embedding models
def hugging_face_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def ollama_embedding_model():
    embeddings = OllamaEmbeddings(model="llama2")
    return embeddings


# Store the text chunks in FAISS Vector DB
def vector_store(text_chunks, embedding):
    vector_db = FAISS.from_documents(documents=text_chunks, embedding=embedding)
    return vector_db


# save the vector to disk
def save_db(vector_db, path):
    vector_db.save_local(path)


# Load the database from disk
def load_db(path, embedding):
    return FAISS.load_local(
        path, embeddings=embedding, allow_dangerous_deserialization=True
    )
