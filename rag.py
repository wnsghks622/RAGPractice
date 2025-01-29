# to load all PDF files in a directory
from langchain_community.document_loaders import PyPDFDirectoryLoader
# For splitting the text documents in the directory
from langchain.text_splitter import RecursiveCharacterTextSplitter
# https://huggingface.co/BAAI/bge-base-en-v1.5 - seems to be the best embedding model as of 1/22/2024
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# Chroma is an easy to use vector database to store embeddings that don't require authentication
from langchain_community.vectorstores import Chroma

def load_docs():
    loader = PyPDFDirectoryLoader("PDF_docs/")
    knowledge_base = loader.load()
    return knowledge_base

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                  chunk_overlap=100,
                  add_start_index=True, 
                  separators = ['\n', '\n\n', " ", ""]) # keep all paragraphs (and then sentences, and then words) together as long as possible
    chunked_docs = splitter.split_documents(docs)
    return chunked_docs

# Load the embeddings model
BGEembeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5", 
                            model_kwargs={"device": "cpu"}, 
                            encode_kwargs={"normalize_embeddings": True}) 

def embed_and_index(chunked_docs):
    # Load the vector store and index chunks
    chroma = Chroma.from_documents(documents=chunked_docs,
                                    embedding=BGEembeddings)
        
    return chroma

