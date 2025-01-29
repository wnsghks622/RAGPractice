# to load all PDF files in a directory
from langchain_community.document_loaders import PyPDFDirectoryLoader
# For splitting the text documents in the directory
from langchain.text_splitter import RecursiveCharacterTextSplitter
# https://huggingface.co/BAAI/bge-base-en-v1.5 - seems to be the best embedding model as of 1/22/2024
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# Chroma is an easy to use vector database to store embeddings that don't require authentication
from langchain_community.vectorstores import Chroma
# For contextual compression retriever
# Alternative compressors include LLMChainExtractor and LLMFilter, but both require an LLM so more $ 
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

# OpenAI models
from langchain_openai import ChatOpenAI
# HuggingFace Hub models
from langchain_community.llms import HuggingFaceHub
# prompt templates
from langchain.prompts import ChatPromptTemplate
# for building a simple chain
from langchain.chains import LLMChain
# Gemini
from langchain_google_vertexai import ChatVertexAI

import os
import streamlit as st

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['GOOGLE_API_KEY'] = st.secrets["GEMINI_API_KEY"]

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

def init_retriever(rerank: bool):
    docs = load_docs()
    chunked_docs = chunk_docs(docs)
    vectorstore = embed_and_index(chunked_docs)
    retriever = vectorstore.as_retriever(search_type="similarity",
                                          search_kwargs={"k": 3})
    if rerank:
        embeddings_filter = EmbeddingsFilter(embeddings=BGEembeddings, 
                                             similarity_threshold=0.7)
        
        compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, 
                                                               base_retriever=retriever)
        return compression_retriever
         
    return retrieverp

def build_rag_chain(provider: str):
    
    # initialize LLM
    if provider == 'openai':
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', # alternatively use gpt-4
                         temperature=0.4,
                         max_tokens=512)
        
    elif provider == "huggingface":
        llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", # experiment with different models - https://huggingface.co/models?sort=trending
                             task="text-generation", # must be text-generation or text2text-generation
                             model_kwargs={
                                "max_new_tokens": 512,
                                "top_k": 30,
                                "temperature": 0.1,
                                "repetition_penalty": 1.03,
                             },
                             )
    elif provider == "gemini":
        llm = ChatVertexAI(model="gemini-1.5-flash")
            
    # design prompt
    ## the system prompt will tell the LLM how to act and how to use the context retrieved
    ## the human template will hold the chat history and last user input
    
    rag_chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{sys_prompt}"), 
            ("human", "{chat_history}"),
        ]
    )
    
    # Build chain
    rag_chain = LLMChain(llm=llm, prompt=rag_chat_prompt)
    
    return rag_chain

def run_rag_chain(chat_messages,
                  user_query):
    # initialize the RAG chain
    rag_chain = build_rag_chain(provider = "gemini")
    
    # initialize the retriever and get context
    retriever = init_retriever(rerank=False)
    context_docs = retriever.get_relevant_documents(user_query)
    context = format_docs(context_docs)
    
    chat_history = build_chat_history(chat_messages)
    SYS_PROMPT = f"""Use the following pieces of context to answer the users question. 
    Maintain a conversational tone and try to be as helpful as possible. 
    Keep the chat history into account and always cite your source as the name of the file.
    
    Chat History:
    {chat_history}
    
    Retrieved Context:
    {context}
    
    Sources:
    {[doc.metadata["source"] for doc in context_docs]}
    
    User Query:
    {user_query}
    
    Answer:
    
    """
    
    # run the chain
    prompt = {"sys_prompt": SYS_PROMPT,
              "chat_history": chat_history}
    
    response = rag_chain(prompt)['text']
    
    # return the output
    return response

