#Importing Librariesm

import json
import os 
import sys
import boto3
import langchain_community
from langchain_aws import BedrockEmbeddings
from langchain_community.llms import Bedrock

import numpy as np 
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import BedrockChat
from langchain.chains import RetrievalQA





#Bedrock Clients 

bedrock = boto3.client(service_name="bedrock-runtime",region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock) 
#Data Alimentation
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    #Text Splitter with PDF data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                    chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

#Vecor_Embeddings and Storing in VS 
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_claude_3_Sonnet_llm():
    llm = BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=bedrock,
        model_kwargs={'max_tokens': 512}  # note: use snake_case here
    )
    return llm


def get_Llama_3_70B_Instruct_llm():
    llm=Bedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock,
                model_kwargs={'max_gen_len':512})
    return llm

from langchain_core.prompts import ChatPromptTemplate

# Build a ChatPromptTemplate instead of a single string
prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']



import streamlit as st 

def main():
    st.set_page_config(page_title="Enter the Chamber and Ask ", page_icon="🚪", layout="wide")
    st.title("💬 Chat with VacuMind by Binar 🌀🤖")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    # Sidebar for vector store operations
    with st.sidebar:
        st.header("Knowledge Vacuum ⚡")
        if st.button("📡 Recalibrate Intelligence System"):
            with st.spinner("Processing PDFs..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("✅ Vector store updated!")

    # --- Custom CSS for model logo buttons ---
    st.markdown("""
    <style>
    .model-container {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin: 20px 0;
    }
    .model-btn {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100px;
        height: 100px;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
        border: none;
    }
    .model-btn:hover {
        transform: scale(1.08);
        box-shadow: 0px 6px 14px rgba(0,0,0,0.25);
    }
    .claude {
        background-color: #FF6B6B;
        color: white;
    }
    .llama {
        background-color: #FFD93D;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Model selection ---
    st.subheader("Select Your Intelligence Core 🧠🧲")

    # Create side-by-side buttons in HTML
    st.markdown(
        f"""
        <div class="model-container">
            <form action="" method="get">
                <button class="model-btn claude" type="submit" name="model" value="Claude">🪄<br>Claude</button>
            </form>
            <form action="" method="get">
                <button class="model-btn llama" type="submit" name="model" value="Llama">🦙<br>LLaMA</button>
            </form>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Sync model selection with session_state ---
    query_params = st.query_params
    if "model" in query_params:
        st.session_state.selected_model = query_params["model"][0]

    # --- Chat history rendering ---
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style='background-color:#DCF8C6; padding:10px; 
                                border-radius:12px; margin:5px 30% 5px 5px;'>
                        <b>You:</b><br>{msg["content"]}
                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style='background-color:#F1F0F0; padding:10px; 
                                border-radius:12px; margin:5px 5px 5px 30%;'>
                        <b>{st.session_state.selected_model or "Assistant"}:</b><br>{msg["content"]}
                    </div>
                    """, unsafe_allow_html=True
                )

    # --- Chat input ---
    user_question = st.chat_input("Fire a Question Into the Flow...	🌪️💬")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Retrieve FAISS index
        faiss_index = FAISS.load_local(
            "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True
        )

        if st.session_state.selected_model == "Claude":
            llm = get_claude_3_Sonnet_llm()
        else:
            llm = get_Llama_3_70B_Instruct_llm()

        with st.spinner("⚙️ Running Vacuum Intelligence Protocol..."):
            response = get_response_llm(llm, faiss_index, user_question)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

if __name__ == "__main__":
    main()
