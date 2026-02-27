from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd
import csv
import os
import pickle


def load_faiss_index(index_file):
    """
    Load the FAISS index from disk.
    Args:
    - index_file (str): Path to the FAISS index file.
    Returns:
    - FAISS: Loaded FAISS index.
    """
    with open(index_file, 'rb') as f:
        db = pickle.load(f)
    return db


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def initialize_chatbot():
    """
    Initialize the Ollama model and load the FAISS index.
    Returns:
    - tuple: Tuple containing Ollama model instance and retriever.
    """
    index_file = 'faiss_index.pkl'

    llm = Ollama(model="llama3")
    db = load_faiss_index(index_file)
    retriever = db.as_retriever()

    return llm, retriever


def get_chatbot_answer(llm, retriever, question, chat_history):
    # Step 1: Reformulate question as standalone using history
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question which might reference "
         "context in the chat history, formulate a standalone question which can be "
         "understood without the chat history. Do NOT answer the question, just "
         "reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    # Only contextualize if there is history
    if chat_history:
        standalone_question = contextualize_chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
    else:
        standalone_question = question

    # Step 2: Retrieve relevant docs
    docs = retriever.invoke(standalone_question)
    context = format_docs(docs)

    # Step 3: Answer with context + history
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer the given questions based on your "
         "knowledge and the given context.\n\n{context}\n\n"
         "You are allowed to rephrase the answer based on the context. "
         "Explain it so that a normal person can understand it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = qa_prompt | llm | StrOutputParser()

    answer = qa_chain.invoke({
        "input": question,
        "context": context,
        "chat_history": chat_history
    })

    # Update history
    chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])

    return answer


def main():
    llm, retriever = initialize_chatbot()
    chat_history = []

    while True:
        user_input = input("User: ")

        if user_input.lower() == 'exit':
            print("Exiting chatbot...")
            break

        response = get_chatbot_answer(llm, retriever, user_input, chat_history)
        print("Bot:", response)
        print()


main()