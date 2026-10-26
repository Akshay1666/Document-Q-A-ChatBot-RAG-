from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS, InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from time import sleep

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")


if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

def document_process(path):
    loader = PyPDFLoader(path)
    docs = loader.load()



    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)


    embeddings= GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )
  
    st.session_state.vector_db = vector_db
    st.session_state.documents_uploaded = True




st.subheader("Document Q&A ChatBot - Ask Anything")

if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []


if not st.session_state.documents_uploaded:
    file = st.file_uploader(label="Upload a PDF file", type=["pdf"])
    if file:
        with open("uploaded_document.pdf", "wb") as f:
            f.write(file.getvalue())

        with st.spinner("Processing the document..."):      
            document_process("./uploaded_document.pdf")

        st.markdown("Document processed successfully! ")
        sleep(2)
        st.rerun()

if  st.session_state.documents_uploaded and st.session_state.vector_db:
    for oneMessage in st.session_state.messages:
        role = oneMessage["role"]
        content = oneMessage["content"]

        st.chat_message(role).markdown(content)
    query = st.chat_input("Ask Anything....")
    if query:

        st.session_state.messages.append({"role":"user", "content": query})


        st.chat_message("User").markdown(query)

        documents = st.session_state.vector_db.similarity_search(query, k=2)
        context =""

        for doc in documents:
            context += doc.page_content + "\n\n"


        prompt = f"""you are ahelpful assistant that answers questions based on the context provided.
        Context: {context} and question: {query}"""

        result = llm.invoke(prompt)

        print(result.content)

        st.session_state.messages.append({"role":"AI", "content": result.content})
        st.chat_message("AI").markdown(result.content)

