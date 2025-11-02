import os
import base64
import tempfile
import json
import re
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    TextLoader
)
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Load environment
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Streamlit page setup
st.set_page_config(page_title="InsightLens", page_icon="💬", layout="wide")

# --- Utility Functions ---

def load_document(file_bytes, file_name):
    """Loads supported document types directly from BytesIO."""
    file_extension = os.path.splitext(file_name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    if file_extension == ".pdf":
        loader = UnstructuredPDFLoader(tmp_path)
    elif file_extension == ".docx":
        loader = UnstructuredWordDocumentLoader(tmp_path)
    elif file_extension == ".pptx":
        loader = UnstructuredPowerPointLoader(tmp_path)
    elif file_extension == ".txt":
        loader = TextLoader(tmp_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = file_name
    return documents


def setup_vectorstore(documents):
    """Create embeddings and store in FAISS (non-cached for Streamlit Cloud)."""
    if not documents:
        return None

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    if not chunks:
        return None

    return FAISS.from_documents(chunks, embedding)


def create_chain(vector_store):
    """Conversation chain with custom prompt."""
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)
    retriever = vector_store.as_retriever()

    memory = ConversationBufferMemory(output_key="answer", memory_key="chat_history", return_messages=True)

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are InsightLens, a friendly and concise document-based chatbot.

        Use Markdown for formatting.

        If the query is unrelated to the context, answer from your knowledge base
        and clearly state it's "out of context" and based on general knowledge.

        Return only JSON in this exact structure:
        {{
            "answer": "<your answer>",
            "has_source": <true_or_false>
        }}

        Question: {question}
        Context: {context}
        """
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

def sanitize_json(raw):
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)
        return json.loads(raw)
    except:
        return {"answer": raw, "has_source": False}


# --- Streamlit UI ---

st.image("InsightLens_logo.png", width=400)
st.sidebar.header("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, Word, PPTx, or TXT",
    type=["pdf", "docx", "pptx", "txt"],
    accept_multiple_files=True
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    all_docs = []
    for file in uploaded_files:
        docs = load_document(file.read(), file.name)
        all_docs.extend(docs)

    vector_store = setup_vectorstore(all_docs)
    if vector_store:
        st.session_state.conversation_chain = create_chain(vector_store)
        st.sidebar.success(f"✅ Processed {len(uploaded_files)} file(s)")
    else:
        st.sidebar.error("Failed to process documents.")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask InsightLens...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    if "conversation_chain" not in st.session_state:
        st.chat_message("assistant").markdown("⚠️ Please upload a file first.")
    else:
        chain = st.session_state.conversation_chain
        response = chain({"question": user_input})
        parsed = sanitize_json(response["answer"])

        assistant_response = parsed.get("answer", "Sorry, I couldn't parse the response.")
        st.chat_message("assistant").markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Show source if available
        if parsed.get("has_source") and response.get("source_documents"):
            src = response["source_documents"][0].metadata.get("source", "Unknown")
            st.markdown(f"**Source:** {src}")

            # Allow viewing uploaded file directly in browser
            for uf in uploaded_files:
                if uf.name == src and uf.type == "application/pdf":
                    pdf_data = base64.b64encode(uf.read()).decode("utf-8")
                    st.markdown(
                        f"""
                        <iframe src="data:application/pdf;base64,{pdf_data}" 
                        width="100%" height="500px" style="border:none;"></iframe>
                        """,
                        unsafe_allow_html=True,
                    )
                    break

# Sidebar metrics (lightweight)
if "query_timestamps" not in st.session_state:
    from collections import deque
    st.session_state.query_timestamps = deque(maxlen=60)

import time
if user_input:
    now = time.time()
    st.session_state.query_timestamps.append(now)

now = time.time()
st.session_state.query_timestamps = [t for t in st.session_state.query_timestamps if now - t < 60]

st.sidebar.divider()
st.sidebar.markdown(f"**Queries per Minute:** {len(st.session_state.query_timestamps)} / 30")
