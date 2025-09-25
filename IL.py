import base64
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import json
import time
from collections import deque

# Load API key from Streamlit Secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# Streamlit app config
st.set_page_config(
    page_title="InsightLens",
    page_icon="💬",
    layout="wide"
)

#### Preprocessing Functions

@st.cache_data
def load_document(file_path):
    """Reads and loads data from PDF and returns LangChain documents."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = file_path
    return documents

@st.cache_data
def setup_vectorstore(_documents):
    """Splits documents, embeds them, stores in FAISS, and returns vector database."""
    embedding = HuggingFaceEmbeddings()
    splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    doc_chunks = splitter.split_documents(_documents)
    vector_store = FAISS.from_documents(doc_chunks, embedding)
    return vector_store

def create_chain(vector_store):
    """Creates a conversational retrieval chain."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=groq_api_key)
    retriever = vector_store.as_retriever()

    memory = ConversationBufferMemory(
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a PDF Chat Bot, named InsightLens.
            
        Provide a detailed yet concise response to user queries,
        based on available documents.
            
        If the query is not in the context provided, answer from your knowledge base
        and mention that it is out of context.

        Always use Markdown formatting.
        Reference document sections when applicable.

        Return ONLY in this JSON format:
        {{
          "answer": "<your_answer_here>",
          "has_source": <true_or_false>
        }}

        Question: {question}
        Context: {context}
        """
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    
    return chain

def reset_chat():
    st.session_state.chat_history = []
    st.session_state.show_source_dialog = False
    st.session_state.current_source_path = None


### Sidebar – Upload PDFs
st.sidebar.title("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader("Upload Your PDF File", type=['pdf'], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_path = tmp_file.name
        documents = load_document(file_path)
        all_docs.extend(documents)

        if "source_path" not in st.session_state:
            st.session_state.source_path = file_path 

    st.session_state.vector_store = setup_vectorstore(all_docs)
    st.session_state.conversation_chain = create_chain(st.session_state.vector_store)
    st.sidebar.success(f"Processed {len(uploaded_files)} file(s)")
    reset_chat()

### Main UI
st.image("assets/InsightLens_logo.png", width=400)
if not uploaded_files:
    st.write("Upload a file to start chatting with InsightLens.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_source_dialog" not in st.session_state:
    st.session_state.show_source_dialog = False
if "current_source_path" not in st.session_state:
    st.session_state.current_source_path = None    

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# User input
user_input = st.chat_input("Ask InsightLens...")

if user_input:
    st.session_state.chat_history.append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        if not uploaded_files:
            assistant_response = "Upload a file first."
        else:
            try:
                response = st.session_state.conversation_chain({"question": user_input})
                try:
                    structured_response = json.loads(response['answer'])
                    assistant_response = structured_response["answer"]
                    has_source = structured_response["has_source"]
                except (json.JSONDecodeError, KeyError):
                    assistant_response = response['answer']
                    has_source = len(response.get("source_documents", [])) > 0

                st.markdown(assistant_response)

                if has_source and response.get("source_documents"):
                    first_source = response["source_documents"][0].metadata.get("source", "Unknown")
                    st.markdown(f"**Response Source:** {first_source}")
                    st.session_state.current_source_path = first_source
                    def show_source(): st.session_state.show_source_dialog = True
                    st.button("View Source", on_click=show_source)

            except Exception as e:
                assistant_response = f"Error: {str(e)}"
                st.markdown(assistant_response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

# RPM tracking
if "query_timestamps" not in st.session_state:
    st.session_state.query_timestamps = deque(maxlen=60)

if user_input:
    st.session_state.query_timestamps.append(time.time())

current_time = time.time()
st.session_state.query_timestamps = deque(
    [t for t in st.session_state.query_timestamps if current_time - t <= 60], 
    maxlen=60
)

with st.sidebar:
    st.sidebar.divider()
    st.sidebar.markdown(f""" **Rate Limits**  
                        Queries per Minute: {len(st.session_state.query_timestamps)}  
                        Max RPM: 30""")

# Source preview dialog
if st.session_state.show_source_dialog and st.session_state.current_source_path:
    with st.container():
        st.markdown("### Source Document Preview")
        with open(st.session_state.current_source_path, "rb") as file:
            pdf_data = base64.b64encode(file.read()).decode("utf-8")
            iframe_html = f"""
            <iframe src="data:application/pdf;base64,{pdf_data}" width="100%" height="500px" style="border:none;"></iframe>
            """
        st.markdown(iframe_html, unsafe_allow_html=True)
        st.button("Close", on_click=lambda: st.session_state.update({"show_source_dialog": False}))



