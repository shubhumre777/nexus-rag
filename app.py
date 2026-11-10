import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from langchain_community.embeddings import FastEmbedEmbeddings

from rag import Rag


st.set_page_config(
    page_title="NEXUS RAG",
    page_icon="📱",
    layout="wide"
)

@st.cache_resource(scope="session")
def get_embedding_model():
    return FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=64,
        threads=2,
        max_length=512
    )



# Read API keys from Streamlit secrets when deployed.
for key in ("GROQ_API_KEY", "GOOGLE_API_KEY"):
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

logo_path = Path("assets/logo.png")
if logo_path.exists():
    st.logo(str(logo_path), size="large")


# Initialize session state.
defaults = {
    "rag": None,
    "messages": [],
    "session_id": str(uuid.uuid4()),
    "index_id": None,
    "index_ready": False,
    "uploaded_files": [],
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def get_sources(answer):
    pattern = r"fetched from\s*(?:→|->)\s*([^\n]+)"
    matches = re.findall(pattern, answer, flags=re.IGNORECASE)

    sources = []
    for match in matches:
        filename = Path(match.strip()).name
        if filename and filename not in sources:
            sources.append(filename)

    return sources


def transparent_color(color, opacity=0.18):
    color = color.lstrip("#")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {opacity})"


def process_documents(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    new_index_id = str(uuid.uuid4())
    index_path = Path("user_indexes") / new_index_id

    try:
        rag = Rag(
            index_path=str(index_path),
            embeddings=get_embedding_model()
        )

        for uploaded_file in uploaded_files:
            safe_name = Path(uploaded_file.name).name
            file_path = Path(temp_dir) / safe_name

            with open(file_path, "wb") as output_file:
                output_file.write(uploaded_file.getbuffer())

        stats = rag.ingest(temp_dir)

        # Switch the session to the new index only after ingestion succeeds.
        old_index_id = st.session_state.index_id

        st.session_state.rag = rag
        st.session_state.index_id = new_index_id
        st.session_state.index_ready = True
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.uploaded_files = [
            Path(file.name).name for file in uploaded_files
        ]

        # Remove this session's previous index after the new one is ready.
        if old_index_id:
            old_path = Path("user_indexes") / old_index_id
            shutil.rmtree(old_path, ignore_errors=True)

        return stats

    except Exception:
        shutil.rmtree(index_path, ignore_errors=True)
        raise

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def display_sources(sources):
    if sources:
        with st.expander("Retrieved sources"):
            for source in sources:
                st.write(f"📄 {source}")


# Sidebar
with st.sidebar:
    st.title("📱 NEXUS RAG")
    st.caption("Your documents, one conversation away.")
    st.divider()

    st.subheader("Upload documents")

    uploaded_files = st.file_uploader(
        "Choose PDF, TXT, DOCX or CSV files",
        type=["pdf", "txt", "docx", "csv"],
        accept_multiple_files=True,
        max_upload_size=20
    )

    if st.button("Process documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        elif len(uploaded_files) > 10:
            st.warning("You can upload a maximum of 10 files at a time.")
        else:
            try:
                with st.spinner("Processing documents and building the search index..."):
                    stats = process_documents(uploaded_files)

                st.success("Documents processed successfully.")
                st.write(f"Documents loaded: {stats['documents_loaded']}")
                st.write(f"Chunks created: {stats['chunks_created']}")
                st.rerun()

            except Exception as error:
                st.error("Could not process the uploaded documents.")
                st.exception(error)

    st.divider()
    st.subheader("Customize chat")

    user_color = st.color_picker("Your message colour", "#224899")
    ai_color = st.color_picker("AI message colour", "#087A55")

    st.divider()

    if st.session_state.index_ready:
        st.success("Knowledge base ready")
        st.caption(f"{len(st.session_state.uploaded_files)} files uploaded")
        for filename in st.session_state.uploaded_files:
            st.caption(f"📄 {filename}")
    else:
        st.info("Upload documents to get started.")

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())

        if st.session_state.rag:
            st.session_state.rag.clear_history(st.session_state.session_id)

        st.rerun()


# Theme
user_background = transparent_color(user_color, 0.20)
ai_background = transparent_color(ai_color, 0.18)

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at 10% 10%, rgba(0, 150, 255, 0.12), transparent 30%),
            radial-gradient(circle at 90% 80%, rgba(0, 255, 140, 0.10), transparent 30%),
            #080D16;
        color: #EAF2F8;
    }}

    section[data-testid="stSidebar"] {{
        background: #0B1420;
        border-right: 1px solid rgba(0, 200, 255, 0.25);
    }}

    .main-title {{
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #00D9FF, #00FF9D, #008CFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .subtitle {{
        font-size: 16px;
        color: #9CB3C8;
        margin-top: -10px;
    }}

    [data-testid="stChatMessage"] {{
        padding: 18px;
        border-radius: 18px;
        margin-bottom: 14px;
        border: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(14px);
        transition: all 0.25s ease;
        animation: messageAppear 0.35s ease;
    }}

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
        background: {user_background};
        border-left: 4px solid {user_color};
        box-shadow: 0 0 12px rgba(0, 150, 255, 0.12);
    }}

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
        background: {ai_background};
        border-left: 4px solid {ai_color};
        box-shadow: 0 0 12px rgba(0, 255, 140, 0.12);
    }}

    [data-testid="stChatMessage"]:hover {{
        transform: translateY(-3px);
        box-shadow: 0 0 20px rgba(0, 200, 255, 0.18);
    }}

    [data-testid="stChatMessage"] p {{
        font-size: 17px !important;
        line-height: 1.8 !important;
        color: #F1F7FC;
    }}

    [data-testid="stChatInput"] {{
        border: 1px solid rgba(0, 220, 255, 0.45);
        border-radius: 14px;
        box-shadow: 0 0 12px rgba(0, 220, 255, 0.12);
    }}

    .stButton button {{
        border-radius: 12px;
        font-weight: 700;
        background: linear-gradient(100deg, #006DFF, #00B982);
        color: white;
        border: none;
        transition: all 0.25s ease;
    }}

    .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 0 18px rgba(0, 220, 180, 0.30);
    }}

    [data-testid="stFileUploader"] {{
        border: 1px dashed rgba(0, 220, 255, 0.5);
        border-radius: 14px;
        padding: 10px;
    }}

    @keyframes messageAppear {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Main heading
st.markdown('<div class="main-title">⚡ NEXUS RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask questions. Explore your documents. Get intelligent answers.</div>',
    unsafe_allow_html=True
)
st.divider()


# Display conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        display_sources(message.get("sources", []))


# Chat
if st.session_state.index_ready:

    question = st.chat_input("Ask something about your uploaded documents...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                try:
                    answer = st.session_state.rag.chat(
                        question,
                        st.session_state.session_id,
                        k=5
                    )

                    sources = get_sources(answer)

                    st.markdown(answer)
                    display_sources(sources)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as error:
                    st.error("Something went wrong while generating your answer.")
                    st.exception(error)

else:
    st.info("Upload your documents from the sidebar and process them to start chatting.")
    st.markdown("## Supported document formats")
    

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PDF", "Supported")
    col2.metric("TXT", "Supported")
    col3.metric("DOCX", "Supported")
    col4.metric("CSV", "Supported")

    st.markdown(
        """
        <style>
        @import url('https://googleapis.com');

        .aesthetic-upload-text {
            font-family: 'Poppins', sans-serif;
            font-size: 28px;
            font-weight: 600;
            text-align: center;
            margin: 20px 0;
            
            /* Smooth shimmering color gradient */
            background: linear-gradient(
                to right, 
                #ff7e5f 0%, 
                #feb47b 25%, 
                #86e3ce 50%, 
                #d6e4f0 75%, 
                #ff7e5f 100%
            );
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            
            /* Pulsing and sliding animation loop */
            animation: shine 10s linear infinite, pulse 0s ease-in-out infinite alternate;
        }

        @keyframes shine {
            to { background-position: 200% center; }
        }

        @keyframes pulse {
            0% { transform: scale(0.98); opacity: 0.9; }
            100% { transform: scale(1.02); opacity: 1; }
        }
        </style>

        <div class="aesthetic-upload-text">
            <<<--- ✨ Please Upload your documents first ✨
        </div>
        """,
        unsafe_allow_html=True
    )
