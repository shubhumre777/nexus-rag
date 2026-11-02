import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


class Rag:

    def __init__(self, index_path="faiss_index"):

        self.index_path = index_path
        self.embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            encode_kwargs={"normalize_embeddings": True}
        )

        groq_api_key = os.getenv("GROQ_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY missing. Add it to your environment or Streamlit secrets.")

        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY missing. Add it to your environment or Streamlit secrets.")

        self.llm_1 = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=groq_api_key
        )

        self.llm_2 = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=google_api_key
        )

        self.llm = self.llm_1.with_fallbacks([self.llm_2])

        self.vector_db = None
        self.ensemble_retriever = None
        self.chat_history = {}

        self.parser = StrOutputParser()

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful document question-answering assistant.

                Answer the user's question using only the supplied context.
                If the answer is not present in the context, clearly say that the uploaded
                documents do not contain that information. Do not invent facts.

                When the context supports an answer, finish with:
                fetched from -> filename.ext

                Use only the source filename, not its folder path. Mention only sources
                that actually support the answer. If the information is absent, do not
                mention a source.

                if anyone ask out of the knowledge of the folder just reply that : I am only allowed to answer 
                according to your uploaded document not out of that .

                make sure do not answer other than the document . 

                Context:
                {context}"""
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Question: {question}")
        ])

        self.chain = self.prompt | self.llm | self.parser

    def get_session_history(self, session_id: str):
        if session_id not in self.chat_history:
            self.chat_history[session_id] = []
        return self.chat_history[session_id]

    def load_documents(self, path: str):

        folder = Path(path)

        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Invalid document folder: {path}")

        loaders = [
            DirectoryLoader(
                str(folder),
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                silent_errors=False,
            ),
            DirectoryLoader(
                str(folder),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                silent_errors=False,
            ),
            DirectoryLoader(
                str(folder),
                glob="**/*.docx",
                loader_cls=Docx2txtLoader,
                silent_errors=False,
            ),
            DirectoryLoader(
                str(folder),
                glob="**/*.csv",
                loader_cls=CSVLoader,
                loader_kwargs={"encoding": "utf-8"},
                silent_errors=False,
            ),
        ]

        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        if not documents:
            raise ValueError(f"No supported documents found in {path}")

        print(f"Documents loaded: {len(documents)}")
        return documents

    def split_document(self, docs):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No chunks were generated.")

        print(f"Chunks generated: {len(chunks)}")
        return chunks

    def setup_ensemble_retriever(self, chunks):

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 5

        faiss_retriever = self.vector_db.as_retriever(
            search_kwargs={"k": 5}
        )

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )

    def create_vector_db(self, chunks):

        index_folder = Path(self.index_path)
        index_folder.mkdir(parents=True, exist_ok=True)

        self.vector_db = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        self.vector_db.save_local(str(index_folder))
        self.setup_ensemble_retriever(chunks)

        print(f"FAISS index saved at: {index_folder}")
        print("Ensemble retriever initialized.")

        return self.vector_db

    def load_vector_db(self):

        index_folder = Path(self.index_path)

        if not (index_folder / "index.faiss").exists():
            raise FileNotFoundError(f"FAISS index not found at: {index_folder}")

        self.vector_db = FAISS.load_local(
            str(index_folder),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # The index is loaded from this application's own trusted local storage.
        chunks = list(self.vector_db.docstore._dict.values())
        self.setup_ensemble_retriever(chunks)

        print(f"FAISS index loaded from: {index_folder}")
        print("Ensemble retriever initialized.")

        return self.vector_db

    def retrieve_documents(self, question: str, k: int = 5):

        if self.ensemble_retriever is None:
            raise RuntimeError("Ensemble retriever is not initialized.")

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if k < 1:
            raise ValueError("k must be greater than 0.")

        bm25_retriever, faiss_retriever = self.ensemble_retriever.retrievers
        bm25_retriever.k = k
        faiss_retriever.search_kwargs["k"] = k

        retrieved_chunks = self.ensemble_retriever.invoke(question)

        context = "\n\n".join(
            f"[Source: {Path(doc.metadata.get('source', 'Unknown')).name}]\n"
            f"{doc.page_content}"
            for doc in retrieved_chunks
        )

        return context

    def ask(self, question: str, context: str):

        return self.chain.invoke({
            "context": context,
            "question": question,
            "history": []
        })

    def chat(self, question: str, session_id: str, k: int = 5):

        context = self.retrieve_documents(question, k=k)
        history = self.get_session_history(session_id)

        response = self.chain.invoke({
            "question": question,
            "context": context,
            "history": history
        })

        history.extend([
            HumanMessage(content=question),
            AIMessage(content=response)
        ])

        return response

    def clear_history(self, session_id: str):
        self.chat_history[session_id] = []

    def ingest(self, path: str):

        documents = self.load_documents(path)
        chunks = self.split_document(documents)
        self.create_vector_db(chunks)

        return {
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "index_path": self.index_path
        }