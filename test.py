from dotenv import load_dotenv
from pathlib import Path
from rag import Rag

load_dotenv(override=True)

rag = Rag()

# Step 1: Check whether FAISS index already exists
index_exists = (
    Path(rag.index_path) / "index.faiss"
).exists()

# Step 2: Create database if it doesn't exist
if not index_exists:
    print("No existing database found. Creating one...")

    stats = rag.ingest("RAG_practice txt files") # Here is the Path of the doument folder  . 
    print("Ingestion completed:", stats)

else:
    print("Existing database found.")
    rag.load_vector_db()

# Step 3: Start a chat session
session_id = "student_1"

while True:
    question = input("\nAsk a question (type 'exit' to quit): ").strip()

    if question.lower() == "exit":
        break

    if not question:
        print("Please enter a question.")
        continue

    answer = rag.chat(question, session_id)
    print("\nAnswer:", answer)




    """
1. What is the minimum attendance required, and what happens if a student fails to maintain it?
2. What activities are strictly prohibited inside the hostel?
3. If a student has 65% attendance and is involved in ragging, what college rules are being violated?
4. What are the responsibilities of students regarding hostel property?
5. What is the hostel fee refund policy, and what is the procedure to apply 

f"""