from markitdown import MarkItDown
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter

md = MarkItDown()
result = md.convert(r"F:\rag_system\RAG_practice txt files\attendance_policy.txt")

text = result.text_content

splitter = RecursiveCharacterTextSplitter(chunk_size = 150 , chunk_overlap=50)

chunks = splitter.split_text(text)

print(len(chunks))

for i , doc in enumerate(chunks) :
    print("\n")
    print(f"Chunk No. {i+1}")
    print( doc , "\n")
