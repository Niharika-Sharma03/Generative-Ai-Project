from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb, uuid

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="data/chroma_db")

def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i + size]))
    return chunks

def ingest_pdf(file_bytes: bytes, collection_name: str) -> int:
    reader = PdfReader(file_bytes)
    full_text = " ".join(p.extract_text() or "" for p in reader.pages)
    chunks = chunk_text(full_text)
    embeddings = model.encode(chunks).tolist()
    col = client.get_or_create_collection(collection_name)
    col.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(uuid.uuid4()) for _ in chunks]
    )
    return len(chunks)