from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="data/chroma_db")

def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def retrieve(query: str, collection_name: str, top_k: int = 5, mmr_lambda: float = 0.7):
    col = client.get_collection(collection_name)
    q_emb = model.encode([query])[0]

    results = col.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k * 2,
        include=["documents", "embeddings"]
    )

    docs = results["documents"][0]
    embs = results["embeddings"][0]

    selected = []
    selected_embs = []

    for _ in range(min(top_k, len(docs))):
        scores = []
        for i in range(len(docs)):
            if docs[i] in selected:
                continue
            rel = cosine(q_emb, embs[i])
            if selected_embs:
                red = max(cosine(embs[i], s) for s in selected_embs)
            else:
                red = 0.0
            score = float(mmr_lambda * rel - (1 - mmr_lambda) * red)
            scores.append((score, docs[i], embs[i]))

        if not scores:
            break

        scores.sort(key=lambda x: x[0], reverse=True)
        selected.append(scores[0][1])
        selected_embs.append(scores[0][2])

    return selected