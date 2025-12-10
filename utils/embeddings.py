from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("all-mpnet-base-v2")

def build_index(chunks):
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return chunks, embeddings

def retrieve(query, chunks, embeddings, k=5):
    q_emb = model.encode(query, convert_to_tensor=True)
    scores = util.semantic_search(q_emb, embeddings, top_k=k)[0]
    return [chunks[s["corpus_id"]] for s in scores]
