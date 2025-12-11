import json
import math
from openai import OpenAI
from utils.io_helper import save_output

client = OpenAI()


# ---------------------------
# 1. Build chunks (FIXED for your JSON format)
# ---------------------------
def build_subject_chunks(data):
    """
    Extract meaningful chunks from the Subject's 'sentences' field.
    Your JSON structure:
    {
        "turn_id": ...,
        "speaker": "Subject",
        "sentences": [
            {"text": "...", "annotations": {...}},
            ...
        ]
    }
    """
    chunks = []
    for turn in data.get("dialogue_turns", []):
        if turn.get("speaker", "").lower() == "subject":
            for i, sent in enumerate(turn.get("sentences", [])):
                chunks.append({
                    "chunk_id": f"{turn['turn_id']}_{i}",
                    "text": sent.get("text", "").strip()
                })
    return chunks


# ---------------------------
# 2. Embedding + similarity
# ---------------------------
def embed_texts(text_list, model="text-embedding-3-small"):
    resp = client.embeddings.create(model=model, input=text_list)
    return [item.embedding for item in resp.data]


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve_top_k(chunks, chunk_embeds, k=8):
    """Use generic query to retrieve top-k relevant chunks."""
    query_text = "Key moments and reflections from this person's life story."
    query_emb = embed_texts([query_text])[0]

    scored = []
    for chunk, emb in zip(chunks, chunk_embeds):
        scored.append((cosine_sim(query_emb, emb), chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


# ---------------------------
# 3. Generate RAG-based memoir
# ---------------------------
def generate_rag_memoir(retrieved_chunks):
    evidence = "\n\n".join(
        f"[Chunk {c['chunk_id']}]\n{c['text']}" for c in retrieved_chunks
    )

    prompt = f"""
    You are a supportive memoir-writing assistant.

    Below are selected excerpts from an interview with an older adult.
    Each excerpt is evidence about their life journey.

    Your task:
    - Write a first-person memoir
    - Past tense, chronological order
    - Warm, reflective tone
    - NO hallucinations: use ONLY the evidence below
    - If details missing, stay vague.

    === Retrieved Evidence ===
    {evidence}
    """

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


# ---------------------------
# 4. 🔥 Exposed function for run_all.py
# ---------------------------
def run_rag(json_data):
    """
    Input: json_data = loaded JSON dict
    Output: rag_memoir (str)
    """

    chunks = build_subject_chunks(json_data)
    if not chunks:
        print("⚠️ No subject chunks found — skipping RAG.")
        return ""

    # embed
    chunk_texts = [c["text"] for c in chunks]
    chunk_embeds = embed_texts(chunk_texts)

    # retrieve
    retrieved = retrieve_top_k(chunks, chunk_embeds, k=8)

    # generate
    rag_text = generate_rag_memoir(retrieved)

    return rag_text
