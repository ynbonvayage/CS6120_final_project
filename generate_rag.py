import os
import sys
import json
import math
from openai import OpenAI
from utils.io_helper import save_output

# 需要环境变量 OPENAI_API_KEY
client = OpenAI()


def load_transcript_json(path):
    """Load annotated JSON transcript file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def build_subject_chunks(data):
    """
    Use each Subject turn as a retrieval chunk.

    Returns: list of dicts:
        [{"chunk_id": turn_id, "text": "..."}]
    """
    chunks = []
    for turn in data.get("dialogue_turns", []):
        if turn.get("speaker", "").lower() == "subject":
            chunks.append(
                {
                    "chunk_id": turn.get("turn_id"),
                    "text": turn.get("text", "").strip(),
                }
            )
    return chunks


def embed_texts(text_list, model="text-embedding-3-small"):
    """Get embeddings for a list of texts."""
    resp = client.embeddings.create(model=model, input=text_list)
    # resp.data[i].embedding is a list[float]
    return [item.embedding for item in resp.data]


def cosine_sim(a, b):
    """Cosine similarity between two vectors (lists of floats)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve_top_k(chunks, chunk_embeds, k=8):
    """
    Simple semantic retrieval:
    - Use a generic 'memoir reconstruction' query
    - Return top-k most relevant chunks
    """
    query_text = (
        "Key moments and reflections from this person's life story, "
        "useful for writing a memoir."
    )
    query_emb = embed_texts([query_text])[0]

    scored = []
    for chunk, emb in zip(chunks, chunk_embeds):
        score = cosine_sim(query_emb, emb)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: min(k, len(scored))]
    # only return the chunk dicts
    return [c for _, c in top]


def generate_rag_memoir(retrieved_chunks):
    """
    RAG-based memoir generation:
    LLM only sees retrieved chunks as evidence.
    """
    evidence = "\n\n".join(
        [f"[Chunk {c['chunk_id']}]\n{c['text']}" for c in retrieved_chunks]
    )

    prompt = f"""
    You are a supportive memoir-writing assistant.

    Below are selected excerpts from an interview with an older adult.
    Each excerpt is evidence about their life journey.

    Your task:
    - Write a first-person life memoir
    - Use past tense and chronological order
    - Keep a warm, reflective tone
    - DO NOT invent new facts; stay grounded in the provided excerpts
    - If some details are missing, stay vague instead of hallucinating

    === Retrieved Evidence Excerpts ===
    {evidence}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_rag.py <json_file>")
        sys.exit(1)

    json_path = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(json_path))[0]

    # 1) load JSON
    data = load_transcript_json(json_path)

    # 2) build chunks from Subject turns
    chunks = build_subject_chunks(data)
    if not chunks:
        print("No Subject turns found in dialogue_turns.")
        sys.exit(1)

    # 3) embed chunks
    chunk_texts = [c["text"] for c in chunks]
    chunk_embeds = embed_texts(chunk_texts)

    # 4) retrieve top-k evidence chunks
    retrieved = retrieve_top_k(chunks, chunk_embeds, k=8)

    # 5) generate memoir based only on retrieved evidence
    memoir_output = generate_rag_memoir(retrieved)

    # 6) save using shared helper
    save_output("rag", base_name, memoir_output)


if __name__ == "__main__":
    main()
