import os
import sys
import json
import math
from openai import OpenAI
from utils.io_helper import save_output

client = OpenAI()

def build_subject_chunks(data):
    """
    从 JSON 里抽取 subject 的句子作为 chunk：
    dialogue_turns[] -> speaker == "Subject" -> sentences[i]["text"]
    """
    chunks = []
    for turn in data.get("dialogue_turns", []):
        if turn.get("speaker", "").lower() == "subject":
            for i, sent in enumerate(turn.get("sentences", [])):
                text = sent.get("text", "").strip()
                if text:
                    chunks.append({
                        "chunk_id": f"{turn.get('turn_id')}_{i}",
                        "text": text
                    })
    return chunks


def embed_texts(text_list):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_list
    )
    return [item.embedding for item in resp.data]


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve_top_k(chunks, chunk_embeds, k=8):
    query = "Life events relevant for a warm reflective memoir."
    q_emb = embed_texts([query])[0]

    scored = [(cosine_sim(q_emb, e), c) for c, e in zip(chunks, chunk_embeds)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:min(k, len(scored))]]


# ====== Few-shot generation ======
EXEMPLAR_MEMOIR = """
[STYLE EXAMPLE — DO NOT COPY FACTS]

I grew up in a small town where every corner held a memory of running barefoot
on sunlit roads. Life was simple, but not always easy. Looking back, I realize
each challenge was a gentle nudge pushing me toward who I would become.

I learned that courage is not loud — it is choosing to keep going even when you’re afraid.
"""


def generate_fewshot_memoir(retrieved_chunks):
    evidence = "\n\n".join(
        f"[Chunk {c['chunk_id']}]\n{c['text']}" for c in retrieved_chunks
    )

    prompt = f"""
You are a supportive memoir-writing assistant.

First, observe the narrative style of the EXAMPLE below:
- First-person voice
- Warm, reflective emotional tone
- Smooth transitions and story arc
- Past tense
- No invented facts

DO NOT copy example content. ONLY copy narrative style.

=== EXAMPLE MEMOIR STYLE ===
{EXEMPLAR_MEMOIR}

=== EVIDENCE FROM TRANSCRIPT ===
{evidence}

Now, write a new memoir:
- Keep style similar to the example
- Only use facts from the evidence
- Preserve chronological flow of the subject's life
- If unsure about details → remain vague rather than hallucinate
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def run_fewshot(json_data):
    chunks = build_subject_chunks(json_data)
    if not chunks:
        print("⚠️ No subject chunks found — skipping few-shot.")
        return ""

    embeds = embed_texts([c["text"] for c in chunks])
    topk = retrieve_top_k(chunks, embeds, k=8)
    memoir = generate_fewshot_memoir(topk)
    return memoir
