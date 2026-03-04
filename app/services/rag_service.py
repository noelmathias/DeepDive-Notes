import os
import re
from collections import Counter
from dotenv import load_dotenv
from groq import Groq

from app.services.embedding_service import create_embedding
from app.services.vector_store_service import search_similar

# Load .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "llama-3.1-8b-instant"  # Fast + accurate on Groq


# ---------------------------------------------------
# Keyword Overlap Scoring
# ---------------------------------------------------
def keyword_score(query, text):
    query_words = set(re.findall(r'\w+', query.lower()))
    text_words = set(re.findall(r'\w+', text.lower()))

    if not query_words:
        return 0.0

    overlap = query_words.intersection(text_words)
    return len(overlap) / len(query_words)


# ---------------------------------------------------
# Hybrid Retrieval (Session-Based)
# ---------------------------------------------------
def retrieve_context(user_query: str, session_id: str, top_k=8):

    query_embedding = create_embedding(user_query)

    raw_results = search_similar(
        query_embedding=query_embedding,
        session_id=session_id,
        top_k=top_k
    )

    ranked = []

    for item in raw_results:
        metadata = item["metadata"]

        semantic_similarity = 1 / (1 + item["score"])

        combined_text = (
            metadata["title"] + " " +
            metadata["summary"] + " " +
            " ".join(metadata["key_points"])
        )

        kw_similarity = keyword_score(user_query, combined_text)

        final_score = 0.7 * semantic_similarity + 0.3 * kw_similarity

        ranked.append({
            "metadata": metadata,
            "score": final_score
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    return ranked[:5]


# ---------------------------------------------------
# Answer Question (Session-Isolated)
# ---------------------------------------------------
def answer_question(user_query: str, session_id: str):

    retrieved = retrieve_context(
        user_query=user_query,
        session_id=session_id,
        top_k=15
    )

    if not retrieved:
        return {
            "answer": "Answer not found in indexed content.",
            "sources": [],
            "confidence": 0.0,
            "supporting_segments": 0
        }

    # ---------------------------------------------------
    # Dominant Video Filtering (CRITICAL FIX)
    # ---------------------------------------------------
    video_counts = Counter(
        item["metadata"]["video_id"] for item in retrieved
    )

    dominant_video, dominant_count = video_counts.most_common(1)[0]
    total_segments = len(retrieved)

    # If one video dominates → filter strictly to that video
    if dominant_count / total_segments >= 0.6:
        retrieved = [
            item for item in retrieved
            if item["metadata"]["video_id"] == dominant_video
        ]

    # ---------------------------------------------------
    # Build Context
    # ---------------------------------------------------
    similarity_scores = []
    context_blocks = []

    for i, item in enumerate(retrieved):
        seg = item["metadata"]
        similarity_scores.append(item["score"])

        block = f"""
Segment {i+1}
Title: {seg['title']}
Summary: {seg['summary']}
Key Points: {', '.join(seg['key_points'])}
"""
        context_blocks.append(block)

    context = "\n".join(context_blocks)
    action_items = extract_action_items(context)

    # ---------------------------------------------------
    # Strict Prompt
    # ---------------------------------------------------
    prompt = f"""
You are a strict retrieval-based QA system.

Rules:
- Use ONLY the information from the context.
- Do NOT add outside knowledge.
- Maximum 3 sentences.
- If answer is not clearly present, respond EXACTLY:
Answer not found in indexed content.

Context:
{context}

Question:
{user_query}

Answer:
"""

    # ---------------------------------------------------
    # Groq Call
    # ---------------------------------------------------
    chat_completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You answer strictly from context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=300,
    )

    answer = chat_completion.choices[0].message.content.strip()

    # Hard guard (extra safety)
    if "Answer not found" not in answer and len(answer) < 3:
        answer = "Answer not found in indexed content."

    # ---------------------------------------------------
    # Confidence
    # ---------------------------------------------------
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    confidence = round(min(avg_similarity, 1.0), 2)

    return {
        "answer": answer,
        "sources": [item["metadata"] for item in retrieved],
        "confidence": confidence,
        "supporting_segments": len(retrieved),
        "action_items":action_items
    }
def extract_action_items(context: str):
    action_prompt = f"""
From the following context, extract any meaningful, structured takeaways.
These may include:
-Key insights
-Important principles
-Practical steps
-Advice given
-Strategies mentioned
-Warnings or cautions

Rules:
- Use ONLY the context.
- Return bullet points.
- Do not invent information
- If no structured takeaways exist, return exactly:
 No takeaways found.

Context:
{context}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Extract grounded action items only."},
            {"role": "user", "content": action_prompt}
        ],
        temperature=0,
        max_tokens=200,
    )

    return response.choices[0].message.content.strip()