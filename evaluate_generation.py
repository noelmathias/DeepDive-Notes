import re
from app.services.rag_service import answer_question

# ==========================================================
# Utility: Text Cleaning
# ==========================================================

def tokenize(text):
    return set(re.findall(r'\w+', text.lower()))

# ==========================================================
# Groundedness Score
# ==========================================================

def groundedness_score(answer, sources):
    """
    Measures how much of the answer is supported by retrieved context.
    Returns score between 0 and 1.
    """

    answer_tokens = tokenize(answer)

    if not answer_tokens:
        return 0.0

    # Combine all retrieved source text
    combined_context = ""

    for src in sources:
        combined_context += (
            src["title"] + " " +
            src["summary"] + " " +
            " ".join(src["key_points"]) + " " +
            " ".join(src["concepts"]) + " "
        )

    context_tokens = tokenize(combined_context)

    overlap = answer_tokens.intersection(context_tokens)

    return round(len(overlap) / len(answer_tokens), 2)

# ==========================================================
# Test Questions
# ==========================================================

test_questions = [
    "What was Mohan's journey about?",
    "Why was Mansook exposed?",
    "What lesson does Mohan's vegetable shop story teach?",
    "What is quantum computing?"  # deliberately unrelated
]

# ==========================================================
# Evaluation Loop
# ==========================================================

print("\n==============================")
print("Running Answer-Level Evaluation")
print("==============================\n")

for q in test_questions:

    result = answer_question(q, mode="global")

    answer = result["answer"]
    sources = result["sources"]

    score = groundedness_score(answer, sources)

    hallucination_flag = score < 0.5

    print("Question:", q)
    print("Answer:", answer)
    print("Groundedness Score:", score)
    print("Hallucination Risk:", "YES" if hallucination_flag else "NO")
    print("-" * 60)