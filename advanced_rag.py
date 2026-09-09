"""
3-Stage RAG Pipeline
  1. Bi-encoder retrieval (nomic-embed-text via Ollama)   -> Top-K
  2. Cross-encoder rerank (sentence-transformers, CPU)     -> Top-N
  3. Answer generation (qwen3:4b via Ollama)
"""

import sys
from dataclasses import dataclass, field

import numpy as np
import ollama
from sentence_transformers import CrossEncoder

EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "qwen3:4b"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 8   # candidates from retrieval
TOP_N = 3   # candidates kept after rerank

CORPUS = [
    ("policy.txt", "Employees are entitled to 15 days of paid vacation per year, accrued monthly."),
    ("policy.txt", "Sick leave is separate from vacation and covers up to 10 days per year."),
    ("policy.txt", "Unused vacation days roll over up to a maximum of 5 days into the next year."),
    ("policy.txt", "Employees must give at least two weeks' notice before resigning."),
    ("onboarding.txt", "New hires must complete IT setup and badge registration on their first day."),
    ("onboarding.txt", "The 90-day probation period includes a mid-point check-in with your manager."),
    ("onboarding.txt", "All new employees complete a mandatory security and compliance training in week one."),
    ("benefits.txt", "Health insurance coverage begins on the first day of the month after hire date."),
    ("benefits.txt", "The company matches 401k contributions up to 4% of salary."),
    ("benefits.txt", "Dental and vision plans are optional add-ons available during open enrollment."),
    ("benefits.txt", "Employees can expense up to $300 per year for professional development courses."),
    ("remote.txt", "Remote employees must be available during core hours of 10am-3pm local time."),
    ("remote.txt", "A one-time $500 stipend is provided for home office equipment."),
    ("remote.txt", "Remote employees are required to attend one in-person team meetup per quarter."),
    ("payroll.txt", "Salaries are paid on the last business day of each month via direct deposit."),
    ("payroll.txt", "Overtime is paid at 1.5x the hourly rate for non-exempt employees."),
    ("expenses.txt", "Travel expenses must be submitted with receipts within 30 days of the trip."),
    ("expenses.txt", "Meal reimbursement is capped at $50 per day while traveling for business."),
    ("performance.txt", "Performance reviews are conducted twice a year, in June and December."),
    ("performance.txt", "Employees rated 'exceeds expectations' are eligible for a merit-based bonus."),
    ("holidays.txt", "The company observes 10 public holidays per year, listed in the HR calendar."),
    ("holidays.txt", "Employees working on a public holiday receive a compensatory day off."),
]


@dataclass
class Candidate:
    source: str
    text: str
    score: float = 0.0
    rerank_score: float = field(default=0.0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def stage1_retrieve(query: str, docs: list[tuple[str, str]], top_k: int) -> list[Candidate]:
    """Embed corpus + query, return the top_k most similar chunks."""
    texts = [text for _, text in docs]
    doc_vecs = ollama.embed(model=EMBED_MODEL, input=texts).embeddings
    query_vec = ollama.embed(model=EMBED_MODEL, input=[query]).embeddings[0]

    candidates = [
        Candidate(source=src, text=text, score=cosine_sim(query_vec, vec))
        for (src, text), vec in zip(docs, doc_vecs)
    ]
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


def stage2_rerank(query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
    """Rescore candidates with a cross-encoder for higher precision."""
    reranker = CrossEncoder(RERANK_MODEL, device="cpu")
    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs)

    for c, score in zip(candidates, scores):
        c.rerank_score = float(score)

    candidates.sort(key=lambda c: c.rerank_score, reverse=True)
    return candidates[:top_n]


def stage3_generate(query: str, final_docs: list[Candidate]) -> str:
    """Build an augmented prompt and call the local LLM for the final answer."""
    context = "\n".join(f"[{d.source}] {d.text}" for d in final_docs)
    prompt = (
        "Answer the question using only the context provided below.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    response = ollama.chat(model=GEN_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or "How much vacation do I get each year?"
    print(f"Query: {query}\n")

    print(f"--- Stage 1: Top-{TOP_K} Retrieved ---")
    retrieved = stage1_retrieve(query, CORPUS, TOP_K)
    for i, c in enumerate(retrieved, 1):
        print(f"  #{i} [{c.source}] (score: {c.score:.4f}) {c.text}")

    print(f"\n--- Stage 2: Top-{TOP_N} Reranked ---")
    reranked = stage2_rerank(query, retrieved, TOP_N)
    for i, c in enumerate(reranked, 1):
        print(f"  #{i} [{c.source}] (rerank: {c.rerank_score:.4f}) {c.text}")

    print("\n--- Stage 3: Final Answer ---")
    print(stage3_generate(query, reranked))


if __name__ == "__main__":
    main()