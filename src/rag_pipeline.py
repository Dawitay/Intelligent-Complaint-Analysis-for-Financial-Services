from transformers import pipeline
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

index = faiss.read_index("vector_store/faiss_index.idx")
with open("vector_store/metadata.pkl", "rb") as f:
    chunks, metadata = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text-generation", model="gpt2")

def answer_question(question, k=5):
    q_vec = model.encode([question])
    D, I = index.search(q_vec, k)
    context = "\n".join([chunks[i] for i in I[0]])
    prompt = f"""
    You are a financial analyst assistant for CrediTrust. Use the following complaint excerpts to answer the question.
    Context:
    {context}
    Question: {question}
    Answer:
    """
    result = generator(prompt, max_length=256, do_sample=True)[0]['generated_text']
    return result, [chunks[i] for i in I[0]]
