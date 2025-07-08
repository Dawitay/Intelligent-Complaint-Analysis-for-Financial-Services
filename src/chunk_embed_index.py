from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv("data/filtered_complaints.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Model loaded with dimension: {model.get_sentence_embedding_dimension()}")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks, metadata = [], []

for idx, row in df.iterrows():
    parts = splitter.split_text(row['cleaned_narrative'])
    chunks.extend(parts)
    metadata.extend([{"product": row['Product'], "index": idx}] * len(parts))

embeddings = model.encode(chunks, show_progress_bar=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

faiss.write_index(index, "vector_store/faiss_index.idx")
with open("vector_store/metadata.pkl", "wb") as f:
    pickle.dump((chunks, metadata), f)
