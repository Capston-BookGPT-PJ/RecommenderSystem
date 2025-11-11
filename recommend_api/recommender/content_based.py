## 콘텐츠 기반 추천 시스템

import faiss, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jhgan/ko-sroberta-multitask")
df_books = pd.read_pickle("data/book_meta.pkl")
embeddings = np.load("data/book_embeddings.npy")
index = faiss.read_index("data/book_faiss.index")

def recommend_content_based(title, author, top_n=3):
    query_text = f"{title} {author}".strip()
    qvec = model.encode([query_text], normalize_embeddings=True).astype('float32')
    sims, inds = index.search(qvec, top_n)

    recs = []
    for idx, sim in zip(inds[0], sims[0]):
        book = df_books.iloc[idx]
        recs.append({
            "book_title": book["BOOK_TITLE_NM"],
            "author": book["AUTHR_NM"],
            "book_cover_url": book.get("COVER_URL") or book.get("image_url"),
            "similarity": float(sim),
            "publisher": book["PUBLISHER_NM"],
            "category": book["KDC_NM"]
        })
    return recs
