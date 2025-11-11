# 유저 협업 필터링 추천 시스템

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from recommender.utils import get_connection

def build_user_similarity():
    conn, server = get_connection()
    query = """
    SELECT 
        r.user_id, 
        r.book_id, 
        r.rating, 
        b.title, 
        b.author,
        b.category_name,
        b.cover AS book_cover_url   -- ✅ 커버 URL 추가
    FROM reviews r
    JOIN books b ON r.book_id = b.book_id
    WHERE r.rating IS NOT NULL
      AND r.book_id IS NOT NULL;
    """
    df = pd.read_sql(query, conn)
    conn.close(); server.stop()

    matrix = df.pivot_table(index='user_id', columns='book_id', values='rating').fillna(0)
    sim_df = pd.DataFrame(cosine_similarity(matrix), index=matrix.index, columns=matrix.index)
    return df, matrix, sim_df

# ============================================================

def recommend_collaborative(user_id, top_n=3):
    df, matrix, sim_df = build_user_similarity()
    if user_id not in sim_df.index:
        return []

    sims = sim_df[user_id].sort_values(ascending=False)[1:]  # 자기 자신 제외
    weighted = np.dot(sims.values, matrix.loc[sims.index])
    pred = weighted / (sims.sum() + 1e-9)

    rated = matrix.loc[user_id][matrix.loc[user_id] > 0].index
    preds = pd.Series(pred, index=matrix.columns).drop(rated, errors='ignore')

    top_books = preds.sort_values(ascending=False).head(top_n)
    recs = df[df['book_id'].isin(top_books.index)].drop_duplicates('book_id')
    recs['predicted_rating'] = recs['book_id'].map(top_books)

    # ✅ 커버 컬럼 포함해 반환
    return recs[[
        'book_id', 
        'title', 
        'author', 
        'category_name', 
        'book_cover_url',    # 커버 URL 포함
        'predicted_rating'
    ]].to_dict(orient='records')