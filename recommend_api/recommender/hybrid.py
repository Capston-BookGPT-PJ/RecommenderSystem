# 하이브리드 추천 (hybrid.py)

#콘텐츠 기반에 가중치 α, 협업필터링 β(=1-α) 부여
# → 최근 읽은 책 4개 × 각각 콘텐츠기반 3개씩 = 12권
# → 협업 기반 top-3과 합침

from recommender.content_based import recommend_content_based
from recommender.collaborative import recommend_collaborative

def hybrid_recommend(user_id, recent_books, alpha=0.8):
    content_recs = []
    for book in recent_books[:4]:
        content_recs.extend(recommend_content_based(book["title"], book["author"], top_n=10))

    collab_recs = recommend_collaborative(user_id, top_n=5)

    merged = []
    seen = set()

    for rec in content_recs:
        key = rec["book_title"]
        if key not in seen:
            rec["hybrid_score"] = alpha * rec["similarity"]
            merged.append({
                "book_title": rec["book_title"],
                "author": rec["author"],
                "book_cover_url": rec.get("book_cover_url"),
                "hybrid_score": rec["hybrid_score"]
            })
            seen.add(key)

    for rec in collab_recs:
        key = rec["title"]
        if key not in seen:
            merged.append({
                "book_title": rec["title"],
                "author": rec["author"],
                "book_cover_url": rec.get("book_cover_url"),  # 협업기반 DB에 cover_url 있으면 포함
                "hybrid_score": (1 - alpha) * rec["predicted_rating"]
            })
            seen.add(key)

    merged.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return merged[:12]
