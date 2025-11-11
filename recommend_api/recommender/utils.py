## DB 연결 유틸

import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

# 실제 접속은 포함하지 않음 — 샘플 형식만 유지
def get_connection():
    """
    🔒 서버 보안 보호를 위해 공개 레포에서는 실제 연결 로직을 제거했습니다.
    """
    pass



# -------------------------------------------------
# 🔹 책 추천 결과를 recommend 테이블에 저장
# -------------------------------------------------
def save_recommendations_to_db(user_id, recs):
    conn, server = get_connection()
    cur = conn.cursor()
    for r in recs:
        cur.execute("""
            INSERT INTO book_recommend (user_id, book_title, author, book_cover_url, hybrid_score)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            user_id,
            r.get("book_title"),
            r.get("author"),
            r.get("book_cover_url"),
            r.get("hybrid_score")
        ))
    conn.commit()
    conn.close(); server.stop()
    print(f"✅ User {user_id} 추천 결과 DB 저장 완료")

# -------------------------------------------------
# 🔹 목표 추천 결과를 goal_recommend 테이블에 저장 (정상 작동 버전)
# -------------------------------------------------
def save_goal_recommendations(recommendations):
    conn, server = get_connection()
    cursor = conn.cursor()

    for user_id, data in recommendations.items():
        g = data.get("goal_prediction", {}) or {}
        r = data.get("rule_recommendation", {}) or {}
        m = data.get("mission_recommendation", {}) or {}
        i = data.get("inactivity", {}) or {}

        cursor.execute("""
            INSERT INTO goal_recommend (
                user_id,
                recommended_books, recommended_minutes, recommended_reviews,
                preferred_period, preferred_hour, session_minutes, days_per_week,
                recommended_weekly_minutes, rationale,
                days_since_last_read, inactive_flag,
                created_at
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
        """, (
            user_id,
            g.get("recommended_books", 0),
            g.get("recommended_minutes", 0),
            g.get("recommended_reviews", 0),
            r.get("preferred_period"),
            r.get("hour"),
            r.get("session_minutes"),
            r.get("days_per_week"),
            m.get("recommended_weekly_minutes"),
            m.get("rationale"),
            i.get("days_since_last_read"),
            int(i.get("inactive", False))
        ))

    conn.commit()
    conn.close()
    server.stop()
    print("✅ goal_recommend 테이블 저장 완료 (정상 데이터)")

# -------------------------------------------------
# 🔹 최근 읽은 책 + 책 메타정보 조인
# -------------------------------------------------
def get_recent_books_from_db(user_id, limit=3):
    """
    ✅ MySQL에서 사용자의 최근 읽은 책 n권(title, author, category, cover) 조회
    """
    try:
        conn, server = get_connection()
        query = f"""
            SELECT 
                b.title, 
                b.author, 
                b.category_name AS category, 
                b.cover AS book_cover_url
            FROM reading_logs r
            JOIN books b ON r.book_id = b.book_id
            WHERE r.user_id = {user_id}
            ORDER BY r.read_at DESC
            LIMIT {limit};
        """
        df = pd.read_sql(query, conn)
        conn.close(); server.stop()
        if df.empty:
            print(f"⚠️ 사용자 {user_id}의 최근 책이 없습니다.")
            return []
        return df.to_dict("records")

    except Exception as e:
        print(f"❌ 최근 읽은 책 조회 오류: {e}")
        return []
