# Flask application

from flask import Flask, request, jsonify
from recommender.hybrid import hybrid_recommend
from recommender.utils import save_recommendations_to_db
from recommender.utils import save_goal_recommendations
from recommender.goal_recommender import recommend_goals_all_users
import pandas as pd  # pd.Timestamp.now()를 위해 필요
from recommender.utils import get_connection

app = Flask(__name__)

@app.route('/')
def index():
    return "📚 Recommendation API is running!"

@app.route('/recommend/books', methods=['POST'])
def recommend_books():
    data = request.get_json()
    user_id = data.get("user_id")

    # 🔹 recent_books를 DB에서 자동으로 불러오기
    from recommender.utils import get_recent_books_from_db
    recent_books = get_recent_books_from_db(user_id, limit=4)

    print(f"📚 최근 읽은 책 불러오기 완료 ({len(recent_books)}권):", recent_books)

    results = hybrid_recommend(user_id, recent_books, alpha=0.8)
    save_recommendations_to_db(user_id, results)
    return jsonify(results)


# ✅ 전체 사용자 추천 (한 번에 DB 저장)
@app.route('/recommend/books/all', methods=['GET'])
def recommend_books_all():
    try:
        conn, server = get_connection()
        df_users = pd.read_sql("SELECT DISTINCT user_id FROM reading_logs;", conn)
        conn.close(); server.stop()

        from recommender.utils import get_recent_books_from_db
        from recommender.hybrid import hybrid_recommend

        all_results = {}
        for user_id in df_users["user_id"]:
            recent_books = get_recent_books_from_db(user_id, limit=3)
            if not recent_books:
                continue
            results = hybrid_recommend(user_id, recent_books, alpha=0.8)
            save_recommendations_to_db(user_id, results)
            all_results[user_id] = results

        return jsonify({
            "status": "success",
            "user_count": len(all_results),
            "timestamp": pd.Timestamp.now(tz='Asia/Seoul').strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        print("❌ 오류 발생:", e)
        return jsonify({"status": "error", "message": str(e)}), 500



# ---------------------------------------------------------
# 🔹 전체 사용자 목표 추천 + DB 저장
# ---------------------------------------------------------
@app.route('/recommend/goals/all', methods=['GET'])
def recommend_goals_all():
    try:
        print("🚀 목표 추천 계산 시작...")
        results = recommend_goals_all_users()

        print("✅ 목표 계산 완료, DB 저장 중...")
        # 🔹 수정된 부분: recommendations 키 내부 접근
        save_goal_recommendations(results["recommendations"])
        print("✅ DB 저장 완료")

        rec_summary = {
            "status": "success",
            "user_count": len(results["recommendations"]),
            "inactive_count": int(results["inactivity_df"]["inactive"].sum()),
            "report_rows": len(results["report_df"]),
            "timestamp": pd.Timestamp.now(tz='Asia/Seoul').strftime("%Y-%m-%d %H:%M:%S")
        }
        return jsonify(rec_summary), 200

    except Exception as e:
        print("❌ 오류 발생:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------------------------------------------------
# 🔹 특정 사용자 목표 추천 (조회만, DB 저장 X)
# ---------------------------------------------------------
@app.route('/recommend/goals/user/<int:user_id>', methods=['GET'])
def recommend_goal_for_user(user_id):
    """
    ✅ 특정 사용자의 목표 추천만 계산해서 즉시 반환
    """
    try:
        from recommender.goal_recommender import load_data, preprocess_logs, \
            recommend_goals_for_user, rule_based_time_recommendation, \
            recommend_weekly_mission, detect_inactivity

        df_logs, df_goals = load_data()
        logs = preprocess_logs(df_logs)
        goals = df_goals.copy()

        df_user_logs = logs[logs["user_id"] == user_id]
        df_user_goals = goals[goals["user_id"] == user_id]

        goal_pred = recommend_goals_for_user(user_id, goals)
        rule_rec = rule_based_time_recommendation(df_user_logs)
        mission = recommend_weekly_mission(df_user_logs, df_user_goals)
        inactivity = detect_inactivity(logs)
        inactive_info = inactivity[inactivity["user_id"] == user_id].to_dict("records")

        return jsonify({
            "user_id": user_id,
            "goal_prediction": goal_pred,
            "rule_recommendation": rule_rec,
            "mission_recommendation": mission,
            "inactivity": inactive_info
        }), 200

    except Exception as e:
        print("❌ 사용자 추천 오류:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    



