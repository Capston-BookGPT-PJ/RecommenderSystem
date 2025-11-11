import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from recommender.utils import get_connection

# ============================================================
# 🔹 데이터 로드 및 전처리
# ============================================================
def load_data():
    conn, server = get_connection()
    df_logs = pd.read_sql("SELECT * FROM reading_logs;", conn)
    df_goals = pd.read_sql("SELECT * FROM reading_goals;", conn)
    conn.close(); server.stop()
    return df_logs, df_goals


def preprocess_logs(df_logs):
    """ read_at/created_at 처리, 요일·시간대 계산 """
    df = df_logs.copy()
    if 'read_at' in df.columns:
        df['read_at'] = pd.to_datetime(df['read_at'], errors='coerce')
    elif 'created_at' in df.columns:
        df['read_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    else:
        raise ValueError("read_at 또는 created_at 컬럼 필요")

    df = df.dropna(subset=['read_at'])
    df['hour'] = df['read_at'].dt.hour
    df['weekday'] = df['read_at'].dt.day_name()
    df['is_weekend'] = df['read_at'].dt.weekday >= 5
    df['minutes_read'] = pd.to_numeric(df.get('minutes_read', 0), errors='coerce').fillna(0).astype(float)
    df['pages_read'] = pd.to_numeric(df.get('pages_read', 0), errors='coerce').fillna(0).astype(float)
    df['ppm'] = df.apply(lambda r: (r['pages_read'] / r['minutes_read']) if r['minutes_read'] > 0 else 0.0, axis=1)
    return df


# ============================================================
# 🔹 목표 예측 (Linear Regression)
# ============================================================
def recommend_goals_for_user(user_id: int, df_goals: pd.DataFrame):
    df_user = df_goals[df_goals["user_id"] == user_id]
    recommendations = {}

    if len(df_user) < 2:
        return None

    try:
        X_time = df_user[["target_minutes"]]; y_time = df_user["completed_minutes"]
        model_time = LinearRegression().fit(X_time, y_time)
        recommendations["recommended_minutes"] = int(model_time.predict([[300]])[0])

        X_books = df_user[["target_books"]]; y_books = df_user["completed_books"]
        model_books = LinearRegression().fit(X_books, y_books)
        recommendations["recommended_books"] = int(model_books.predict([[5]])[0])

        X_reviews = df_user[["target_reviews"]]; y_reviews = df_user["completed_reviews"]
        model_reviews = LinearRegression().fit(X_reviews, y_reviews)
        recommendations["recommended_reviews"] = int(model_reviews.predict([[3]])[0])

    except Exception as e:
        recommendations["error"] = str(e)

    return recommendations


# ============================================================
# 🔹 규칙 기반 시간 추천
# ============================================================
def rule_based_time_recommendation(df_logs_user, min_sessions=3):
    if df_logs_user.shape[0] < min_sessions:
        return {'reason': 'cold_start', 'hour': 20, 'preferred_period': 'evening', 'session_minutes': 20, 'days_per_week': 3}

    def period_of_hour(h):
        if 5 <= h < 11: return 'morning'
        if 11 <= h < 15: return 'afternoon'
        if 15 <= h < 19: return 'late_afternoon'
        if 19 <= h < 23: return 'evening'
        return 'night'

    df_logs_user['period'] = df_logs_user['hour'].apply(period_of_hour)
    top_period = df_logs_user['period'].mode().iloc[0]
    top_hour = int(df_logs_user['hour'].mode().iloc[0])
    avg_minutes = df_logs_user['minutes_read'].median() or df_logs_user['minutes_read'].mean() or 20
    session_minutes = int(max(5, round(avg_minutes * 1.1)))

    df_logs_user['week'] = df_logs_user['read_at'].dt.isocalendar().week
    recent_weeks = df_logs_user.groupby('week').size().sort_index(ascending=False).head(4)
    days_per_week = int(round(recent_weeks.mean())) if len(recent_weeks) > 0 else 3
    days_per_week = max(1, min(7, days_per_week))

    return {
        'reason': 'rule_based',
        'preferred_period': top_period,
        'hour': top_hour,
        'session_minutes': session_minutes,
        'days_per_week': days_per_week
    }


# ============================================================
# 🔹 독서 중단 감지
# ============================================================
def detect_inactivity(df_logs, threshold_days=5, as_of=None):
    if as_of is None:
        as_of = pd.Timestamp.now()
    last = df_logs.groupby('user_id')['read_at'].max().reset_index().rename(columns={'read_at': 'last_read'})
    last['days_since_last_read'] = (as_of - last['last_read']).dt.days
    last['inactive'] = last['days_since_last_read'] >= threshold_days
    return last[['user_id', 'last_read', 'days_since_last_read', 'inactive']]


# ============================================================
# 🔹 개인화 미션 추천
# ============================================================
def recommend_weekly_mission(df_logs_user, df_goals_user=None):
    df = df_logs_user.copy()
    if df.shape[0] == 0:
        return {'recommended_weekly_minutes': 60, 'rationale': 'cold_start_default'}

    df['week'] = df['read_at'].dt.isocalendar().week
    weekly = df.groupby('week')['minutes_read'].sum().sort_index(ascending=False).head(4)
    base = int(round(weekly.mean() if not weekly.empty else df['minutes_read'].median() * 3))
    recommended = int(round(base * 1.1))
    rationale = 'no_goal_info'

    if df_goals_user is not None and not df_goals_user.empty:
        df_goals_user['success_rate'] = df_goals_user.apply(lambda r: (r['completed_minutes'] / r['target_minutes']) if r['target_minutes'] > 0 else np.nan, axis=1)
        success = df_goals_user['success_rate'].dropna().tail(3).mean()
        if not np.isnan(success):
            if success < 0.6:
                recommended = max(10, int(round(base * 0.9)))
                rationale = f'low_success_rate({success:.2f})_reduce'
            elif success > 0.9:
                recommended = int(round(base * 1.2))
                rationale = f'high_success_rate({success:.2f})_increase'
            else:
                rationale = f'avg_success_rate({success:.2f})_small_inc'

    recommended = max(10, min(2000, recommended))
    return {'recommended_weekly_minutes': recommended, 'rationale': rationale}


# ============================================================
# 🔹 월간 리포트
# ============================================================
def monthly_report(df_goals, df_logs, year=None, month=None):
    goals = df_goals.copy()
    logs = df_logs.copy()
    if year is not None:
        goals = goals[goals['year'] == year]
        logs = logs[logs['read_at'].dt.year == year]
    if month is not None:
        goals = goals[goals['month'] == month]
        logs = logs[logs['read_at'].dt.month == month]

    agg_logs = logs.groupby('user_id').agg(
        total_minutes=('minutes_read', 'sum'),
        sessions=('log_id', 'count'),
        avg_minutes=('minutes_read', 'median')
    ).reset_index()

    agg_goals = goals.groupby('user_id').agg(
        target_minutes=('target_minutes', 'sum'),
        completed_minutes=('completed_minutes', 'sum'),
        target_books=('target_books', 'sum'),
        completed_books=('completed_books', 'sum')
    ).reset_index()

    report = pd.merge(agg_logs, agg_goals, on='user_id', how='outer').fillna(0)
    report['time_success_rate'] = report.apply(lambda r: (r['completed_minutes'] / r['target_minutes']) if r['target_minutes'] > 0 else np.nan, axis=1)
    report['book_success_rate'] = report.apply(lambda r: (r['completed_books'] / r['target_books']) if r['target_books'] > 0 else np.nan, axis=1)
    return report


# ============================================================
# 🔹 전체 사용자 추천 통합
# ============================================================
def compute_all_recommendations(df_logs, df_goals):
    logs = preprocess_logs(df_logs)
    goals = df_goals.copy()
    users = logs['user_id'].unique().tolist()
    inactivity = detect_inactivity(logs)
    reports = monthly_report(goals, logs)
    recs = {}

    for uid in users:
        df_user_logs = logs[logs['user_id'] == uid]
        df_user_goals = goals[goals['user_id'] == uid] if 'user_id' in goals.columns else pd.DataFrame()

        goal_pred = recommend_goals_for_user(uid, goals)
        rule_rec = rule_based_time_recommendation(df_user_logs)
        mission = recommend_weekly_mission(df_user_logs, df_user_goals)
        inactive_info = inactivity[inactivity['user_id'] == uid].to_dict('records')[0]

        recs[uid] = {
            'goal_prediction': goal_pred,
            'rule_recommendation': rule_rec,
            'mission_recommendation': mission,
            'inactivity': inactive_info
        }

    return {'recommendations': recs, 'report_df': reports, 'inactivity_df': inactivity}


# ============================================================
# 🔹 Flask용 외부 호출 함수 (전체 사용자 예측)
# ============================================================
def recommend_goals_all_users():
    df_logs, df_goals = load_data()
    results = compute_all_recommendations(df_logs, df_goals)
    return results

