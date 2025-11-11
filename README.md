# RecommenderSystem – 추천 시스템 모델 및 서버

사용자의 독서 기록, 평가, 행동 패턴을 기반으로
**콘텐츠 기반 추천 + 협업 필터링 + 목표 추천(Goal Recommendation)**을 제공하는
MeltingBooks 전용 추천 시스템입니다.

이 리포지토리는 다음을 포함합니다:

✅ 책 추천 모델 코드

✅ 목표(독서습관) 추천 모델 코드

✅ 추천 API 서버 (Flask 기반)

✅ 임베딩/FAISS 기반 유사도 검색

✅ 하이브리드 추천 알고리즘

✅ 모델 실험용 노트북

### 📦 Folder Structure
```
RecommenderSystem/
│
├── recommend_api/                # Flask 추천 API 서버
│   ├── app.py                    # 책/목표 추천 API 엔드포인트
│   ├── Dockerfile                # 컨테이너 빌드 설정
│   ├── requirements.txt          # 필요한 파이썬 패키지
│   ├── data/                     # 임베딩/메타데이터 (공개 버전엔 샘플만)
│   │   ├── book_embeddings.npy
│   │   ├── book_faiss.index
│   │   └── book_meta.pkl
│   └── recommender/              # 추천 로직 모듈
│       ├── content_based.py      # SentenceTransformer + FAISS 기반 추천
│       ├── collaborative.py      # User-based 협업 필터링(CF)
│       ├── goal_recommender.py   # 독서 목표 추천 모델
│       ├── hybrid.py             # 하이브리드 추천 알고리즘
│       └── utils.py              # DB, 데이터 핸들링 유틸
│
└── recommend_model/              # 모델링 & 분석 노트북
    ├── recommend_cf.ipynb        # 협업 필터링 실험
    ├── recommend_goal.ipynb      # 목표 예측 모델 실험
    ├── book_recommend_hf_popular_books.ipynb
    └── (샘플 데이터 or 참고 CSV)
```
---
### 🧠 System Overview

이 추천 시스템은 MeltingBooks 사용자들의 독서 데이터를 활용하여
다음과 같은 기능을 제공하기 위해 설계되었습니다.

✅ 1) 콘텐츠 기반 책 추천 (Content-Based Recommendation)

> 책의 제목 + 작가 + 카테고리 정보를 BERT 기반 임베딩으로 변환, 
> SentenceTransformer(ko-sroberta-multitask) 사용, 
> FAISS index를 활용하여 빠른 유사도 검색, 
> 가장 유사한 상위 N권 추천


✅ 2) 협업 필터링 추천 (User-Based Collaborative Filtering)

> 다른 사용자와의 독서/평가 유사도를 기반으로 책을 추천, 
> Cosine Similarity + User-Book Matrix, 
> 기존 평가 기반 추천 시스템


✅ 3) 하이브리드 추천 (Hybrid Recommendation)

> 사용자 최신 읽은 책 목록 기반 콘텐츠 추천, 유사한 사용자 기반 협업 필터링 추천, α(콘텐츠) + β(협업) 가중치 기반 스코어 통합, 상위 12권 최종 추천 리스트 생성


✅ 4) 독서 목표 추천 (Goal Recommendation)

사용자의 최근 독서패턴을 기반으로:

> ✅ 이번달/이번주 추천 독서 시간

> ✅ 읽을 책 권수

> ✅ 리뷰 작성 횟수

> ✅ 최적의 독서 시간대(period 분석)

> ✅ 최근 독서 중단 감지(inactivity detection)

사용되는 기법:

> 선형 회귀(Linear Regression)

> 최근 시간대/세션 기반 통계 모델

> 최근 4주 읽은 패턴 기반 rule-based 추론


✅ 5) 전체 사용자 대상 자동 추천 계산

/recommend/books/all

/recommend/goals/all
→ 모든 사용자에 대한 추천을 계산하여 DB에 저장하는 엔드포인트 제공

---
### 🚀 API Endpoints (요약)
✅ 책 추천 API
```
POST /recommend/books


Request

{ "user_id": 12 }


Response

[
  {
    "book_title": "...",
    "author": "...",
    "book_cover_url": "...",
    "hybrid_score": 0.83
  }
]
```

✅ 전체 사용자 책 추천 저장
GET /recommend/books/all

✅ 목표 추천 (Goal Recommendation)
GET /recommend/goals/all

✅ 특정 사용자 목표 추천
GET /recommend/goals/user/{user_id}

---
### 🔍 Technical Details
✅ 모델

SentenceTransformer: "jhgan/ko-sroberta-multitask"

FAISS Index: L2/Inner product 기반 검색

CF: Cosine Similarity(User-based)


✅ DB 연동(비공개 버전 제거됨)

환경 변수 기반 DB 연결 구조 (공개 버전에서는 제거)

---
### 🛠 Development Setup
✅ 1) Install dependencies
pip install -r recommend_api/requirements.txt

✅ 2) Run API Server
cd recommend_api
python app.py

✅ 3) Docker Build
docker build -t recommender-api .
docker run -p 8000:8000 recommender-api

---
### 🌱 Future Work

✅ TensorFlow/LightGBM 기반 랭킹 모델 추가

✅ BERT-based book description embedding

✅ Sequential Recommendation (GRU4Rec / SASRec) 적용

✅ User clustering 기반 그룹 추천

✅ Cold-start 사용자 대응 강화

✅ 실시간 추천 캐싱

---
### ❤️ About
This repository is part of MeltingBooks,
a personalized reading SNS and habit management platform.
문의: @Capston-BookGPT-PJ / MeltingBooks 개발팀
