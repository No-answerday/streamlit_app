# Hugging Face API를 활용한 문맥 검색 설정 가이드

## 📋 목차

1. [모델 업로드](#1-모델-업로드)
2. [API 토큰 발급](#2-api-토큰-발급)
3. [환경 설정](#3-환경-설정)
4. [테스트](#4-테스트)
5. [문제 해결](#5-문제-해결)

---

## 1. 모델 업로드

### 1-1. Hugging Face CLI 설치 및 로그인

```bash
# CLI 설치
pip install huggingface-hub

# 로그인
huggingface-cli login
# 프롬프트에서 토큰 입력 (https://huggingface.co/settings/tokens 에서 발급)
```

### 1-2. 모델 업로드

```bash
# 모델 디렉토리로 이동
cd models/fine_tuned/roberta_semantic_final

# Hugging Face Hub에 업로드
huggingface-cli upload [your-username]/roberta-semantic-final . .

# 예시:
# huggingface-cli upload choimanseon/cosmetic-review-semantic . .
```

### 1-3. 모델 설정 (Hugging Face 웹사이트)

1. https://huggingface.co/[your-username]/roberta-semantic-final 접속
2. **Settings** 탭 이동
3. **Model Card** 작성 (선택사항):

   ```markdown
   ---
   license: mit
   language: ko
   tags:
   - sentence-transformers
   - feature-extraction
   - cosmetics
   - korean
   ---

   # 화장품 리뷰 의미 검색 모델

   한국어 화장품 리뷰 데이터로 파인튜닝된 RoBERTa 모델입니다.
   ```

4. **Visibility**: Public으로 설정 (무료 Inference API 사용 가능)

---

## 2. API 토큰 발급

### 2-1. 토큰 생성

1. https://huggingface.co/settings/tokens 접속
2. **New token** 클릭
3. 토큰 이름 입력 (예: `streamlit-app`)
4. **Role**: `read` 선택
5. **Generate** 클릭
6. 토큰 복사 (⚠️ 한 번만 표시됨)

### 2-2. 토큰 저장

```bash
# 로컬 개발 환경 (.env 파일)
echo "HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx" >> .env

# Streamlit Cloud 배포 시
# Settings > Secrets 에서 추가:
# HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## 3. 환경 설정

### 3-1. `.env` 파일 생성

프로젝트 루트에 `.env` 파일 생성:

```bash
# Hugging Face API 사용 설정
USE_HF_API=true

# Hugging Face API 토큰
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx

# 업로드한 모델 ID
HF_MODEL_ID=your-username/roberta-semantic-final
```

### 3-2. 환경변수 로드 (로컬 개발)

```bash
# python-dotenv 설치
pip install python-dotenv

# main.py 최상단에 추가 (이미 있을 수도 있음)
from dotenv import load_dotenv
load_dotenv()
```

### 3-3. Streamlit Cloud 배포 시

1. Streamlit Cloud Dashboard 접속
2. 앱 선택 > **Settings** > **Secrets**
3. 다음 내용 추가:

```toml
USE_HF_API = "true"
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
HF_MODEL_ID = "your-username/roberta-semantic-final"
```

---

## 4. 테스트

### 4-1. 로컬 테스트

```bash
# 앱 실행
streamlit run main.py

# 브라우저에서:
# 1. 검색 타입 선택: "문맥"
# 2. 검색어 입력: "보습이 잘되는 크림"
# 3. 검색 실행
```

### 4-2. API 동작 확인

터미널에서 다음과 같은 메시지 확인:

```
✓ Hugging Face API Vectorizer 초기화 완료
  - Model: your-username/roberta-semantic-final
  - API: Hugging Face Inference API
```

### 4-3. 직접 API 호출 테스트 (Python)

```python
import requests
import os

api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/your-username/roberta-semantic-final"
headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

response = requests.post(
    api_url,
    headers=headers,
    json={"inputs": "보습이 잘되는 크림", "options": {"wait_for_model": True}}
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()[:2]}")  # 첫 2개 토큰만 출력
```

---

## 5. 문제 해결

### 5-1. "API 오류 (HTTP 403)"

**원인**: 토큰이 잘못되었거나 권한이 없음

**해결**:

1. 토큰 재확인: https://huggingface.co/settings/tokens
2. `.env` 파일의 `HF_TOKEN` 값 확인
3. 모델이 Public인지 확인

### 5-2. "API 오류 (HTTP 503)"

**원인**: 모델이 처음 로드되는 중 (Cold Start)

**해결**:

- 자동 재시도 대기 (최대 3회, 지수 백오프)
- 보통 10-30초 후 정상 작동

### 5-3. "Model ID를 찾을 수 없음"

**원인**: `HF_MODEL_ID`가 잘못되었거나 모델이 Private

**해결**:

1. Hugging Face에서 모델 URL 확인:
   - 예: `https://huggingface.co/choimanseon/roberta-semantic-final`
   - Model ID: `choimanseon/roberta-semantic-final`
2. `.env` 파일 수정
3. 모델 Visibility를 Public으로 변경

### 5-4. 속도가 느림

**원인**: Hugging Face 무료 API는 Rate Limit 있음

**해결 옵션**:

1. **Hugging Face Pro 구독** ($9/월): 더 빠른 API
2. **Dedicated Endpoint**: 전용 서버 ($60-300/월)
3. **로컬 모델 사용**: `USE_HF_API=false` 설정

---

## 6. 비용 정보

### Hugging Face Inference API

- **무료 티어**:
  - Rate Limit: 초당 1-2회 요청
  - Cold Start: 10-30초
  - 적합: 개발/테스트, 소규모 트래픽

- **Pro 티어** ($9/월):
  - Rate Limit: 초당 10회
  - 우선순위 로딩
  - 적합: 중간 규모 프로덕션

- **Dedicated Endpoint** (종량제):
  - 전용 GPU 인스턴스
  - 무제한 요청
  - 적합: 대규모 프로덕션

### 비교: 로컬 vs API

| 항목          | 로컬 모델   | HF API                     |
| ------------- | ----------- | -------------------------- |
| **초기 로딩** | 10-20초     | 없음 (서버에서 관리)       |
| **메모리**    | ~2GB        | 없음                       |
| **속도**      | 빠름 (로컬) | 네트워크 지연 (~100-500ms) |
| **확장성**    | 제한적      | 자동 스케일링              |
| **비용**      | 서버 비용   | API 요금                   |

---

## 7. 추천 사용 시나리오

### 로컬 모델 사용 (USE_HF_API=false)

- ✅ 개발 환경
- ✅ GPU 서버 보유
- ✅ 실시간 응답 속도 중요
- ✅ 많은 검색 요청 (비용 절감)

### HF API 사용 (USE_HF_API=true)

- ✅ 프로토타입/MVP
- ✅ Streamlit Cloud 무료 티어
- ✅ GPU 없는 서버
- ✅ 간헐적 검색 요청
- ✅ 모델 파일 업로드 제한 (Streamlit Cloud 1GB)

---

## 8. 코드 예시

### API Vectorizer 직접 사용

```python
from services.hf_api_vectorizer import HuggingFaceAPIVectorizer
import os

# 초기화
vectorizer = HuggingFaceAPIVectorizer(
    model_id="your-username/roberta-semantic-final",
    api_token=os.getenv("HF_TOKEN")
)

# 단일 텍스트 인코딩
text = "보습이 잘되는 크림"
embedding = vectorizer.encode(text)
print(f"Vector shape: {embedding.shape}")  # (768,)

# 배치 인코딩
texts = ["보습 크림", "수분 크림", "영양 크림"]
embeddings = vectorizer.encode_batch(texts, batch_size=8)
print(f"Batch shape: {embeddings.shape}")  # (3, 768)
```

### 기존 코드 호환성

API Vectorizer는 기존 `BERTVectorizer`와 **동일한 인터페이스**를 제공하므로,
`recommend_similar_products.py`나 다른 코드 수정 없이 바로 사용 가능합니다.

```python
# 기존 코드 (변경 없음)
from services.recommend_similar_products import recommend_similar_products

results = recommend_similar_products(
    query_text="보습 크림",
    vectorizer=st.session_state.vectorizer,  # HF API or Local
    categories=None,
    top_n=10
)
```

---

## 9. 참고 자료

- [Hugging Face Inference API 문서](https://huggingface.co/docs/api-inference/index)
- [Hugging Face Hub Python 라이브러리](https://huggingface.co/docs/huggingface_hub/index)
- [Sentence Transformers 문서](https://www.sbert.net/)
