# 🚀 Streamlit Cloud 배포 가이드

## 배포 전 체크리스트

### 1. requirements.txt 확인

```bash
streamlit
pandas
numpy
pyarrow
plotly
awswrangler
torch
transformers
huggingface_hub
sentencepiece
```

### 2. 환경 변수 설정 (.streamlit/secrets.toml)

Streamlit Cloud 대시보드에서 Secrets 설정:

```toml
# AWS Athena 연결 정보
AWS_ACCESS_KEY_ID = "your_access_key"
AWS_SECRET_ACCESS_KEY = "your_secret_key"
AWS_DEFAULT_REGION = "ap-northeast-2"

# Hugging Face (문맥 검색 모델용 - Private 모델인 경우)
HF_TOKEN = "your_huggingface_token"
```

### 3. 모델 파일 처리

#### 옵션 A: 문맥 검색 비활성화 (간단)

- 모델 파일 없이 배포
- 상품명/키워드 검색만 사용
- 문맥 검색 시 안내 메시지 표시

#### 옵션 B: Hugging Face Hub 사용 (추천)

1. 모델을 Hugging Face에 업로드
2. `utils/model_loader.py`에서 자동 다운로드 활성화
3. 첫 실행 시 자동으로 모델 다운로드

```python
# main.py에 추가
from utils.model_loader import get_model_path

model_path = get_model_path("./models/fine_tuned/roberta_semantic_final")
vectorizer = BERTVectorizer(model_name=model_path)
```

## 배포 단계

### 1. GitHub에 Push

```bash
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main
```

### 2. Streamlit Cloud에서 배포

1. https://share.streamlit.io/ 접속
2. "New app" 클릭
3. Repository 선택: `your-username/streamlit_app`
4. Branch: `main`
5. Main file path: `main.py`
6. Deploy 클릭

### 3. Secrets 설정

1. 배포된 앱의 "Manage app" 클릭
2. Settings → Secrets 탭
3. `.streamlit/secrets.toml` 내용 복사/붙여넣기
4. Save

## 문제 해결

### 문맥 검색이 작동하지 않을 때

**증상**: "문맥 검색 모델을 찾을 수 없습니다" 메시지

**해결 방법**:

1. 모델이 Hugging Face에 업로드되었는지 확인
2. `utils/model_loader.py`의 `MODEL_ID_MAP` 설정 확인
3. Private 모델인 경우 HF_TOKEN 설정 확인

### 메모리 부족 에러

**증상**: "MemoryError" 또는 앱이 느려짐

**해결 방법**:

1. Streamlit Cloud 플랜 업그레이드 (무료: 1GB → 유료: 더 많은 리소스)
2. 모델 경량화: `torch.quantization` 사용
3. CPU 전용 PyTorch 사용:
   ```
   # requirements.txt
   torch --index-url https://download.pytorch.org/whl/cpu
   ```

### Import 에러

**증상**: "ModuleNotFoundError" 또는 "ImportError"

**해결 방법**:

1. `requirements.txt`에 모든 의존성 추가 확인
2. 로컬에서 테스트:
   ```bash
   pip install -r requirements.txt
   streamlit run main.py
   ```

## 성능 최적화

### 1. 캐싱 활용

```python
@st.cache_data(ttl=3600)
def load_data():
    return fetch_data()
```

### 2. 지연 로딩

```python
# 필요할 때만 import
if search_type == "문맥":
    from services.bert_vectorizer import BERTVectorizer
```

### 3. 세션 상태 활용

```python
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = load_vectorizer()
```

## 모니터링

### Streamlit Cloud 로그 확인

1. "Manage app" → Logs 탭
2. 실시간 로그 확인
3. 에러 발생 시 즉시 확인 가능

### 사용자 피드백 수집

```python
# 앱 하단에 추가
st.sidebar.markdown("---")
feedback = st.sidebar.text_area("피드백을 남겨주세요")
if st.sidebar.button("제출"):
    # 피드백 저장 로직
    st.success("피드백이 제출되었습니다!")
```

## 참고 자료

- [Streamlit Cloud 공식 문서](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit 배포 가이드](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- [Requirements.txt 작성법](https://pip.pypa.io/en/stable/reference/requirements-file-format/)
