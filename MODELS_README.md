# 🤖 모델 파일 관리 가이드

## 문제 상황

`roberta_semantic_final` 모델은 423MB로 Git에 직접 올리기엔 너무 큽니다.

## 해결 방법

### 옵션 1: Hugging Face Hub 사용 (추천) ⭐

#### 1단계: Hugging Face에 모델 업로드

```bash
# Hugging Face CLI 설치
pip install huggingface_hub

# 로그인
huggingface-cli login

# 모델 업로드
cd models/fine_tuned
huggingface-cli upload roberta-semantic-final ./roberta_semantic_final
```

#### 2단계: `utils/model_loader.py` 수정

```python
MODEL_ID_MAP = {
    "./models/fine_tuned/roberta_semantic_final": "YOUR_USERNAME/roberta-semantic-final",
}
```

#### 3단계: 앱 실행 시 자동 다운로드

모델이 없으면 자동으로 Hugging Face에서 다운로드됩니다.

---

### 옵션 2: Git LFS 사용

```bash
# Git LFS 설치 (Mac)
brew install git-lfs

# Git LFS 초기화
git lfs install

# 모델 파일 추적
git lfs track "models/fine_tuned/**/*.safetensors"
git lfs track "models/fine_tuned/**/*.bin"

# .gitattributes 파일이 생성됨
git add .gitattributes
git add models/
git commit -m "Add model files with Git LFS"
```

**주의**: Git LFS는 무료 계정에서 1GB 저장소, 1GB 대역폭 제한이 있습니다.

---

### 옵션 3: Google Drive 링크 공유

#### 1단계: Google Drive에 업로드

`roberta_semantic_final` 폴더를 압축하여 Google Drive에 업로드

#### 2단계: 다운로드 스크립트 생성

```bash
# scripts/download_models.sh
#!/bin/bash

# Google Drive 파일 ID (공유 링크에서 추출)
FILE_ID="YOUR_FILE_ID"

# 다운로드
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O models.zip

# 압축 해제
unzip models.zip -d models/fine_tuned/
rm models.zip

echo "✅ 모델 다운로드 완료!"
```

#### 3단계: 팀원 사용

```bash
chmod +x scripts/download_models.sh
./scripts/download_models.sh
```

---

## 권장 사항

1. **개발 환경**: Hugging Face Hub 사용 (무료, 자동화)
2. **프로덕션**: AWS S3 또는 Azure Blob Storage
3. **팀 협업**: Git LFS (소규모) 또는 Hugging Face (대규모)

## 현재 .gitignore 설정

```gitignore
/models/fine_tuned/roberta_sentiment_final
/models/fine_tuned/roberta_semantic_final
```

모델 파일은 Git에서 무시되므로, 위 방법 중 하나를 선택하여 공유하세요.

## Streamlit Cloud 배포 시

### requirements.txt에 추가

```
huggingface-hub
```

### Streamlit Secrets 설정 (Private 모델인 경우)

```toml
# .streamlit/secrets.toml
HF_TOKEN = "your_hugging_face_token"
```

### main.py에서 사용

```python
# 자동 다운로드 활성화
from utils.model_loader import get_model_path

model_path = get_model_path("./models/fine_tuned/roberta_semantic_final")
vectorizer = BERTVectorizer(model_name=model_path)
```
