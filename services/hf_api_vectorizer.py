"""
Hugging Face Inference API를 사용한 벡터화 모듈
"""

import numpy as np
import requests
from typing import List, Optional
import os
import time


class HuggingFaceAPIVectorizer:
    """
    Hugging Face Inference API를 사용한 벡터화 클래스
    로컬 모델 로드 없이 API 호출만으로 임베딩 생성
    """

    def __init__(
        self,
        model_id: str,
        api_token: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        """
        Args:
            model_id: Hugging Face 모델 ID (예: "your-username/roberta-semantic-final")
            api_token: Hugging Face API 토큰 (환경변수 HF_TOKEN에서 자동 로드)
            api_url: 커스텀 API URL (기본값: Hugging Face Inference API)
        """
        self.model_id = model_id

        # API 토큰 로드 (환경변수 우선)
        self.api_token = api_token or os.getenv("HF_TOKEN")
        if not self.api_token:
            raise ValueError(
                "Hugging Face API 토큰이 필요합니다. "
                "환경변수 HF_TOKEN을 설정하거나 api_token 파라미터를 전달하세요."
            )

        # API URL 설정
        if api_url:
            self.api_url = api_url
        else:
            # Hugging Face Inference API (무료/Pro)
            self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"

        self.headers = {"Authorization": f"Bearer {self.api_token}"}

        print(f"✓ Hugging Face API Vectorizer 초기화 완료")
        print(f"  - Model: {model_id}")
        print(f"  - API: Hugging Face Inference API")

    def encode(self, text: str, max_retries: int = 3) -> np.ndarray:
        """
        단일 텍스트를 벡터로 변환 (API 호출)

        Args:
            text: 입력 텍스트
            max_retries: API 실패 시 재시도 횟수

        Returns:
            768차원 벡터 (또는 모델에 따라 다른 차원)
        """
        if not text or not text.strip():
            return np.zeros(768)  # 빈 텍스트는 zero 벡터

        payload = {
            "inputs": text,
            "options": {"wait_for_model": True},  # 모델 로딩 대기
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url, headers=self.headers, json=payload, timeout=30
                )

                if response.status_code == 200:
                    # API 응답: [[token1_vec, token2_vec, ...]]
                    embeddings = response.json()

                    # Mean Pooling (CLS 토큰 제외)
                    if isinstance(embeddings, list) and len(embeddings) > 0:
                        # 첫 번째 배치의 토큰 벡터들을 평균
                        token_embeddings = np.array(embeddings[0])
                        mean_embedding = np.mean(token_embeddings, axis=0)
                        return mean_embedding
                    else:
                        return np.zeros(768)

                elif response.status_code == 503:
                    # 모델 로딩 중 - 재시도
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt  # 지수 백오프
                        print(
                            f"⏳ 모델 로딩 중... {wait_time}초 후 재시도 ({attempt+1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception("모델 로딩 타임아웃. 나중에 다시 시도해주세요.")

                else:
                    raise Exception(
                        f"API 오류 (HTTP {response.status_code}): {response.text}"
                    )

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"⏳ 타임아웃 - 재시도 중... ({attempt+1}/{max_retries})")
                    time.sleep(2)
                    continue
                else:
                    raise Exception("API 호출 타임아웃")

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ 오류 발생 - 재시도 중... ({attempt+1}/{max_retries})")
                    time.sleep(2)
                    continue
                else:
                    raise

        raise Exception("API 호출 최대 재시도 횟수 초과")

    def encode_batch(
        self, texts: List[str], batch_size: int = 8, show_progress: bool = False
    ) -> np.ndarray:
        """
        여러 텍스트를 배치로 벡터화 (API 호출)

        Args:
            texts: 입력 텍스트 리스트
            batch_size: 배치 크기 (API 제한 고려)
            show_progress: 진행상황 표시 여부

        Returns:
            (len(texts), embedding_dim) 크기의 numpy 배열
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            if show_progress:
                print(f"처리 중: {i}/{len(texts)}")

            for text in batch:
                emb = self.encode(text)
                embeddings.append(emb)

            # API Rate Limit 방지
            time.sleep(0.5)

        return np.array(embeddings)


# 싱글톤 인스턴스 캐싱
_hf_api_vectorizer_instance = None


def get_hf_api_vectorizer(
    model_id: str = "fullfish/multicampus_semantic",
    api_token: Optional[str] = None,
) -> HuggingFaceAPIVectorizer:
    """
    Hugging Face API Vectorizer 싱글톤 인스턴스 반환

    Args:
        model_id: Hugging Face 모델 ID
        api_token: Hugging Face API 토큰 (선택, 환경변수에서 자동 로드)

    Returns:
        HuggingFaceAPIVectorizer 인스턴스
    """
    global _hf_api_vectorizer_instance

    if _hf_api_vectorizer_instance is None:
        _hf_api_vectorizer_instance = HuggingFaceAPIVectorizer(
            model_id=model_id, api_token=api_token
        )

    return _hf_api_vectorizer_instance
