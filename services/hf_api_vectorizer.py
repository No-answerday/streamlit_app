"""
Hugging Face Inference API를 사용한 벡터화 모듈 (공식 클라이언트 사용)
"""

import numpy as np
from huggingface_hub import InferenceClient
from typing import List, Optional
import os
import time


class HuggingFaceAPIVectorizer:
    """
    Hugging Face Inference API를 사용한 벡터화 클래스
    공식 InferenceClient 라이브러리 사용 (자동 라우팅)
    """

    def __init__(
        self,
        model_id: str,
        api_token: Optional[str] = None,
    ):
        """
        Args:
            model_id: Hugging Face 모델 ID (예: "fullfish/multicampus_semantic")
            api_token: Hugging Face API 토큰 (환경변수 HF_TOKEN에서 자동 로드)
        """
        self.model_id = model_id

        # API 토큰 로드
        self.api_token = api_token or os.getenv("HF_TOKEN")
        if not self.api_token:
            raise ValueError(
                "Hugging Face API 토큰이 필요합니다. "
                "환경변수 HF_TOKEN을 설정하거나 api_token 파라미터를 전달하세요."
            )

        # InferenceClient 초기화 (자동으로 올바른 엔드포인트 사용)
        self.client = InferenceClient(model=model_id, token=self.api_token)

        print(f"✓ Hugging Face API Vectorizer 초기화 완료")
        print(f"  - Model: {model_id}")
        print(f"  - API: Hugging Face Inference API (공식 클라이언트)")

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

        for attempt in range(max_retries):
            try:
                # feature_extraction: 문장 임베딩 반환
                response = self.client.feature_extraction(text)

                # numpy 배열로 변환
                embedding = np.array(response)

                # 2D (토큰 x hidden) → Mean Pooling
                if embedding.ndim == 2:
                    return np.mean(embedding, axis=0)
                # 1D (이미 문장 벡터) → 그대로 반환
                elif embedding.ndim == 1:
                    return embedding
                else:
                    return np.zeros(768)

            except Exception as e:
                error_msg = str(e)

                # 모델 로딩 중 (503)
                if "loading" in error_msg.lower() or "503" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)  # 5초, 10초, 15초
                        print(
                            f"⏳ 모델 로딩 중... {wait_time}초 후 재시도 ({attempt+1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(
                            f"모델 로딩 타임아웃 (30초 대기). "
                            f"Hugging Face에서 모델이 아직 준비되지 않았습니다. "
                            f"1-2분 후 다시 시도해주세요. 원본 에러: {error_msg}"
                        )

                # 기타 오류
                if attempt < max_retries - 1:
                    print(
                        f"⚠️ 오류 발생 - 재시도 중... ({attempt+1}/{max_retries}): {error_msg}"
                    )
                    time.sleep(3)
                    continue
                else:
                    raise Exception(f"API 호출 실패 ({type(e).__name__}): {error_msg}")

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
