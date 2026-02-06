"""
제품 분석 컴포넌트 (대표 키워드, 대표 리뷰, 평점 추이)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.load_data import rating_trend
from services.athena_queries import fetch_representative_review_text
from utils.data_utils import load_reviews_athena, load_top_reviews_athena
from services.recommend_similar_products import recommend_similar_products


def render_top_keywords(product_info: pd.Series):
    """대표 키워드 렌더링"""
    st.markdown("---")
    st.markdown("### 📃 대표 키워드")
    top_kw = product_info.get("top_keywords_str", "")
    if isinstance(top_kw, (list, np.ndarray)):
        top_kw = ", ".join(map(str, top_kw))
    st.write(top_kw if top_kw else "-")


def render_representative_review(
    container,
    positive_reviews_df: pd.DataFrame,
    negative_reviews_df: pd.DataFrame,
    skip_scroll_callback,
):
    """대표 리뷰 렌더링 (긍정/부정 분리)"""
    with container.container():
        st.markdown("### ✒️ 대표 리뷰")

        # 탭으로 긍정/부정 구분
        tab_positive, tab_negative = st.tabs(["😊 긍정 리뷰", "😟 부정 리뷰"])

        pid = st.session_state.get("_analysis_cache_product_id", "unknown")

        # 긍정 리뷰 탭
        with tab_positive:
            if positive_reviews_df is None or positive_reviews_df.empty:
                st.info("긍정 대표 리뷰가 없습니다.")
            else:
                _render_review_pagination(
                    positive_reviews_df, "positive", pid, skip_scroll_callback
                )

        # 부정 리뷰 탭
        with tab_negative:
            if negative_reviews_df is None or negative_reviews_df.empty:
                st.info("부정 대표 리뷰가 없습니다.")
            else:
                _render_review_pagination(
                    negative_reviews_df, "negative", pid, skip_scroll_callback
                )


@st.fragment
def _render_review_pagination(
    reviews_df: pd.DataFrame, review_type: str, product_id: str, skip_scroll_callback
):
    """개별 리뷰 페이지네이션 렌더링 (fragment로 독립 실행)"""
    page_key = f"rep_review_page_{review_type}_{product_id}"

    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    total = len(reviews_df)
    page = int(st.session_state[page_key])
    page = max(0, min(page, total - 1))
    st.session_state[page_key] = page

    # 현재 페이지 리뷰 표시
    row = reviews_df.iloc[page]

    meta = []
    if "date" in reviews_df.columns and pd.notna(row.get("date")):
        meta.append(str(row.get("date")))
    if "score" in reviews_df.columns and pd.notna(row.get("score")):
        score = row.get("score")
        stars = "⭐" * int(score) if pd.notna(score) else ""
        meta.append(f"{stars} {score}점")
    if "sentiment_score" in reviews_df.columns and pd.notna(row.get("sentiment_score")):
        sentiment = row.get("sentiment_score")
        if pd.notna(sentiment):
            sentiment_pct = f"{float(sentiment) * 100:.1f}%"
            emoji = "😊" if float(sentiment) >= 0.5 else "😟"
            meta.append(f"{emoji} {sentiment_pct}")

    if meta:
        st.caption(" · ".join(meta))

    # full_text 우선, 없으면 title+content
    text = ""
    if "full_text" in reviews_df.columns and pd.notna(row.get("full_text")):
        text = str(row.get("full_text") or "")
    if not text:
        title = str(row.get("title") or "") if "title" in reviews_df.columns else ""
        content = (
            str(row.get("content") or "") if "content" in reviews_df.columns else ""
        )
        text = (title + "\n\n" + content).strip()

    if text:
        st.text(text)
    else:
        st.info("표시할 리뷰 텍스트가 없습니다.")

    # 페이지네이션 버튼
    col_l, col_m, col_r = st.columns([2, 6, 2])

    def prev_page():
        skip_scroll_callback()
        st.session_state[page_key] = max(0, st.session_state[page_key] - 1)

    def next_page():
        skip_scroll_callback()
        st.session_state[page_key] = min(total - 1, st.session_state[page_key] + 1)

    with col_l:
        st.button(
            "◀ 이전",
            on_click=prev_page,
            disabled=(page == 0),
            use_container_width=True,
            key=f"rep_prev_{review_type}_{product_id}",
        )
    with col_m:
        st.markdown(
            f"<div style='text-align:center; padding-top:6px;'>({page+1} / {total})</div>",
            unsafe_allow_html=True,
        )
    with col_r:
        st.button(
            "다음 ▶",
            on_click=next_page,
            disabled=(page >= total - 1),
            use_container_width=True,
            key=f"rep_next_{review_type}_{product_id}",
        )


def render_rating_trend(container, reviews_df: pd.DataFrame, skip_scroll_callback):
    """평점 추이 렌더링"""
    with container.container():
        st.markdown("### 📈 평점 추이")

        if (
            reviews_df.empty
            or "date" not in reviews_df.columns
            or "score" not in reviews_df.columns
        ):
            st.info("평점 추이를 그릴 리뷰 데이터가 없습니다.")
            return

        review_df = reviews_df[["date", "score"]].copy()
        review_df["date"] = pd.to_datetime(review_df["date"], errors="coerce")
        review_df["score"] = pd.to_numeric(review_df["score"], errors="coerce")
        review_df = review_df.dropna(subset=["date", "score"]).sort_values("date")

        if review_df.empty:
            st.info("평점 추이를 그릴 수 있는 날짜/평점 데이터가 없습니다.")
            return

        min_date = review_df["date"].min().date()
        max_date = review_df["date"].max().date()

        # product_id + 렌더 카운터
        pid = st.session_state.get("_analysis_cache_product_id", "unknown")
        render_key = st.session_state.get("_rating_render_key", 0)
        st.session_state["_rating_render_key"] = render_key + 1

        freq_key = f"rating_freq_{pid}_{render_key}"
        date_key = f"rating_date_{pid}_{render_key}"
        reset_key = f"rating_reset_{pid}_{render_key}"

        col_left, col_mid, col_right, _ = st.columns([1, 1, 1, 1])

        with col_left:
            freq_label = st.selectbox(
                "평균 기준",
                ["일간", "주간", "월간"],
                index=2,
                key=freq_key,
                on_change=skip_scroll_callback,
            )

        freq_map = {
            "일간": ("D", 7),
            "주간": ("W", 4),
            "월간": ("ME", 3),
        }
        freq, ma_window = freq_map[freq_label]

        DATE_RANGE_KEY = "rating_date_range"
        default_date_range = (min_date, max_date)

        with col_mid:
            date_range = st.date_input(
                "기간 선택",
                value=default_date_range,
                min_value=min_date,
                max_value=max_date,
                key=date_key,
                on_change=skip_scroll_callback,
            )

        def reset_date_range():
            skip_scroll_callback()
            st.session_state[DATE_RANGE_KEY] = (min_date, max_date)

        with col_right:
            st.markdown("<br>", unsafe_allow_html=True)
            st.button(
                "↺",
                key=reset_key,
                help="날짜 초기화",
                on_click=reset_date_range,
            )

        trend_df = pd.DataFrame()
        is_date_range_ready = False

        if isinstance(date_range, tuple) and len(date_range) == 2:
            is_date_range_ready = True
            start_date, end_date = date_range
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            date_df = review_df.loc[
                (review_df["date"] >= start_date) & (review_df["date"] <= end_date)
            ]
            if not date_df.empty:
                trend_df = rating_trend(date_df, freq=freq, ma_window=ma_window)
        else:
            st.info("마지막 날짜를 선택해주세요.📆")

        if is_date_range_ready and not trend_df.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=trend_df["date"],
                    y=trend_df["avg_score"],
                    name=f"{freq_label} 평균",
                    marker_color="slateblue",
                    opacity=0.4,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=trend_df["date"],
                    y=trend_df["ma"],
                    mode="lines",
                    name=f"추세 ({ma_window}개{freq_label} 이동평균)",
                    line=dict(color="royalblue", width=3),
                )
            )
            fig.update_layout(
                yaxis=dict(range=[1, 5.1]),
                xaxis_title="날짜",
                yaxis_title="평균 평점",
                hovermode="x unified",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
        elif is_date_range_ready and trend_df.empty:
            st.info("선택한 기간에 대한 평점 데이터가 없습니다.")


def load_product_analysis_async(
    product_id: str,
    product_info: pd.Series,
    review_id,
    container_review,
    container_trend,
    skip_scroll_callback,
):
    """
    비동기로 대표 리뷰, 평점 추이, 추천 상품 로드

    Args:
        product_id: 제품 ID
        product_info: 제품 정보 Series
        review_id: 대표 리뷰 ID
        container_review: 대표 리뷰 placeholder
        container_trend: 평점 추이 placeholder
        skip_scroll_callback: 스크롤 스킵 콜백
    """
    # 초기 로딩 메시지 표시
    with container_review.container():
        st.markdown("### ✒️ 대표 리뷰")
        st.info("✒️ 대표 리뷰를 분석 중입니다...")

    with container_trend.container():
        st.markdown("### 📈 평점 추이")
        st.info("📈 평점 데이터를 불러오는 중입니다...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_type = {}

        # 1. 긍정 대표 리뷰 요청
        if product_id:
            f_pos = executor.submit(
                load_top_reviews_athena, str(product_id), product_info, 5, "positive"
            )
            future_to_type[f_pos] = "REVIEW_POSITIVE"

        # 2. 부정 대표 리뷰 요청
        if product_id:
            f_neg = executor.submit(
                load_top_reviews_athena, str(product_id), product_info, 5, "negative"
            )
            future_to_type[f_neg] = "REVIEW_NEGATIVE"

        # 3. 평점 추이 데이터 요청
        if product_id:
            f_trend = executor.submit(load_reviews_athena, str(product_id))
            future_to_type[f_trend] = "TREND"

        # 4. 추천 상품 요청 (캐시 체크)
        if product_id and st.session_state.get("reco_target_product_id") != product_id:
            f_reco = executor.submit(
                recommend_similar_products,
                product_id=product_id,
                categories=None,
                top_n=100,
            )
            future_to_type[f_reco] = "RECO"

        # 리뷰 데이터 저장용
        positive_df = pd.DataFrame()
        negative_df = pd.DataFrame()

        # 먼저 끝나는 순서대로 결과 처리
        for future in as_completed(future_to_type):
            task_type = future_to_type[future]

            try:
                result = future.result()

                if task_type == "REVIEW_POSITIVE":
                    positive_df = result
                    st.session_state["_rep_positive_reviews_df_cache"] = result

                elif task_type == "REVIEW_NEGATIVE":
                    negative_df = result
                    st.session_state["_rep_negative_reviews_df_cache"] = result

                elif task_type == "TREND":
                    st.session_state["_reviews_df_cache"] = result
                    st.session_state["_analysis_cache_product_id"] = str(product_id)
                    render_rating_trend(container_trend, result, skip_scroll_callback)

                elif task_type == "RECO":
                    reco_list = (
                        result
                        if isinstance(result, list)
                        else [item for items in result.values() for item in items]
                    )
                    st.session_state["reco_cache"] = reco_list
                    st.session_state["reco_target_product_id"] = product_id

            except Exception as e:
                if task_type == "REVIEW_POSITIVE":
                    st.session_state["_rep_positive_reviews_df_cache"] = pd.DataFrame()
                elif task_type == "REVIEW_NEGATIVE":
                    st.session_state["_rep_negative_reviews_df_cache"] = pd.DataFrame()
                elif task_type == "TREND":
                    with container_trend.container():
                        st.markdown("### 📈 평점 추이")
                        st.error(f"평점 추이 로드 실패: {e}")
                elif task_type == "RECO":
                    st.error(f"추천 상품 로드 실패: {e}")

        # 모든 리뷰가 로드된 후 렌더링
        st.session_state["_analysis_cache_product_id"] = str(product_id)
        render_representative_review(
            container_review, positive_df, negative_df, skip_scroll_callback
        )
