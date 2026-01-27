import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math
import css
import sys
import os
import re
import scroll
from load_data import load_raw_df, make_df, load_reviews, load_date_score, rating_trend
from sidebar import sidebar, product_filter
from recommend_similar_products import recommend_similar_products, print_recommendations
from pathlib import Path

st.cache_data.clear()

if "product_search" not in st.session_state:
    st.session_state["product_search"] = ""
if "search_keyword" not in st.session_state:
    st.session_state["search_keyword"] = ""
if "page" not in st.session_state:
    st.session_state.page = 1

sys.path.append(os.path.dirname(__file__))

st.set_page_config(layout="wide")

# 그래프 UI 조작 시, 이번 rerun에서는 스크롤 적용 스킵
if "_skip_scroll_apply_once" not in st.session_state:
    st.session_state["_skip_scroll_apply_once"] = False

def _skip_scroll_apply_once():
    st.session_state["_skip_scroll_apply_once"] = True

# 요청 시 상단 스크롤 이동 적용 (단, 그래프 조작 직후 1회는 스킵)
if not st.session_state.get("_skip_scroll_apply_once", False):
    scroll.apply_scroll_to_top_if_requested()
else:
    st.session_state["_skip_scroll_apply_once"] = False

def safe_scroll_to_top():
    scroll.request_scroll_to_top()

# ===== parquet 로딩 =====
base_dir = Path(__file__).resolve().parent
PRODUCTS_BASE_DIR = base_dir / "data" / "processed_data" / "integrated_products_final"
REVIEWS_BASE_DIR = base_dir / "data" / "processed_data" / "partitioned_reviews"

product_df = load_raw_df(PRODUCTS_BASE_DIR)
df = make_df(product_df)

# 키워드 문자열 컬럼 생성
df["top_keywords_str"] = df["top_keywords"].apply(lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else str(x))

skin_options = df["skin_type"].unique().tolist()
product_options = df["product_name"].unique().tolist()

# ===== 사이드바 =====
selected_sub_cat, selected_skin, min_rating, max_rating, min_price, max_price = sidebar(df)

# ===== 메인 =====
st.title("🎀 화장품 추천 대시보드")
st.markdown("---")

search_keyword = st.session_state.get("search_keyword", "")


# def on_search_change():
#     if "product_search" in st.session_state:
#         st.session_state["search_keyword"] = st.session_state["product_search"]


# 제품 선택 해제 버튼
def clear_selected_product():
    # 제품 선택, 검색 상태 초기화
    st.session_state["product_search"] = ""
    st.session_state["search_keyword"] = ""
    safe_scroll_to_top()


# selectbox 컨테이너 안으로 이동
with st.container(border=True):
    col_text, col_sel, col_clear = st.columns([5, 5, 1], vertical_alignment="bottom")

    with col_text:
        st.text_input(
            "🗝️키워드 검색",
            placeholder="예: 수분, 촉촉, 진정",
            key="search_keyword"
        )

    with col_sel:
        selected_product = st.selectbox(
            "🔎 제품명 검색",
            options=[""] + product_options,
            index=0,
            key="product_search",
            # on_change=on_search_change,  # 제품 선택 시 검색 상태 동기화
        )

    with col_clear:
        # 클릭 시 선택 제품 초기화
        st.button("✕", help="검색 초기화", 
                  on_click=lambda: (st.session_state.update({"product_search":"", "search_keyword":""}), safe_scroll_to_top()))


# 추천 상품 클릭
def select_product_from_reco(product_name: str):
    st.session_state["product_search"] = product_name
    st.session_state["search_keyword"] = product_name
    safe_scroll_to_top()


# 검색어로 사용할 값
# search_text = selected_product if selected_product else ""
if st.session_state.product_search:
    search_text = st.session_state.product_search
else:
    search_text = st.session_state.search_keyword.strip()

# 초기 상태 여부
is_initial = (not search_text and not selected_sub_cat and not selected_skin)

# ===== 인기상품 TOP 5 (리뷰 수, 평점 ) =====
if is_initial:
    st.markdown("## 🔥 인기 상품 TOP 5")

    popular_df = (
        df.sort_values(
            by=["total_reviews", "score"],
            ascending=[False, False]
        )
        .head(5)
        .reset_index(drop=True)
    )

    cols = st.columns(len(popular_df))

    for i, row in enumerate(popular_df.iterrows()):
        row = row[1]

        with cols[i]:
            with st.container(border=True):
                if row.get("image_url"):
                    st.image(row["image_url"], use_container_width=True, output_format="PNG")

                st.markdown(
                    f"""
                    <div style="font-size:14px;color:#888;margin-top:4px;">
                    {row.get('brand','')}
                    </div>
                    """, unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div style="font-size:13px;font-weight:500;line-height:1.3;margin:2px 0;">
                    {row['product_name']}
                    </div>
                    """, unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div style="font-size:14px;font-weight:700;">
                        ₩{int(row.get('price',0)):,}
                    </div>
                    </div>
                    """, unsafe_allow_html=True,
                )

                empty_col, btn_col = st.columns([7, 3], vertical_alignment="center")
                
                with btn_col:
                    st.button(
                        "선택",
                        key=f"reco_select_{st.session_state.page}_{i}",
                        on_click=select_product_from_reco,
                        args=(row["product_name"],),
                        use_container_width=True,
                    )

    st.markdown("---")



# 제품 정보
if selected_product:
    product_info = df[df["product_name"] == selected_product].iloc[0]

    st.subheader("🎁 선택한 제품 정보")
    col1, col2, col3 = st.columns(3)

    col1.metric("제품명", product_info["product_name"])
    col2.metric("브랜드", product_info.get("brand", ""))
    col3.metric("피부 타입", product_info.get("skin_type", ""))

    col4, col5, col6 = st.columns(3)
    col4.metric("가격", f"₩{int(product_info.get('price', 0)):,}")
    col5.metric("리뷰 수", f"{int(product_info.get('total_reviews', 0)):,}")
    col6.metric("카테고리", product_info.get("sub_category", ""))

    if product_info.get("product_url"):
        st.link_button("상품 페이지", product_info["product_url"])

    # 대표 키워드
    st.markdown("### 📃 대표 키워드")
    top_kw = product_info.get("top_keywords", "")
    if isinstance(top_kw, (list, np.ndarray)):
        top_kw = ", ".join(map(str, top_kw))
    st.write(top_kw if top_kw else "-")

    sub_cat = product_info.get("sub_category", "")

    # 대표 리뷰
    if selected_product:
        product_info = df[df["product_name"] == selected_product].iloc[0]
        product_id = product_info["product_id"]
        review_id = product_info["representative_review_id_roberta"]
        category = product_info["category"]
        
        text = load_reviews(product_id, review_id, category, REVIEWS_BASE_DIR)

    st.markdown("### ✒️ 대표 리뷰")

    if not text:
        st.info("대표 리뷰가 없습니다.")
    else:
        st.text(text)

    # 평점 추이 그래프  
    if selected_product:
        product_info = df[df["product_name"] == selected_product].iloc[0]
        product_id = product_info["product_id"]
        category = product_info["category"]
        
        review_df = load_date_score(product_id, category, REVIEWS_BASE_DIR)
        min_date = review_df["date"].min().date()
        max_date = review_df["date"].max().date()


    st.markdown("### 📈 평점 추이")
    col_left, col_mid, col_right, col_empty = st.columns([1, 1, 1, 1])

    # 집계 기준
    with col_left:
        freq_label = st.selectbox( "평균 기준", ["일간", "주간", "월간"], index=1, key="rating_freq_label", on_change=_skip_scroll_apply_once)

    freq_map = {"일간": ("D", 7), "주간": ("W", 4), "월간": ("M", 3)}
    freq, ma_window = freq_map[freq_label]

    DATE_RANGE_KEY = "rating_date_range"

    default_date_range = (min_date, max_date)

    with col_mid:
        date_range = st.date_input(
            "기간 선택",
            value=default_date_range,
            min_value=min_date,
            max_value=max_date,
            key=DATE_RANGE_KEY,
            on_change=_skip_scroll_apply_once,  # 그래프 조작 시 스크롤 apply 1회 스킵
        )

    def reset_date_range():
        _skip_scroll_apply_once()  # reset 클릭도 그래프 조작으로 간주
        st.session_state[DATE_RANGE_KEY] = (min_date, max_date)

    with col_right:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("↺", key="reset_date", help="날짜 초기화", on_click=reset_date_range)


    trend_df = pd.DataFrame()
    is_date_range_ready = False

    if isinstance(date_range, tuple) and len(date_range) == 2:
        is_date_range_ready = True
        start_date, end_date = date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        date_df = review_df.loc[(review_df["date"] >= start_date) & (review_df["date"] <= end_date)]

        if not date_df.empty:
            trend_df = rating_trend(date_df, freq=freq, ma_window=ma_window)

    else:
        st.info("마지막 날짜를 선택해주세요.📆")
        date_df = pd.DataFrame()

    if not is_date_range_ready:
        pass

    elif trend_df.empty:
        st.info("선택한 기간에 대한 평점 데이터가 없습니다.")

    else:
        fig = go.Figure()

        # 기간별 평균
        fig.add_trace(go.Bar(
            x=trend_df["date"], 
            y=trend_df["avg_score"], 
            name=f"{freq_label} 평균", 
            marker_color="slateblue", 
            opacity=0.4
            ))
        
        # 이동 평균
        fig.add_trace(go.Scatter(
            x=trend_df["date"], 
            y=trend_df["ma"], 
            mode="lines", 
            name=f"추세 ({ma_window}개{freq_label} 이동평균)", 
            line=dict(color="royalblue", width=3)
            ))
        
        fig.update_layout(
            yaxis=dict(range=[1, 5]),
            xaxis_title="날짜",
            yaxis_title="평균 평점",
            hovermode="x unified",
            template="plotly_white",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

# ===== 추천 페이지 =====
if not is_initial:
    if selected_product:
        st.subheader("👍 이 상품과 유사한 추천 상품")
    else:
        st.subheader("🌟 검색 결과")

if is_initial:
    st.info("왼쪽 사이드바 또는 검색어를 입력하여 상품을 찾아보세요.")
else:
    # 제품 필터링
    filtered_df = product_filter(df, search_text, selected_sub_cat, selected_skin, min_rating, max_rating, min_price, max_price)

    page_df = pd.DataFrame()
    reco_df_view = pd.DataFrame()
    search_df_view = filtered_df.copy()

    # 유사도 / 추천점수 기본값
    search_df_view["reco_score"] = 0.0
    search_df_view["similarity"] = 0.0

    badge_order = {"BEST": 0, "추천": 1, "": 2}
    search_df_view["badge_rank"] = (
    search_df_view["badge"].map(badge_order).fillna(2)
    )

    # 벡터 기반 추천 점수
    if selected_product:
        target_product = df[df["product_name"] == selected_product]

        if not target_product.empty:
            target_product_id = target_product.iloc[0]["product_id"]

            reco_results = recommend_similar_products(
                product_id=target_product_id,
                categories=None,
                top_n=100
            )

            reco_list = []
            for _, items in reco_results.items():
                reco_list.extend(items)

            if reco_list:
                tmp_reco_df = pd.DataFrame(reco_list)

                tmp_reco_df = tmp_reco_df.rename(columns={
                    "recommend_score": "reco_score",
                    "cosine_similarity": "similarity"
                })

                merged_df = df.merge(
                    tmp_reco_df[["product_id", "reco_score", "similarity"]],
                    on="product_id",
                    how="left"
                )

                merged_df["reco_score"] = merged_df["reco_score"].fillna(0)
                merged_df["similarity"] = merged_df["similarity"].fillna(0)

                merged_df = merged_df[merged_df["product_id"] != target_product_id]

                search_df_view = merged_df.copy()

                reco_df_view = (
                    merged_df
                    .query("reco_score > 0")
                    .query("product_id != @target_product_id")
                    .sort_values(
                        by=["reco_score", "similarity"],
                        ascending=[False, False]
                    )
                    .head(6)
                )


    # 페이지네이션
    items_page = 6
    total_items = len(filtered_df)
    total_pages = max(1, math.ceil(total_items / items_page))

    # 페이지 초기화
    if "page" not in st.session_state:
        st.session_state.page = 1

    st.session_state.page = min(st.session_state.page, total_pages)

    cur_filter = (search_text, tuple(selected_sub_cat), tuple(selected_skin), min_rating, max_rating, min_price, max_price)

    # 검색어/필터 변경시
    if st.session_state.get("prev_filter") != cur_filter:
        st.session_state.page = 1
        st.session_state.prev_filter = cur_filter
        safe_scroll_to_top()

    search_df_view = search_df_view.sort_values(
    by=["score", "total_reviews"],
    ascending=[False, False]
    )

    # 데이터 슬라이싱
    start = (st.session_state.page - 1) * items_page
    end = start + items_page
    if not selected_product:
        page_df = search_df_view.iloc[start:end]
    else:
        page_df = pd.DataFrame()



# 상품 출력
if (not is_initial) and (not selected_product) and page_df.empty:
    st.warning("표시할 상품이 없어요.🥺")
elif (not is_initial) and (not selected_product) and (not page_df.empty):
    rows = page_df.reset_index(drop=True)

    for i in range(0, len(rows), 2):
        cols = st.columns(2)

        for j in range(2):  # 한 줄에 2개씩 출력
            if i + j < len(rows):
                row = rows.iloc[i + j]

                with cols[j]:
                    with st.container(border=True):
                        col_image, col_info = st.columns([3, 7])
                        
                        with col_image:
                            st.image(row["image_url"], width=200)

                        with col_info:
                            badge_html = ""
                            if row.get("badge") == "BEST":
                                badge_html = "<span style='background:#ffea00;padding:2px 8px;border-radius:8px;font-size:12px;margin-left:8px;'>BEST</span>"
                            elif row.get("badge") == "추천":
                                badge_html = "<span style='background:#d1f0ff;padding:2px 8px;border-radius:8px;font-size:12px;margin-left:8px;'>추천</span>"

                            st.markdown(
                                f"""
                                <div style="font-size:14px;color:#888;">
                                {row.get('brand','')}
                                {badge_html}
                                </div>

                                <div style="font-size:18px;font-weight:600;margin:4px 0;">
                                {row['product_name']}
                                </div>

                                <div style="font-size:15px;color:#111;font-weight:500;">
                                ₩{int(row.get('price',0)):,}
                                </div>
                                
                                <div style="margin-top:6px;font-size:13px;color:#555;">
                                🏷️ 카테고리: {row.get('category_path_norm')}<br>
                                😊 피부 타입: {row.get('skin_type','')}<br>
                                ⭐ 평점: {row.get('score','')}<br>
                                💬 리뷰 수: {int(row.get('total_reviews',0)):,}
                                </div>
                                """, unsafe_allow_html=True,
                            )

                            empty_col, btn_col = st.columns([8, 2], vertical_alignment="center")
            
                            with btn_col:
                                st.button(
                                    "선택",
                                    key=f"reco_select_{st.session_state.page}_{i+j}",
                                    on_click=select_product_from_reco,
                                    args=(row["product_name"],),
                                    use_container_width=True,
                                )


# ===== 3. 추천 상품 출력 =====
if selected_product:
    if reco_df_view.empty:
        st.info("추천 가능한 유사 상품이 없어요.😥")
    else:
        rows = reco_df_view.reset_index(drop=True)

        for i in range(0, len(rows), 3):
            cols = st.columns(3)

            for j in range(3):
                if i + j < len(rows):
                    row = rows.iloc[i + j]

                    with cols[j]:
                        with st.container(border=True):
                            col_image, col_info = st.columns([3, 7])

                            with col_image:
                                if row.get("image_url"):
                                    st.image(row["image_url"], width=180)

                            with col_info:
                                st.markdown(
                                    f"""
                                    <div style="font-size:14px;color:#888;">
                                    {row.get('brand','')}
                                    </div>

                                    <div style="font-size:18px;font-weight:600;">
                                    {row['product_name']}
                                    </div>

                                    <div style="font-size:15px;font-weight:500;">
                                    ₩{int(row.get('price',0)):,}
                                    </div>

                                    <div style="margin-top:6px;font-size:13px;color:#555;">
                                    🔗 유사도: {row['similarity']:.3f}<br>
                                    ⭐ 추천 점수: {row['reco_score']:.3f}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                st.button(
                                    "선택",
                                    key=f"reco_only_{row['product_id']}",
                                    on_click=select_product_from_reco,
                                    args=(row["product_name"],),
                                    use_container_width=True,
                                )


show_pagination = (
    selected_product
    or selected_sub_cat
)

# 페이지 이동 버튼
if show_pagination and total_pages > 1:
    st.markdown("---")

    col_prev, col_info, col_next = st.columns([1, 2, 1])

    def go_prev():
        if st.session_state.page > 1:
            st.session_state.page -= 1
            safe_scroll_to_top()

    def go_next():
        if st.session_state.page < total_pages:
            st.session_state.page += 1
            safe_scroll_to_top()

    with col_prev:
        st.button("이전", key="prev_page", on_click=go_prev)

    with col_next:
        st.button("다음", key="next_page", on_click=go_next)

    with col_info:
        st.markdown(
            f"<div style='text-align:center; font-weight:bold;'>"
            f"{st.session_state.page} / {total_pages} 페이지"
            f"</div>",
            unsafe_allow_html=True
        )

css.set_css()
