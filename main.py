import streamlit as st
import pandas as pd
import math
import random # 임시
import scroll
from load_data import load_raw_df, make_df
from sidebar import sidebar, product_filter
import css
from pathlib import Path
import sys
import os
import re

if "product_search" not in st.session_state:
    st.session_state["product_search"] = ""
if "search_keyword" not in st.session_state:
    st.session_state["search_keyword"] = ""

sys.path.append(os.path.dirname(__file__))

st.set_page_config(layout="wide")

# 요청 시 상단 스크롤 이동 적용
scroll.apply_scroll_to_top_if_requested()

# ===== parquet 로딩 =====
base_dir = Path(__file__).resolve().parent
PRODUCTS_BASE_DIR = base_dir / "data" / "integrated_products_final"

raw_df = load_raw_df(PRODUCTS_BASE_DIR)  

df = make_df(raw_df)                      

skin_options = df["skin_type"].unique().tolist()
product_options = df["product_name"].unique().tolist()


# ===== 리뷰 로딩: partitioned_reviews/category=XXX/review.parquet 사용 =====
REVIEWS_BASE_DIR = base_dir / "data" / "partitioned_reviews"


# 디버그 출력 제거

def _candidate_category_dirs(sub_category: str):
    """
    sub_category 값과 실제 폴더명(category=...)이 불일치할 수 있어 후보를 여러 개 만들어서 탐색.
    예) "틴트 립글로스" -> "틴트_립글로스"
    """
    if not sub_category:
        return []

    s = str(sub_category).strip()
    if not s:
        return []

    cands = [s]

    # 공백/슬래시/구분자 -> 언더스코어
    s2 = s.replace(" / ", "_").replace("/", "_").replace(" ", "_").replace(">", "_").replace("|", "_")
    cands.append(s2)

    # 연속 언더스코어 정리
    s3 = re.sub(r"_+", "_", s2).strip("_")
    cands.append(s3)

    # 이미 category= 접두가 들어오는 경우도 방어
    out = []
    for x in cands:
        x = x.replace("category=", "").strip()
        if x and x not in out:
            out.append(x)
    return out

@st.cache_data(show_spinner=False)
def load_reviews_by_subcategory(sub_category: str):
    """
    data/partitioned_reviews/category=<sub_category>/review.parquet 로드
    - 반환: (DataFrame, found_path or None)
    """
    if not sub_category:
        return pd.DataFrame(), None

    for cand in _candidate_category_dirs(sub_category):
        review_path = REVIEWS_BASE_DIR / f"category={cand}" / "review.parquet"
        if review_path.exists():
            rdf = pd.read_parquet(review_path)

            # review_text 컬럼명 표준화 (혹시 다른 이름이면)
            if "review_text" not in rdf.columns:
                for col in ["full_text", "content", "text", "review", "review_content"]:
                    if col in rdf.columns:
                        rdf = rdf.rename(columns={col: "review_text"})
                        break

            # 최소 컬럼 체크
            if "product_id" not in rdf.columns or "review_text" not in rdf.columns:
                return pd.DataFrame(), str(review_path)

            return rdf, str(review_path)

    return pd.DataFrame(), None


# ===== 사이드바 =====
selected_sub_cat, selected_skin, min_rating, max_rating, min_price, max_price = sidebar(df)

# ===== 메인 =====
st.title("화장품 추천 대시보드")
st.subheader("제품명 검색")

search_keyword = st.session_state.get("search_keyword", "")


def on_search_change():
    if "product_search" in st.session_state:
        st.session_state["search_keyword"] = st.session_state["product_search"]


# 제품 선택 해제 버튼
def clear_selected_product():
    # 제품 선택, 검색 상태 초기화
    st.session_state["product_search"] = ""
    st.session_state["search_keyword"] = ""
    scroll.request_scroll_to_top()


# selectbox 컨테이너 안으로 이동
with st.container(border=True):
    col_sel, col_clear = st.columns([10, 1], vertical_alignment="bottom")

    with col_sel:
        selected_product = st.selectbox(
            "제품명을 입력하거나 선택하세요",
            options=[""] + product_options,
            index=0,
            key="product_search",
            on_change=on_search_change,  # 제품 선택 시 검색 상태 동기화
        )

    with col_clear:
        # 클릭 시 선택 제품 초기화
        st.button("✕", key="clear_product", help="선택 해제", on_click=clear_selected_product)


# 추천 상품 클릭
def select_product_from_reco(product_name: str):
    st.session_state["product_search"] = product_name
    st.session_state["search_keyword"] = product_name
    scroll.request_scroll_to_top()


# 검색어로 사용할 값
search_text = selected_product if selected_product else ""

# 초기 상태 여부
is_initial = (not search_text and not selected_sub_cat and not selected_skin)


def rep_ids_to_list(rep_ids, n=3):
    """representative_review_id가 list/str/단일값 어떤 형태든 n개로 정규화"""
    if rep_ids is None or (isinstance(rep_ids, float) and pd.isna(rep_ids)):
        return []
    if isinstance(rep_ids, list):
        return rep_ids[:n]
    if isinstance(rep_ids, str):
        return [x.strip() for x in re.split(r"[;,]", rep_ids) if x.strip()][:n]
    return [rep_ids][:n]


# 제품 정보
if selected_product:
    product_info = df[df["product_name"] == selected_product].iloc[0]

    st.subheader("선택한 제품 정보")
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

    # ===== 요청사항: 대표 긍정 키워드 + 대표 리뷰 =====
    # 대표 긍정 키워드
    st.markdown("### 대표 긍정 키워드")
    top_kw = product_info.get("top_keywords", "")
    if isinstance(top_kw, list):
        top_kw = ", ".join(top_kw)
    st.write(top_kw if top_kw else "-")

    sub_cat = product_info.get("sub_category", "")

    # 디버그 출력 제거
    # 대표 리뷰
    st.markdown("### 대표 리뷰")

    # category_path_norm 마지막 토큰을 리뷰 파티션 카테고리로 사용 (폴더명 불일치 대응)
    cat_norm = product_info.get("category_path_norm", "")
    if isinstance(cat_norm, str) and ">" in cat_norm:
        review_cat = cat_norm.split(">")[-1].strip()
    else:
        review_cat = product_info.get("sub_category", "")

    reviews_df, found_path = load_reviews_by_subcategory(review_cat)

    # 디버그 출력 제거 + 해당 카테고리 리뷰 파일이 없으면 안내 후 스킵
    if not found_path:
        st.info("현재 제공된 리뷰 데이터(partitioned_reviews)에 해당 카테고리 리뷰 파일이 없습니다.")
    elif reviews_df.empty:
        st.info("대표 리뷰를 불러올 수 없습니다.")
    else:
        q = reviews_df[reviews_df["product_id"] == product_info["product_id"]]

        if q.empty:
            # 데이터 범위/ID 체계 차이로 상품 단위 매칭이 안 될 수 있음
            st.info("해당 상품의 리뷰가 없습니다. (현재 리뷰 데이터는 일부 카테고리만 제공될 수 있습니다.)")
        else:
            # 점수 있으면 높은 순
            if "score" in q.columns:
                q = q.sort_values("score", ascending=False)

            texts = q["review_text"].dropna().head(3).tolist()

            if not texts:
                st.info("대표 리뷰가 비어 있습니다.")
            else:
                for i, t in enumerate(texts, 1):
                    st.write(f"{i}. {t}")
# ===== 추천 페이지 =====
st.subheader("추천 상품")

if is_initial:
    st.info("왼쪽 사이드바 또는 검색어를 입력하여 상품을 찾아보세요.")
else:
    # 제품 필터링
    filtered_df = product_filter(df, search_text, selected_sub_cat, selected_skin, min_rating, max_rating, min_price, max_price)

    badge_order = {"BEST": 0, "추천": 1, "": 2}
    filtered_df["badge_rank"] = filtered_df["badge"].map(badge_order).fillna(2)

    filtered_df = filtered_df.sort_values(by=["badge_rank", "score", "total_reviews"], ascending=[True, False, False])

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
        # 필터 변경 시에도 상단으로 이동
        scroll.request_scroll_to_top()

    # 데이터 슬라이싱
    start = (st.session_state.page - 1) * items_page
    end = start + items_page
    page_df = filtered_df.iloc[start:end]

    # 추천 상품 출력
    if page_df.empty:
        st.warning("조건에 맞는 상품이 없습니다.")
    else:
        for i, row in page_df.reset_index(drop=True).iterrows():
            col_btn, col_card, col_space = st.columns([1.2, 6.5, 2.3])  # 레이아웃
            # 카드 컨테이너 안에서 버튼, 내용 배치
            with col_card:
                with st.container(border=True):

                    # 오른쪽 선택 버튼
                    top_left, top_right = st.columns([8, 2], vertical_alignment="center")
                    with top_right:
                        st.button(
                            "선택",
                            key=f"reco_select_{st.session_state.page}_{i}",
                            on_click=select_product_from_reco,
                            args=(row["product_name"],),
                            use_container_width=True,  # 버튼이 컬럼 폭을 꽉 채움
                        )

                    # 카드형 UI
                    col_image, col_info = st.columns([3, 7])

                    with col_image:
                        # 제품 이미지
                        st.image(row["image_url"], width=200)

                    with col_info:
                        badge_html = ""
                        if row.get("badge") == "BEST":
                            badge_html = "<span style='background:#ffea00;padding:2px 8px;border-radius:8px;font-size:12px;margin-left:8px;'>BEST</span>"
                        if row.get("badge") == "추천":
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
                                카테고리: {row.get('category_path_norm')}<br>
                                피부 타입: {row.get('skin_type','')}<br>
                                평점: {row.get('score','')}<br>
                                리뷰 수: {int(row.get('total_reviews',0)):,}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if row.get("product_url"):
                            st.link_button("상품 페이지", row["product_url"])

    # 페이지 이동 버튼
    st.markdown("---")

    col_prev, col_info, col_next = st.columns([1, 2, 1])

    # 이전 페이지 이동 콜백 함수
    def go_prev():
        if st.session_state.page > 1:
            st.session_state.page -= 1
            scroll.request_scroll_to_top() 

    # 다음 페이지 이동 콜백 함수
    def go_next():
        if st.session_state.page < total_pages:
            st.session_state.page += 1
            scroll.request_scroll_to_top()

    with col_prev:
        # on_click 콜백 방식으로 변경
        st.button("이전", key="prev_page", on_click=go_prev)

    with col_next:
        # on_click 콜백 방식으로 변경
        st.button("다음", key="next_page", on_click=go_next)

    with col_info:
        st.markdown(
            f"<div style='text-align:center; font-weight:bold;'>"
            f"{st.session_state.page} / {total_pages} 페이지"
            f"</div>",
            unsafe_allow_html=True
        )

css.set_css()
