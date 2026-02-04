"""
검색창 컴포넌트
"""

import streamlit as st


def render_search_bar(
    product_options: list, on_clear_callback, filtered_products: list = None
):
    """
    검색창 렌더링

    Args:
        product_options: 전체 제품명 옵션 목록
        on_clear_callback: 초기화 버튼 클릭 시 콜백
        filtered_products: 왼쪽 검색으로 필터링된 제품 목록 (없으면 전체 목록 사용)

    Returns:
        selected_product: 선택된 제품명
    """
    with st.container(border=True):
        col_type, col_input, col_clear = st.columns(
            [2, 7, 1], vertical_alignment="bottom"
        )

        with col_type:
            search_type = st.selectbox(
                "검색 타입",
                options=["상품명", "키워드", "문맥"],
                key="search_type",
            )

        with col_input:
            if search_type == "상품명":
                selected = st.selectbox(
                    "검색어 입력",
                    options=[""] + product_options,
                    key="product_search",
                )
                st.session_state["search_keyword"] = selected
            
            else:
                placeholder_map = {
                    "키워드": "예: 수분, 진정, 저자극",
                    "문맥": "예: 건조한 피부에 잘 맞는 크림",
                }

                st.text_input(
                    "검색어 입력",
                    key="search_keyword",
                    placeholder=placeholder_map.get(search_type, ""),
                )


        with col_clear:
            st.button(
                "✕",
                help="검색 초기화",
                on_click=on_clear_callback,
            )

    return st.session_state.get("search_keyword", "")


def get_search_text() -> str:
    """현재 검색어 반환"""
    if st.session_state.get("product_search"):
        return st.session_state.product_search
    return st.session_state.get("search_keyword", "").strip()


def is_initial_state(selected_sub_cat: list, selected_skin: list) -> bool:
    """초기 상태인지 확인"""
    search_text = get_search_text()
    return not search_text and not selected_sub_cat and not selected_skin
