# streamlit_app.py
"""
Streamlit 엔트리포인트.
1) 환경변수 로드 → 필수 키 검증
2) 오류 있으면 Streamlit 화면에 표시 후 종료
3) 문제 없으면 src.app.main.main() 실행
"""

from __future__ import annotations

import os
import streamlit as st
from typing import Callable

from src import load_env 
from src.app.main import main as app_main


# ─────────────────────────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────────────────────────
def bootstrap(load_env_fn: Callable[..., None]) -> None:
    """환경 로드 & 검증 → 실패 시 Streamlit 에러 출력 후 stop."""
    try:
        load_env_fn(disable_ssl=False)       # 필요 시 True
        print("Environment ready - launching app\n")
    except RuntimeError as err:
        st.error(str(err))
        st.stop()


# ─────────────────────────────────────────────────────────────
# 실행 구간
# ─────────────────────────────────────────────────────────────
def main() -> None:
    """Streamlit 런처."""
    bootstrap(load_env)
    app_main()               # 실제 애플리케이션 진입점


if __name__ == "__main__":
    main()
