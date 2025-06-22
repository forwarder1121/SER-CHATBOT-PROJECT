"""
src 패키지 초기화 파일
- .env + Streamlit secrets를 한 번에 로드하는 load_env() 제공
- Side-effect(즉시 실행 코드)는 넣지 않습니다
"""

from __future__ import annotations

import os
from typing import Iterable

import streamlit as st
from dotenv import load_dotenv, find_dotenv


# ▶ 프로젝트 전역에서 요구하는 필수 키
_REQUIRED_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_ENVIRONMENT",
    "PINECONE_INDEX_NAME",
)


def _print_env(keys: Iterable[str]) -> None:
    print("\n=== Environment Snapshot ===")
    for k in keys:
        print(f"{k}: {'Present' if os.getenv(k) else 'Missing'}")
    print("=== End Snapshot ===\n")


def load_env(*, disable_ssl: bool = False) -> None:
    """
    1) .env 로드
    2) Streamlit secrets → OS 환경변수
    3) (선택) SSL 검증 비활성화
    4) 필수 변수 존재 확인

    raise RuntimeError  : 필수 키 누락 시
    """
    # ① .env
    load_dotenv(find_dotenv())

    # ② st.secrets
    if "secrets" in st.__dict__:
        for k, v in st.secrets.items():
            os.environ.setdefault(k, v)

    # ③ SSL (옵션)
    if disable_ssl:
        os.environ["CURL_CA_BUNDLE"] = ""

    # ④ 필수 키 체크
    missing = [k for k in _REQUIRED_VARS if not os.getenv(k)]
    _print_env(_REQUIRED_VARS)
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


__all__ = ["load_env"]
