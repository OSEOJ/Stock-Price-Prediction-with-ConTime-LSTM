"""
프로젝트 공통 유틸리티
"""
from pathlib import Path


def get_project_root() -> Path:
    """프로젝트 루트 디렉토리를 반환합니다."""
    return Path(__file__).parent.parent


def ensure_directory(directory_path) -> Path:
    """디렉토리가 존재하지 않으면 생성합니다."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    return Path(directory_path)
