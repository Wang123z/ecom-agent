"""配置加载测试：路径必须是绝对路径，数值必须转成正确类型。"""
from pathlib import Path

from app import config


def test_base_dir_is_project_root():
    assert (config.BASE_DIR / "app" / "config.py").exists()


def test_paths_are_absolute():
    for p in (config.MODEL_DIR, config.DATA_DIR, config.OUTPUT_DIR):
        assert isinstance(p, Path)
        assert p.is_absolute()


def test_numeric_types():
    assert isinstance(config.EMBEDDING_DIM, int)
    assert isinstance(config.RRF_K, int)
    assert isinstance(config.RERANK_LOW, float)
    assert isinstance(config.RATE_LIMIT_GLOBAL_QPS, float)


def test_rrf_weights_parsed():
    assert len(config.RRF_WEIGHTS) >= 2
    assert all(isinstance(w, float) for w in config.RRF_WEIGHTS)
