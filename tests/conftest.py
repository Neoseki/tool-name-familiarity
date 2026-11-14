import pytest

from src import bfcl_io
from src.models import MODELS
from src.render import load_tokenizer

# multi_turn_base_0: file system + Twitter; _1: file system only; _2: tickets + file system.
DEV_IDS = ["multi_turn_base_0", "multi_turn_base_1", "multi_turn_base_2"]


@pytest.fixture(scope="session")
def entries():
    return bfcl_io.load_entries(DEV_IDS)


@pytest.fixture(scope="session")
def gts():
    return bfcl_io.load_ground_truth(DEV_IDS)


@pytest.fixture(scope="session")
def qwen_cfg():
    return MODELS["qwen3-local"]


@pytest.fixture(scope="session")
def qwen_tok(qwen_cfg):
    return load_tokenizer(qwen_cfg)
