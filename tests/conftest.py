import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("SDXAI_VERSION", "mini")


@pytest.fixture(scope="session")
def runner():
    from sdxai.runner import SDRunner
    return SDRunner(version="mini")
