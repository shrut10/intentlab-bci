import pytest
from fastapi.testclient import TestClient

from intentlab.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client
