from fastapi.testclient import TestClient
from ..src.main import app

client = TestClient(app)

def test_api_locally_get_root():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {'greeting': 'Hi, what about making an inference?'}