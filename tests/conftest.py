import fnmatch

import logging_db
import main
import pytest


class FakeAsyncRedis:
    """In-memory stand-in for the four Redis calls main.py makes (get/set/scan_iter/delete).
    Deliberately tiny: no extra dependency, and no version coupling to the redis client."""

    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def scan_iter(self, match="*"):
        for key in list(self.store):
            if fnmatch.fnmatch(key, match):
                yield key

    async def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Every test gets its own empty SQLite file, never the real predictions.db."""
    monkeypatch.setattr(logging_db, "DB_PATH", tmp_path / "test.db")
    logging_db.init_db()


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, "redis_client", FakeAsyncRedis())
    with TestClient(main.app) as test_client:
        yield test_client
