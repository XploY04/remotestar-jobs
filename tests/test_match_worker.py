"""Unit tests for the on-demand match worker's Redis resilience."""

import redis

from src.services import match_worker


class _DeadPubSub:
    """Pub/sub whose connection has dropped: get_message raises, mimicking a
    silently-closed Upstash connection surfacing on the next read."""

    def __init__(self):
        self.closed = False

    def get_message(self, timeout=None):
        raise redis.exceptions.ConnectionError("connection closed by server")

    def close(self):
        self.closed = True


class _FreshPubSub:
    def __init__(self):
        self.patterns = []

    def get_message(self, timeout=None):
        return None

    def psubscribe(self, pattern):
        self.patterns.append(pattern)

    def close(self):
        pass


class _FakeRedis:
    def __init__(self, fresh):
        self._fresh = fresh

    def pubsub(self):
        return self._fresh


def test_next_message_reconnects_on_connection_error():
    fresh = _FreshPubSub()
    dead = _DeadPubSub()

    message, pubsub = match_worker._next_message(dead, _FakeRedis(fresh))

    # The dead connection is closed and a new subscription is established.
    assert dead.closed is True
    assert pubsub is fresh
    assert "match:start:*" in fresh.patterns
    # The reconnect cycle itself yields no message.
    assert message is None


def test_next_message_passes_through_when_healthy():
    class _HealthyPubSub:
        def get_message(self, timeout=None):
            return {"type": "pmessage", "data": "{}"}

    ps = _HealthyPubSub()
    message, pubsub = match_worker._next_message(ps, _FakeRedis(_FreshPubSub()))

    assert pubsub is ps  # unchanged when the connection is healthy
    assert message["type"] == "pmessage"
