"""Redis Pub/Sub worker for on-demand job matching.

Subscribes to match:start:* channels. When a message arrives,
runs the matching pipeline for that user and publishes progress
events to match:progress:{user_id}.

Run with: python -m src.main --worker
"""

from __future__ import annotations

import asyncio
import json

import redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from src.database.operations import db
from src.services.matcher import MatchingUnavailableError, run_matching_for_user
from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _subscribe(redis_client: redis.Redis):
    """Create a pub/sub handle subscribed to match:start:*."""
    pubsub = redis_client.pubsub()
    pubsub.psubscribe("match:start:*")
    return pubsub


def _next_message(pubsub, redis_client: redis.Redis):
    """Read the next pub/sub message, reconnecting if the connection dropped.

    Upstash (like most managed Redis) closes idle pub/sub connections. redis-py
    surfaces that as a ConnectionError on the next read. Without this the worker
    sits on a dead connection receiving nothing while still reporting healthy.
    Returns (message_or_None, pubsub); a reconnect yields a fresh pubsub.
    """
    try:
        return pubsub.get_message(timeout=1.0), pubsub
    except (RedisConnectionError, RedisTimeoutError) as err:
        logger.warning("Redis pub/sub connection lost (%s); reconnecting...", err)
        try:
            pubsub.close()
        except Exception:
            pass
        pubsub = _subscribe(redis_client)
        logger.info("Re-subscribed to match:start:* after reconnect")
        return None, pubsub


async def start_worker() -> None:
    """Main worker loop: subscribe to Redis and process match requests."""

    if not settings.redis_url:
        logger.error("REDIS_URL not configured, cannot start worker")
        return

    await db.connect()
    logger.info("Worker connected to MongoDB")

    redis_client = redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        health_check_interval=30,
        socket_keepalive=True,
    )
    redis_client.ping()
    logger.info("Worker connected to Redis")

    pubsub = _subscribe(redis_client)
    logger.info("Worker listening for match requests on match:start:*")

    try:
        while True:
            message, pubsub = _next_message(pubsub, redis_client)
            if message and message["type"] == "pmessage":
                user_id = None
                try:
                    data = json.loads(message["data"])
                    user_id = data.get("user_id")
                    if not user_id:
                        logger.warning("Received match:start without user_id: %s", data)
                        continue

                    logger.info("Match request received for user %s", user_id)
                    await run_matching_for_user(user_id, run_ai=True, redis_client=redis_client)

                except json.JSONDecodeError:
                    logger.error("Invalid JSON in match:start message: %s", message["data"])
                except Exception as e:
                    if user_id:
                        # Never leak raw provider errors to the client. Known
                        # transient failures carry their own friendly message;
                        # everything else gets a generic one. Full detail is
                        # still logged below.
                        message = (
                            str(e)
                            if isinstance(e, MatchingUnavailableError)
                            else "Something went wrong while matching. Please try again."
                        )
                        error_event = json.dumps({"stage": "error", "message": message})
                        redis_client.publish(f"match:progress:{user_id}", error_event)
                    logger.error("Error processing match request: %s", e, exc_info=True)
                finally:
                    # Always clear the matching lock so the user can retry
                    if user_id:
                        try:
                            redis_client.delete(f"matching:lock:{user_id}")
                            logger.info("Cleared matching lock for user %s", user_id)
                        except Exception as lock_err:
                            logger.error("Failed to clear lock for %s: %s", user_id, lock_err)

            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Worker shutting down")
    finally:
        pubsub.unsubscribe()
        redis_client.close()
        await db.disconnect()
