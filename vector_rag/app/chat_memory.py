"""Optional versioned Redis memory; SQL remains authoritative."""
import json
import logging
import os

from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


async def recent_messages(repository, conversation):
    client = None
    # Versioned keys prevent an old request or failed invalidation serving stale memory.
    key = f"conversation:{conversation['id']}:recent:{conversation['message_count']}"
    try:
        if os.getenv("REDIS_URL"):
            from redis.asyncio import Redis
            client = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True,
                                    socket_connect_timeout=0.3, socket_timeout=0.3)
            cached = await client.get(key)
            if cached:
                rows = json.loads(cached)
                if isinstance(rows, list) and all(isinstance(row, dict) and
                        row.get("role") in ("user", "assistant") and
                        isinstance(row.get("content"), str) for row in rows):
                    return rows[-6:]
    except Exception:
        logger.warning("recent_cache_read_failed; using SQL")
    finally:
        if client:
            try:
                await client.aclose()
            except Exception:
                logger.warning("recent_cache_close_failed")
    rows = await run_in_threadpool(repository.messages, conversation["id"], 6)
    if os.getenv("REDIS_URL"):
        try:
            from redis.asyncio import Redis
            async with Redis.from_url(os.environ["REDIS_URL"], decode_responses=True,
                                      socket_connect_timeout=0.3, socket_timeout=0.3) as cache:
                await cache.set(key, json.dumps(rows), ex=1800)
        except Exception:
            logger.warning("recent_cache_write_failed; continuing with SQL")
    return rows
