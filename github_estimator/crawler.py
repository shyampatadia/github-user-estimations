"""Async API interaction for fetching GitHub user data."""

import asyncio
import logging
import time
from typing import Optional

import aiohttp

from . import config, database
from .token_manager import TokenRotator

logger = logging.getLogger(__name__)


def parse_user_response(data: dict) -> dict:
    return {
        "login": data.get("login"),
        "type": data.get("type"),
        "public_repos": data.get("public_repos"),
        "public_gists": data.get("public_gists"),
        "followers": data.get("followers"),
        "following": data.get("following"),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
    }


async def fetch_user(
    session: aiohttp.ClientSession,
    user_id: int,
    token: dict,
    rotator: TokenRotator,
) -> dict:
    url = f"{config.API_BASE_URL}/{user_id}"
    headers = {
        "Authorization": f"token {token['token']}",
        "Accept": "application/vnd.github.v3+json",
    }
    start = time.monotonic()
    try:
        async with session.get(url, headers=headers) as resp:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            rotator.update_from_headers(token["id"], dict(resp.headers))

            if resp.status == 200:
                data = await resp.json()
                parsed = parse_user_response(data)
                return {
                    "github_id": user_id,
                    "is_valid": True,
                    "http_status": 200,
                    "response_time_ms": elapsed_ms,
                    "token_id": token["id"],
                    **parsed,
                }
            elif resp.status == 404:
                return {
                    "github_id": user_id,
                    "is_valid": False,
                    "http_status": 404,
                    "response_time_ms": elapsed_ms,
                    "token_id": token["id"],
                    "login": None, "type": None,
                    "public_repos": None, "public_gists": None,
                    "followers": None, "following": None,
                    "created_at": None, "updated_at": None,
                }
            elif resp.status in (401, 403):
                logger.warning(f"Auth/rate error ({resp.status}) on token {token['id']} for user {user_id}")
                rotator.update_from_headers(token["id"], dict(resp.headers))
                return None  # Signal retry with different token
            else:
                logger.error(f"Unexpected status {resp.status} for user {user_id}")
                return {
                    "github_id": user_id,
                    "is_valid": False,
                    "http_status": resp.status,
                    "response_time_ms": elapsed_ms,
                    "token_id": token["id"],
                    "login": None, "type": None,
                    "public_repos": None, "public_gists": None,
                    "followers": None, "following": None,
                    "created_at": None, "updated_at": None,
                }
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.error(f"Request error for user {user_id}: {e}")
        return None


async def fetch_batch(
    user_ids: list[int],
    rotator: TokenRotator,
    stratum_id: str = None,
    run_id: str = None,
    progress_callback=None,
) -> list[dict]:
    """Fetch a batch of user IDs with concurrency control and retries."""
    semaphore = asyncio.Semaphore(config.CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=config.CONCURRENT_REQUESTS)
    timeout = aiohttp.ClientTimeout(total=30)
    results = []
    retry_queue = list(user_ids)
    max_retries = 3

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for attempt in range(max_retries):
            if not retry_queue:
                break

            current_batch = retry_queue[:]
            retry_queue = []
            tasks = []

            async def _fetch_one(uid):
                async with semaphore:
                    token = await rotator.get_token()
                    result = await fetch_user(session, uid, token, rotator)
                    if result is None:
                        retry_queue.append(uid)
                    else:
                        if stratum_id:
                            result["stratum_id"] = stratum_id
                        if run_id:
                            result["run_id"] = run_id
                        results.append(result)
                        if progress_callback:
                            progress_callback(1)

            tasks = [_fetch_one(uid) for uid in current_batch]
            await asyncio.gather(*tasks)

            if retry_queue:
                logger.info(f"Retrying {len(retry_queue)} failed requests (attempt {attempt + 2})")
                await asyncio.sleep(2 ** attempt)

    return results


async def collect_ground_truth(
    stratum_id: str,
    stratum_config: dict,
    db_conn,
    rotator: TokenRotator,
    progress_callback=None,
):
    """Collect ground truth for a validation stratum (every ID in range)."""
    start_id = stratum_config["start"]
    end_id = stratum_config["end"]

    # Check which IDs are already collected
    already_collected = database.get_collected_ids_for_stratum(db_conn, stratum_id)
    all_ids = list(range(start_id, end_id + 1))
    remaining_ids = [uid for uid in all_ids if uid not in already_collected]

    if not remaining_ids:
        logger.info(f"Stratum {stratum_id}: All {len(all_ids)} IDs already collected")
        return

    logger.info(f"Stratum {stratum_id}: Collecting {len(remaining_ids)} remaining IDs "
                f"({len(already_collected)} already done)")

    # Process in batches of 1000 for checkpointing
    batch_size = 1000
    for i in range(0, len(remaining_ids), batch_size):
        batch_ids = remaining_ids[i:i + batch_size]
        results = await fetch_batch(
            batch_ids, rotator, stratum_id=stratum_id, progress_callback=progress_callback
        )

        # Save to database as ground truth
        records = []
        for r in results:
            records.append({
                "github_id": r["github_id"],
                "is_valid": r["is_valid"],
                "login": r["login"],
                "type": r["type"],
                "public_repos": r["public_repos"],
                "public_gists": r["public_gists"],
                "followers": r["followers"],
                "following": r["following"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "stratum_id": stratum_id,
                "token_id": r["token_id"],
            })
        database.insert_ground_truth_batch(db_conn, records)

    valid_count = database.get_ground_truth_valid_count(db_conn, stratum_id)
    total_count = database.get_ground_truth_count(db_conn, stratum_id)
    rate = valid_count / total_count if total_count > 0 else 0
    logger.info(f"Stratum {stratum_id}: {valid_count}/{total_count} valid ({rate:.1%})")


async def collect_random_sample(
    user_ids: list[int],
    run_id: str,
    stratum_id: str,
    db_conn,
    rotator: TokenRotator,
    progress_callback=None,
) -> list[dict]:
    """Collect a random sample of user IDs for estimation."""
    results = await fetch_batch(
        user_ids, rotator, stratum_id=stratum_id, run_id=run_id,
        progress_callback=progress_callback,
    )

    # Save to samples table
    records = []
    for r in results:
        records.append({
            "github_id": r["github_id"],
            "is_valid": r["is_valid"],
            "login": r.get("login"),
            "type": r.get("type"),
            "public_repos": r.get("public_repos"),
            "public_gists": r.get("public_gists"),
            "followers": r.get("followers"),
            "following": r.get("following"),
            "created_at": r.get("created_at"),
            "updated_at": r.get("updated_at"),
            "run_id": run_id,
            "stratum_id": stratum_id,
            "token_id": r.get("token_id"),
            "response_time_ms": r.get("response_time_ms"),
            "http_status": r.get("http_status"),
        })
    database.insert_samples_batch(db_conn, records)

    return results


async def find_frontier(rotator: TokenRotator) -> int:
    """Binary search for the current maximum valid GitHub user ID."""
    low = config.FRONTIER_M - 5_000_000
    high = config.FRONTIER_M + 5_000_000

    connector = aiohttp.TCPConnector(limit=5, ssl=False)
    timeout = aiohttp.ClientTimeout(total=15)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while low < high - 1:
            mid = (low + high) // 2
            token = await rotator.get_token()
            result = await fetch_user(session, mid, token, rotator)
            if result and result["http_status"] == 404:
                # Check a few IDs around mid to be sure
                high = mid
            elif result and result["is_valid"]:
                low = mid
            else:
                high = mid

    logger.info(f"Frontier found at approximately {low}")
    return low
