"""Delete Adzuna jobs and their derived records from the configured environment.

Dry-run is the default. Pass --execute after checking the printed database,
Pinecone index, namespace, and document count.
"""

from __future__ import annotations

import argparse
import asyncio

from pinecone import Pinecone

from src.database.operations import db
from src.utils.config import settings


SOURCE = "adzuna"
BATCH_SIZE = 500


async def remove(execute: bool) -> None:
    await db.connect()
    try:
        query = {"source": SOURCE}
        count = await db.jobs.count_documents(query)
        index_name = settings.pinecone_index or "remotestar"
        namespace = settings.pinecone_namespace or "jobs-pool"
        print(f"database={db.db.name} index={index_name} namespace={namespace} adzuna_jobs={count}")
        if not execute or count == 0:
            print("dry-run only; pass --execute to delete" if count else "nothing to delete")
            return
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is required before deleting MongoDB rows")

        index = Pinecone(api_key=settings.pinecone_api_key).Index(index_name)
        removed = 0
        while True:
            rows = await db.jobs.find(query, {"_id": 1}).limit(BATCH_SIZE).to_list(length=BATCH_SIZE)
            if not rows:
                break
            ids = [row["_id"] for row in rows]
            index.delete(ids=ids, namespace=namespace)
            if db.db is not None:
                await db.db["job_matches"].delete_many({"job_id": {"$in": ids}})
            result = await db.jobs.delete_many({"_id": {"$in": ids}, "source": SOURCE})
            removed += result.deleted_count
            print(f"removed={removed}/{count}")
        print(f"complete removed={removed}")
    finally:
        await db.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    asyncio.run(remove(args.execute))
