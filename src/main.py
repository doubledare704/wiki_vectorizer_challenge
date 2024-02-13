from logging.config import dictConfig
from typing import Dict

from fastapi import FastAPI

from src.utils.dto import SearchByQueryResult, SearchResultList
from src.utils.log_config import LogConfig
from src.utils.populate_db import store_chunks_in_db, search

dictConfig(LogConfig().dict())
app = FastAPI(debug=True)


@app.get("/populate-db")
async def populate_neo4j() -> Dict:
    store_chunks_in_db()
    return {"status": "Neo4j db was populated with wiki pages."}


@app.get("/search")
async def search_in_embeddings(search_q: str):
    # # Step 5: Search
    top_results = search(search_q)

    data = []
    for res in top_results:
        ident, similarity, arr = res
        o = SearchByQueryResult(
            id=ident,
            similarity=similarity,
            vector=arr.tolist(),
        )
        data.append(o)
    return SearchResultList(data=data)
