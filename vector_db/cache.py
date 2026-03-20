import sqlite3
import hashlib
import os

from config.settings import DB_DIR

CACHE_DB = os.path.join(DB_DIR, "semantic_cache.db")

def _get_hash(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode('utf-8')).hexdigest()

def init_cache():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cache
                 (query_hash TEXT PRIMARY KEY, response TEXT)''')
    conn.commit()
    conn.close()

def get_cached_response(query: str) -> str | None:
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute("SELECT response FROM cache WHERE query_hash=?", (_get_hash(query),))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def set_cached_response(query: str, response: str):
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO cache (query_hash, response) VALUES (?, ?)", 
              (_get_hash(query), response))
    conn.commit()
    conn.close()

# Initialize cache table on import
init_cache()
