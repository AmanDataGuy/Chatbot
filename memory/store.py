import psycopg
from langgraph.store.postgres import PostgresStore
from config import DB_URI


def get_store():
    conn = psycopg.connect(DB_URI, autocommit=True)
    store = PostgresStore(conn)
    store.setup()
    return store