import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from config import DB_URI


def get_checkpointer():
    conn = psycopg.connect(DB_URI, autocommit=True)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    return checkpointer