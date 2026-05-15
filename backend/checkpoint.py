from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect(database='rag-system.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)