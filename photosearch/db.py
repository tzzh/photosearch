import sqlite3
import io
import numpy as np

#https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

def connect():
    return  sqlite3.connect("./data.db", detect_types=sqlite3.PARSE_DECLTYPES)
