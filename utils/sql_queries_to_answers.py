import pandas as pd
import pathlib
import sqlite3
from data_prep import proc_file

DATA_DIR = pathlib.Path("./data/for_sql_gtlabels")
DB_DIR = pathlib.Path("/System/Volumes/Data/Users/dsp/code/school/sqlova/data_and_model/")

train_path = DATA_DIR / "train_tok_cypher.txt"
train_db = DB_DIR / "train.db"

eval_path = DATA_DIR / "dev_tok_cypher.txt"
eval_db = DB_DIR / "dev.db"

test_path = DATA_DIR / "test_tok_cypher.txt"
test_db = DB_DIR / "test.db"

input_data_bundle = zip(*[
    [train_path, eval_path, test_path],
    [train_db, eval_db, test_db]
])

for data_path, db_path in input_data_bundle:
    _, _, sql_queries = proc_file(file_path=data_path, version=4)
    db = sqlite3.connect(db_path)
    c = db.cursor()

    out = {
        "query": [],
        "answer": []
    }

    failed_count = 0
    failed_with_warning = 0
    for sql_query in sql_queries:

        try:
            ans = c.execute(sql_query).fetchone()
            out["query"].append(sql_query)
            out["answer"].append(ans if ans is None else ans[0])
        except sqlite3.OperationalError:
            failed_count += 1
        except sqlite3.Warning:
            failed_with_warning += 1

    print(f'{failed_count}/{len(sql_queries)} failed because of incorrect input')
    print(f'{failed_with_warning}/{len(sql_queries)} failed because of exe')
    pd.DataFrame(out).to_csv(str(data_path) + ".csv", index=False)

    db.close()
