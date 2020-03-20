import pathlib
import json
from ipdb import launch_ipdb_on_exception

WIKISQL_DIR = pathlib.Path("./data/WikiSQL")

all_tables = []
for file_path in WIKISQL_DIR.iterdir():
    all_tables += file_path.read_text().splitlines()

all_tables = list(map(lambda tb: json.loads(tb), all_tables))
for tb in all_tables:
    tb["name"] = f'table_{tb["id"].replace("-", "_")}'

print(f'Read {len(all_tables)} tables (WikiSQL)')

with launch_ipdb_on_exception():
    table_columns_map = {
        tb["name"]: tb["header"]
        for tb in all_tables
    }

with open("tables_columns.json", "w+") as out:
    json.dump(obj=table_columns_map, fp=out)
