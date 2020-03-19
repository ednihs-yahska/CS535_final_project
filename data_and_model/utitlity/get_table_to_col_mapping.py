from lib.query import Query
import json
import re
import sys
import sqlparse
from collections import defaultdict

def get_mapping(header_mapping_dictionary, idx, table_id = None, col = None):
    print(table_id, col)
    if table_id and col:
        print(header_mapping_dictionary[table_id][col])
    
    elif table_id:
        print(header_mapping_dictionary[table_id])



if __name__ == "__main__":
    file_type = sys.argv[1]
    header_mapping_dictionary = defaultdict(lambda : [])
    for idx, line in enumerate(open(f"{file_type}.tables.jsonl")):
        tableJson = json.loads(line)
        if len(header_mapping_dictionary[tableJson["id"]]) == 0:
            header_mapping_dictionary[tableJson["id"]] = tableJson["header"]

    if len(sys.argv) == 2:
        for mapping in list(header_mapping_dictionary.keys()):
            print(mapping)
            for idx, column in enumerate(header_mapping_dictionary[mapping]):
                print("\t", column, f"col{idx}")

    elif len(sys.argv) == 4:
        index = int(sys.argv[3][-1])
        print(header_mapping_dictionary[sys.argv[2]][index])
    
    elif len(sys.argv) == 3:
        print(header_mapping_dictionary[sys.argv[2]])

        
    