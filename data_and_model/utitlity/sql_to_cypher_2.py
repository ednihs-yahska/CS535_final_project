from lib.query import Query
import json
import re
import sys
import sqlparse
from collections import defaultdict

def map_name_to_col(header_mapping_dictionary, select_col, table_name):
    try:
        headers = json.loads(header_mapping_dictionary[table_name].replace("'", "\""))["header"]
        select_col = select_col[1:-1]
        for idx, header in enumerate(headers):
            if select_col == header:
                return f"col{idx}"
    except:
        return None
    
    return select_col

def process_where_for_sql(all_where, header_mapping_dictionary, table_name):
    result = []
    for idx, conditions in enumerate(re.split("(AND|OR)", all_where)):
        if idx%2==0:
            ops = re.split("(=|<|>|<=|>=)", conditions)
            ops[0] = map_name_to_col(header_mapping_dictionary ,ops[0].strip(r"\s"), table_name)
            ops[0] = "alias.({})".format(ops[0].strip()) if ops[0] else "<NOT PROCESSED>"
            ops = " ".join(ops)
            result.append(ops)
        else:
            result.append(conditions)
        result = [r.strip() for r in result]
    return " ".join(result)

def convert_file_from_sql(filename, header_mapping_dictionary):
    sql_pattern = '''SELECT\s+(?P<agg>(count|sum|avg|max|min)?)(?P<select_col>\(.*\)) FROM (?P<table_name>\d[-]\d+[-]\d+).*'''
    for line in open(filename):
        linesJson = json.loads(line)
        sqlQuery = linesJson["sql"]
        text = sqlQuery 
        where_index = text.find("WHERE")
        matches = re.search(sql_pattern, text, re.IGNORECASE)
        if matches:
            select_col = matches.group("select_col").strip()
            agg = matches.group("agg").strip() if matches.group("agg") else None
            table_name = matches.group("table_name").strip()
            where_clauses = process_where_for_sql(text[where_index+5:], header_mapping_dictionary, table_name)
            select_col = map_name_to_col(header_mapping_dictionary, select_col.strip(), table_name)
            table_name = table_name.replace('-', '_')
            if agg:
                text = f'SELECT {agg}{select_col} FROM {table_name} WHERE {where_clauses}'
                output = f'match(alias.({table_name})) where {where_clauses} return {agg}(alias.({select_col}))'
            else:
                text = f'SELECT {select_col} FROM {table_name} WHERE {where_clauses}'
                output = f'match(alias:({table_name})) where {where_clauses} return alias.({select_col})'
            print(text)
            print(output)
            print()
            
        
# def convert_from_sql(line):
#     sql_pattern = '''SELECT\s+(?P<agg>(count|sum|avg|max|min)?)(?P<select_col>\(.*\)) FROM \d(?P<table_name>[-]\d+[-]\d).*'''
#     text = line 
#     where_index = text.find("WHERE")
#     where_clauses = process_where_for_sql(text[where_index+5:])
#     print(text)
#     matches = re.search(sql_pattern, text, re.IGNORECASE)
#     if matches:
#         select_col = matches.group("select_col").strip()
#         agg = matches.group("agg").strip() if matches.group("agg") else None
#         table_name = matches.group("table_name").strip().replace('-', '_')
#         print(matches.group("agg"))
#         if agg:
#             output = f'match(alias:table{table_name}) where {where_clauses} return {agg}(alias.{select_col})'
#         else:
#             output = f'match(alias:table{table_name}) where {where_clauses} return alias.{select_col}'
#     return output

if __name__ == "__main__":
    header_mapping_dictionary = defaultdict(lambda : [])
    for idx, line in enumerate(open("test.tables.jsonl")):
        tableJson = json.loads(line)
        if len(header_mapping_dictionary[tableJson["id"]]) == 0:
            header_mapping_dictionary[tableJson["id"]] = "{\"header\": "+str(tableJson["header"])+"}"

    #line = "SELECT avg(Ranking) FROM 2-1232836-4 WHERE Nationality = saudi arabia AND Years = 2000"
    #print(convert_from_sql(line))
    convert_file_from_sql("results_test.jsonl", header_mapping_dictionary)


{"query": {"agg": 0, "sel": 0, "conds": [[1, 0, "retief goosen"], [2, 0, "south africa"]]}, "table_id": "2-12626983-5", "nlu": "Name the place for south africa and retief goosen", "sql": "SELECT (Place) FROM 2-12626983-5 WHERE Player = retief goosen AND Country = south africa"}

