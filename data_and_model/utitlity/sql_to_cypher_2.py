from lib.query import Query
import json
import re
import sys
import sqlparse
from collections import defaultdict

def process_where_for_sql(all_where):
    result = []
    for idx, conditions in enumerate(re.split("(AND|OR)", all_where)):
        if idx%2==0:
            ops = re.split("(=|<|>|<=|>=)", conditions)
            ops[0] = "alias.({})".format(ops[0].strip())
            ops = " ".join(ops)
            result.append(ops)
        else:
            result.append(conditions)
        result = [r.strip() for r in result]
    return " ".join(result)

def convert_file_from_sql(filename):
    sql_pattern = '''SELECT\s+(?P<agg>(count|sum|avg|max|min)?)(?P<select_col>\(.*\)) FROM \d(?P<table_name>[-]\d+[-]\d).*'''
    for line in open(filename):
        linesJson = json.loads(line)
        sqlQuery = linesJson["sql"]
        text = sqlQuery 
        where_index = text.find("WHERE")
        where_clauses = process_where_for_sql(text[where_index+5:])
        print(text)
        matches = re.search(sql_pattern, text, re.IGNORECASE)
        if matches:
            select_col = matches.group("select_col").strip()
            agg = matches.group("agg").strip() if matches.group("agg") else None
            table_name = matches.group("table_name").strip().replace('-', '_')
            if agg:
                output = f'match(alias:table{table_name}) where {where_clauses} return {agg}(alias.{select_col})'
            else:
                output = f'match(alias:table{table_name}) where {where_clauses} return alias.{select_col}'
        print(output)
            
        
def convert_from_sql(line):
    sql_pattern = '''SELECT\s+(?P<agg>(count|sum|avg|max|min)?)(?P<select_col>\(.*\)) FROM \d(?P<table_name>[-]\d+[-]\d).*'''
    text = line 
    where_index = text.find("WHERE")
    where_clauses = process_where_for_sql(text[where_index+5:])
    print(text)
    matches = re.search(sql_pattern, text, re.IGNORECASE)
    if matches:
        select_col = matches.group("select_col").strip()
        agg = matches.group("agg").strip() if matches.group("agg") else None
        table_name = matches.group("table_name").strip().replace('-', '_')
        print(matches.group("agg"))
        if agg:
            output = f'match(alias:table{table_name}) where {where_clauses} return {agg}(alias.{select_col})'
        else:
            output = f'match(alias:table{table_name}) where {where_clauses} return alias.{select_col}'
    return output

if __name__ == "__main__":
    #line = "SELECT avg(Ranking) FROM 2-1232836-4 WHERE Nationality = saudi arabia AND Years = 2000"
    #print(convert_from_sql(line))
    convert_file_from_sql("results_test.jsonl")
