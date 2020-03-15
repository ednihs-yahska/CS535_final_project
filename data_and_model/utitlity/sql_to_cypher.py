from lib.query import Query
import json
import re
import sys
from collections import defaultdict

def map_col_to_name(header_mapping_dictionary, select_col, table_name):
    select_col_no = select_col[-1]
    col_no = int(select_col_no)
   #print(header_mapping_dictionary[table_name])
    try:
        select_col = json.loads(header_mapping_dictionary[table_name].replace("'", "\""))["header"][col_no]
    except:
        return None
    
    return select_col

def process_where(all_where, header_mapping_dictionary, isSql=False):
    where_pattern = "col\d"
    where_conds = []
    where_matches = re.finditer(where_pattern, all_where, re.IGNORECASE)
    for idx, match in enumerate(where_matches):
        
        col = map_col_to_name(header_mapping_dictionary, match.group(), table_name)
        if not col:
            return None
        
        if not isSql:  
            col = "alias.("+col+")"

        first_all_where_part = all_where[:match.start()]
        
        second_all_where_part = all_where[match.end():]
        
        _temp = first_all_where_part + col + second_all_where_part
        where_conds.append(_temp.split("AND")[idx])
    return " AND ".join(where_conds)

if __name__ == "__main__":
    lines = 0
    file_type = sys.argv[1]
    header_mapping_dictionary = defaultdict(lambda : [])
    for idx, line in enumerate(open(file_type+".tables.jsonl")):
        tableJson = json.loads(line)
        if len(header_mapping_dictionary[tableJson["id"]]) == 0:
            header_mapping_dictionary[tableJson["id"]] = "{\"header\": "+str(tableJson["header"])+"}"


    for line in open(file_type+"_tok.jsonl"):
        linesJson = json.loads(line)
        jsonQuery = linesJson["sql"]
        text = Query(jsonQuery["sel"], jsonQuery["agg"], conditions=jsonQuery["conds"]).__repr__()
        sql_pattern = '''SELECT\s*(?P<agg>^(\s+).*)?\s*(?P<select_col_1>^(\s+).*)|(?P<select_col>^(\s+).*)\s*FROM\s*(?P<table_name>^(\s+).*)\s*WHERE\s*(?P<where_col>[\w]*)\s*(?P<where_op>[\S><&|+\-\%*!=]{1,2})\s*(?P<where_cond>[\w\s]*)'''
        #text = "SELECT col1 FROM tblName WHERE col2 op val"
        sql_pattern = '''SELECT\s+(?P<agg>COUNT|SUM|AVG|MAX|MIN)?\s+(?P<select_col>[^\s]+)\s+FROM\s+(?P<table_name>[^\s]+)\s*WHERE\s+(?P<where_col>[\w]*)\s+(?P<where_op>[\S><&|+\-\%*!=]{1,2})\s+(?P<where_cond>.*)'''
        matches = re.search(sql_pattern, text, re.IGNORECASE)
        
        index = text.find("table")

        first_part = text[0:index]
        second_part = text[index+5:]


        
        if matches:
            agg = matches.group("agg").strip() if matches.group("agg") else None

            table_name = linesJson["table_id"]#matches.group("table_name").strip()

            select_col = matches.group("select_col").strip() if matches.group("select_col") else None
            select_col = map_col_to_name(header_mapping_dictionary, select_col, table_name)

            if not select_col:
                continue
            # select_col_no = select_col[-1]
            # col_no = int(select_col_no)
            # print(header_mapping_dictionary[table_name])
            # json.loads('{"header": ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"]}')
            # select_col = json.loads(header_mapping_dictionary[table_name].replace("'", "\""))["header"][col_no]

            where_col = matches.group("where_col").strip()
            #where_col = map_col_to_name(header_mapping_dictionary, where_col, table_name)
            where_op = matches.group("where_op").strip()
            where_cond = matches.group("where_cond").strip()

            all_where = where_col + where_op + where_cond

            all_where_cypher = process_where(all_where, header_mapping_dictionary)
            if not all_where_cypher:
                continue
            all_where_sql = process_where(all_where, header_mapping_dictionary, isSql=True)
            
            if agg:
                agg = agg.lower()

            if agg:
                output = f'match(alias:{table_name}) where {all_where_cypher} return {agg}(alias.{select_col})'
            else:
                output = f'match(alias:{table_name}) where {all_where_cypher} return alias.({select_col})'

            first_part = process_where(first_part, header_mapping_dictionary, isSql=True)

            if not first_part:
                continue

            print(first_part+" <table id="+table_name+"/> WHERE "+all_where_sql)
            print(linesJson["question"])
            print(output)
            print("\n")


            # lines += 1
            # if lines >= 5:
            #     exit()
