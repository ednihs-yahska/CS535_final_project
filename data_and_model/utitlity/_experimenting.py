import re

unnamed_pattern = "SELECT(.*)FROM(.*)WHERE\s*([\w]*)\s*([\S><&|+\-\%*!=]{1,2})\s*([\w]*)"
#sql_pattern = '''SELECT (?P<select_col>.*) FROM(?P<table_name>.*)WHERE\s*(?P<where_col>[\w]*)\s*(?P<where_op>[\S><&|+\-\%*!=]{1,2})\s*(?P<where_cond>[\w]*)'''
sql_pattern = '''SELECT ((?P<agg>.*)\((?P<select_col_1>.*)\)|(?P<select_col>.*)) FROM(?P<table_name>.*)WHERE\s*(?P<where_col>[\w]*)\s*(?P<where_op>[\S><&|+\-\%*!=]{1,2})\s*(?P<where_cond>[\w]*)'''
text = "SELECT col1 FROM tblName WHERE col2 op val"
matches = re.search(sql_pattern, text, re.IGNORECASE)

if matches:
    agg = matches.group("agg").strip() if matches.group("agg") else None
    if agg: 
        select_col = matches.group("select_col_1").strip() if matches.group("select_col_1") else None
    else:
        select_col = matches.group("select_col").strip() if matches.group("select_col") else None
    table_name = matches.group("table_name").strip()
    where_col = matches.group("where_col").strip()
    where_op = matches.group("where_op").strip()
    where_cond = matches.group("where_cond").strip()

    if agg:
        output = f'MATCH (alias:{table_name}) WHERE alias.{where_col} {where_op} {where_cond} RETURN alias.{agg}({select_col})'
    else:
        output = f'MATCH (alias:{table_name}) WHERE alias.{where_col} {where_op} {where_cond} RETURN alias.{select_col}'
        
    print(output)