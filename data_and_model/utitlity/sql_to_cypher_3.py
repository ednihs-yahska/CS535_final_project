import sys
import re

def process_where_for_sql(all_where):
    result = []
    for idx, conditions in enumerate(re.split("(\sAND\s|\sOR\s)", all_where)):
        if idx%2==0:
            ops = re.split("(=|<|>|<=|>=)", conditions)
            ops[2] = "\"{}\"".format(ops[2].strip()) if ops[2] else "<NOT PROCESSED>"
            ops[0] = f"alias.{ops[0].strip()}"
            ops = " ".join(ops)            
            result.append(ops)
        else:
            result.append(conditions)
        result = [r.strip() for r in result]
    return " ".join(result)

if __name__ == "__main__":

    sql_pattern = '''SELECT\s+(?P<agg>COUNT|SUM|AVG|MAX|MIN)?\s+(?P<select_col>[^\s]+)\s+FROM\s+(?P<table_name>[^\s]+)\s*WHERE.*'''
    cypher_pattern = '''match\(alias:(?P<c_table_name>\d[-]\d+[-]\d+)\)(?P<rest_cypher>.*)'''
    lines = []
    file_type = sys.argv[1]
    for line in open(f"{file_type}_tok_cypher.txt"):
        lines.append(line)

    no = 0
    agg = None
    select_col = None
    for line in range(0, len(lines), 5):
        try:
            cypher = lines[line+2]
            nl = lines[line+1]
            sql = lines[line]
            cypher_matches = re.search(cypher_pattern, cypher, re.IGNORECASE)
            matches = re.search(sql_pattern, sql, re.IGNORECASE)
            if matches:
                agg = matches.group("agg") if matches.group("agg") else None
                select_col = matches.group("select_col")
                if agg:
                    parts = sql.split("FROM")
                    sql = f"SELECT {agg.lower()}({select_col}) FROM {parts[1]}"
                    
            if cypher_matches:
                table_name = cypher_matches.group("c_table_name")
                rest = cypher_matches.group("rest_cypher")
                sql = re.sub(r'FROM\s+table', f"FROM table_{table_name.replace('-', '_')}", sql)
                sql_where_parts = sql.split("WHERE")
                
                sql = sql_where_parts[0]+" WHERE "+process_where_for_sql(sql_where_parts[1])
                
                if agg:
                    cypher = f"match(alias:table_{table_name.replace('-', '_')}) where {process_where_for_sql(sql_where_parts[1])} and alias.split='{file_type}' return {agg.lower()}(alias.{select_col})"
                else:
                    cypher = f"match(alias:table_{table_name.replace('-', '_')}) where {process_where_for_sql(sql_where_parts[1])} and alias.split='{file_type}' return alias.{select_col}"

        except Exception as e:
            print(e)
            continue
        print(sql.strip())
        print(nl.strip())
        print(cypher.strip())
        print()

    


