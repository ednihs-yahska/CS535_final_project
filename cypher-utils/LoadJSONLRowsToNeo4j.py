import sys
import json
import jsonlines
from py2neo import Graph

if __name__ == '__main__':
    graph = Graph(host='localhost', auth=('neo4j', 'password'))

    q1, q3 = '', ''
    split = 'dev'
    l = 0

    sys.stdout = open('eval.'+split+'.log', 'w', encoding="utf-8")
    
    for table in jsonlines.open(split+'.tables.jsonl'):
        try:
            l += 1
            q1 = "call apoc.create.nodes(['" + table['name'] + "'], "
            q3 = ''
            for i, row in enumerate(table['rows']):
                # print(i, row)
                if i == 0:
                    q3 += '[{'
                if i != 0:
                    q3 += '{'
                for j in range(len(row)):
                    if j != len(row) - 1:
                        if str(row[j]).startswith('"') and str(row[j]).startswith('"'):
                            temp = str(row[j])
                            temp = temp[:-1] + '\\"'
                            q3 += f'col{j}: "\\'+temp+' \",'
                        else:
                            q3 += f'col{j}: "' + str(row[j]) + '",'
                    else:
                        if str(row[j]).startswith('"') and str(row[j]).startswith('"'):
                            temp = str(row[j])
                            temp = temp[:-1] + '\\"'
                            q3 += f'col{j}: "\\'+temp+' \"'
                        else:
                            q3 += f'col{j}: "' + str(row[j]) + '"'
                    # print(row[j])
                if i < len(table['rows']) - 1:
                    q3 += ', split: "'+split+'"},'
                if i==len(table['rows'])-1:
                    q3 += ', split: "' + split + '"}]) '
            q1 += q3 + 'yield node return node'
            print(f'{table["id"]} - {graph.run(q1)}')
        except:
            print(f'{table["id"]} - {table["rows"]}')
            continue

    