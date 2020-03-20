import sys
from neo4j import GraphDatabase

def segregateData(split):    
    lines = [line.strip('\n') for line in open('./'+split+'_tok_cypher.txt', 'r', encoding='utf-8').readlines() if line != '\n']
    sql, cypher = [], []

    for i,line in enumerate(lines):
        if i%3 == 0:
            sql.append(line)
        elif i % 3 == 2:
            cypher.append(line)
    print(len(sql))
    print(len(cypher))

    return sql, cypher


if __name__ == '__main__':
    split = sys.argv[1]

    sys.stdout = open('execute.'+split+'.log', 'w', encoding="utf-8")

    sql, cypher = segregateData(split)

    uri = "bolt://localhost"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))
    results = []

    for i in range(len(cypher)):
        with driver.session() as session:
            try:
                result = session.read_transaction(lambda tx: tx.run(cypher[i]))
                temp = (result.values()[0], cypher[i])
                print(f'{temp} - {cypher[i]}')
                results.append(temp)
            except:
                print(f'<ERROR> - {cypher[i]}')

    with open('./' + split + '_cypher_result.txt', 'w', encoding='utf-8') as resultFile:
        resultFile.writelines('\n'.join(map(str,results)))