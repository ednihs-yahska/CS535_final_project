def segregateData(split):    
    lines = [line.strip('\n') for line in open('./'+split+'_tok_cypher.txt', 'r').readlines() if line != '\n']
    sql, cypher = [], []

    for i,line in enumerate(lines):
        if i%3 == 0:
            sql.append(line)
        elif i%3 == 2:
            cypher.append(line)
    print(len(sql))
    print(len(cypher))

    with open(f'{split}_sql.txt', 'w') as f:
        f.writelines(f'{line}\n' for line in sql)
    with open(f'{split}_cypher.txt', 'w') as f:
        f.writelines(f'{line}\n' for line in cypher)

if __name__ == '__main__':
    segregateData('test')