from argparse import ArgumentParser

def constructHyperParameters(parser):
    parser.add_argument('--data', default = './data', type=str, help='Please provide location of data directory')
    args = parser.parse_args()
    return args

def readDataFile():
    trainLines, devLines, testLines = [], [], []
    dataPath = args.data

    trainLines = [line.strip('\n') for (i, line) in enumerate(open(dataPath + '/train_tok_cypher.txt').readlines()) if line.strip('\n') != '']
    devLines = [line.strip('\n') for (i, line) in enumerate(open(dataPath + '/dev_tok_cypher.txt').readlines()) if line.strip('\n') != '']
    testLines = [line.strip('\n') for (i, line) in enumerate(open(dataPath + '/test_tok_cypher.txt').readlines()) if line.strip('\n') != '']

    return trainLines, devLines, testLines

def getSQLCypherQueryLines(lines):
    sql, cypher = [], []
    
    for i,line in enumerate(lines):
        if i % 3 == 0:
            sql.append(line)
        elif i % 3 == 2:
            cypher.append(line)
        else:
            continue
            # Do nothing for other cases
    
    return sql, cypher
    

if __name__ == '__main__':
    parser = ArgumentParser()
    args = constructHyperParameters(parser)

    trainLines, devLines, testLines = readDataFile()

    trainSQL, trainCypher = getSQLCypherQueryLines(trainLines)
    devSQL, devCypher = getSQLCypherQueryLines(devLines)
    testSQL, testCypher = getSQLCypherQueryLines(testLines)

    trainSQLString, trainCypherString = '\n'.join(trainSQL), '\n'.join(trainCypher)
    devSQLString, devCypherString = '\n'.join(devSQL), '\n'.join(devCypher)
    testSQLString, testCypherString = '\n'.join(testSQL), '\n'.join(testCypher)

    # with open('./data/trainsql.txt', 'w') as writeFile:
    #     writeFile.write(trainSQLString)

    # with open('./data/devsql.txt', 'w') as writeFile:
    #     writeFile.write(devSQLString)

    # with open('./data/testsql.txt', 'w') as writeFile:
    #     writeFile.write(testSQLString)

    # with open('./data/traincypher.txt', 'w') as writeFile:
    #     writeFile.write(trainCypherString)

    # with open('./data/devcypher.txt', 'w') as writeFile:
    #     writeFile.write(devCypherString)

    # with open('./data/testcypher.txt', 'w') as writeFile:
    #     writeFile.write(testCypherString)

