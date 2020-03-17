from argparse import ArgumentParser

def constructHyperParameters(parser):
    parser.add_argument('--data', default = './data', type=str, help='Please provide location of data directory')
    args = parser.parse_args()
    return args

def readFile(fileName):
    return trainLines = [(i,line.strip('\n')) for (i, line) in enumerate(open('./data/train_tok_cypher.txt').readlines()) if line.strip('\n') != '']

if __name__ == '__main__':
    parser = ArgumentParser()
    args = constructHyperParameters(parser)
    print(args)

    trainLines = [(i,line.strip('\n')) for (i, line) in enumerate(open('./data/train_tok_cypher.txt').readlines()) if line.strip('\n') != '']
    devLines = [line.strip('\n') for line in open('./data/dev_tok_cypher.txt') if line.strip('\n') != '']
    testLines = [line.strip('\n') for line in open('./data/test_tok_cypher.txt') if line.strip('\n') != '']

    for line in trainLines[:10]:
        print(line)

    trainSQL, devSQL, testSQL = [], [], []
    trainCypher, devCypher, testCypher = [], [], []

    for i,line in enumerate(trainLines):
        if i % 3 == 0:
            trainSQL.append(line)
        elif i % 2 == 2:
            trainCypher.append(line)
        else:
            continue
            # Do nothing for other cases


    for line in trainSQL[:10]:
        print(line)
    print(trainCypher[:10])