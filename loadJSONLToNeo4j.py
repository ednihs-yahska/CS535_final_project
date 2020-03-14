import jsonlines
import json
from py2neo import Graph

if __name__ == "__main__":
    graph = Graph(host='localhost', auth=('neo4j', 'password'))

    readFile = './data/dev.tables.jsonl'
    writeFile = './data/mdev.tables.jsonl'

    jsonData = []
    with jsonlines.open(readFile) as reader:
    # with jsonlines.open('./data/temp.jsonl') as reader:
        for obj in reader.iter(type=dict):
            jsonData.append(obj)
            # jsonData[obj['id']] = {"id": obj['id'], "headers": obj['header'], "rows": obj['rows']}
        
    # print('jsonData: {}'.format(jsonData))
    jsonData = json.dumps(jsonData)
    # print('jsonData: {}'.format( jsonData))

    with open(writeFile, mode='w') as writer:
        writer.writelines(jsonData)
    
    # query = """
    # with {jsonData} as json
    # call apoc.create.node([json.name], {id: json.name})

    # yield node
    # return node
    # """

    # for i, data in enumerate(jsonData.items()):
    # for key, val in jsonData.items():
    #     print(key, val)
        # print(i,data[0], data[1])
        # print(i,data[0],jsonData[data[0]])
        # print(graph.run(query, {'jsonData':jsonData}))

    # print(graph.run(query, parameters={'jsonData':jsonData}))
