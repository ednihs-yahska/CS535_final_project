import pathlib
import random
import pandas as pd


def proc_file(file_path):
    '''
    Parses file_path and picks natural language query & cypher query
    Returns nlq and cq lists of equal lengths
    '''
    lines = file_path.read_text().splitlines()

    natural_lang_queries = []
    cypher_queries = []

    for count, line in enumerate(lines, 1):

        if count % 5 == 2:
            natural_lang_queries.append(line.strip())

        elif count % 5 == 3:
            cypher_queries.append(line.strip())

    assert len(natural_lang_queries) == len(cypher_queries)

    total_count = len(cypher_queries)
    random_idx = random.randint(0, total_count - 1)
    print(
        f'Set: {file_path}\n\
        NLQ: {natural_lang_queries[10]}\n\
        Cypher: {cypher_queries[10]}\n\
        Total count: {total_count}'
    )

    return natural_lang_queries, cypher_queries


if __name__ == "__main__":
    DATA_DIR = pathlib.Path("./data")

    train_path = DATA_DIR / "train_tok_cypher.txt"
    eval_path = DATA_DIR / "dev_tok_cypher.txt"
    test_path = DATA_DIR / "test_tok_cypher.txt"

    proc_file(train_path)
    proc_file(eval_path)
    proc_file(test_path)
