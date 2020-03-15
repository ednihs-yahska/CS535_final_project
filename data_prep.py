import pathlib
import random
import pandas as pd
import re
from tokenizers import BertWordPieceTokenizer


def custom_parser(lines):
    for count, line in enumerate(lines, 1):

        if count % 5 == 2:
            nl_query = line.strip()

        elif count % 5 == 3:
            cypher_query = line.strip()

        elif count % 5 == 1:
            sql_query = line.strip()

        elif count % 5 == 0:
            yield (nl_query, sql_query, cypher_query)


def proc_file(file_path, verbose=True):
    '''
    Parses file_path and picks natural language query & cypher query
    Returns nlq and cq lists of equal lengths
    '''
    lines = file_path.read_text().splitlines()

    natural_lang_queries = []
    cypher_queries = []
    sql_queries = []

    stream = custom_parser(lines=lines)
    table_id_extractor = "match\(alias:([0-9\-]*)\)"

    for nlq, sq, cq in stream:

        # Deriving table name from cypher queries to modify SQL
        match = re.finditer(table_id_extractor, cq).__next__()
        table_id = match[1].replace("-", "_")
        table_name = f'table_{table_id}'
        sq = sq.replace(" table ", f' {table_name} ')

        natural_lang_queries.append(nlq)
        cypher_queries.append(cq)
        sql_queries.append(sq)

    assert len(natural_lang_queries) == len(cypher_queries) == len(sql_queries)

    total_count = len(cypher_queries)
    random_idx = random.randint(0, total_count - 1)

    if verbose:
        print(
            f'Set: {file_path}\n\
            NLQ: {natural_lang_queries[10]}\n\
            Cypher: {cypher_queries[10]}\n\
            SQL: {sql_queries[10]}\n\
            Total count: {total_count}'
        )

    return natural_lang_queries, cypher_queries, sql_queries


def generate_input_corpus(data_files):

    out_dir = DATA_DIR / "input_corpus"

    # Get all natural lang queries
    natural_lang_queries = []
    for file in data_files:
        nq, _, _ = proc_file(file, verbose=False)
        natural_lang_queries += nq

    natural_lang_queries_file_path = out_dir / "natural_lang_queries.txt"
    with open(natural_lang_queries_file_path, "w+") as out:
        out.write('\n'.join(natural_lang_queries))

    # All table and their coln names
    #

    input_corpus = [str(natural_lang_queries_file_path)]

    return input_corpus


def generate_output_corpus(data_files):
    out_dir = DATA_DIR / "output_corpus"

    # Get all natural lang queries
    cypher_queries = []
    for file in data_files:
        _, cq, _ = proc_file(file, verbose=False)
        cypher_queries += cq

    cypher_queries_file_path = out_dir / "cypher_queries.txt"
    with open(cypher_queries_file_path, "w+") as out:
        out.write('\n'.join(cypher_queries))

    # All table and their coln names
    #

    output_corpus = [str(cypher_queries_file_path)]

    return output_corpus


def train_tokenizer_from_corpus(corpus_files):
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train([corpus_files])

    return tokenizer


if __name__ == "__main__":
    DATA_DIR = pathlib.Path("./data")

    train_path = DATA_DIR / "train_tok_cypher.txt"
    eval_path = DATA_DIR / "dev_tok_cypher.txt"
    test_path = DATA_DIR / "test_tok_cypher.txt"

    proc_file(train_path)
    proc_file(eval_path)
    proc_file(test_path)
