import torch
import pathlib
import random
import pandas as pd
import re
import json
from tokenizers import BertWordPieceTokenizer


class WikiSQL_S2S(torch.utils.data.Dataset):
    '''
    Takes natural language queries, table names, column names sequence and
    their respective Cypher queries to load, transform and return input-output
    pairs
    '''

    def __init__(self, data_dir):
        self.DATA_DIR = pathlib.Path(data_dir)
        self.tables_schema = self.DATA_DIR / "tables_columns.json"
        self.input_corpus_dir = self.DATA_DIR / "input_corpus"
        self.output_corpus_dir = self.DATA_DIR / "output_corpus"

        self.special_tokens = [
            "[SOS]",
            "[EOS]",
            "[SEP]",
            "[CLS]"
        ]
        self.build_io_tokenizers()
        self.load_data()
        self.form_X()
        self.form_Y()
        self.compute_max_input_sequence_length()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (
            torch.Tensor(self.X[idx].ids).long(),
            torch.Tensor(self.Y[idx].ids).long()
        )

        return sample

    def compute_max_input_sequence_length(self):
        self.MAX_SEQ_LEN = max(map(lambda encoded_str: len(encoded_str), self.X))

    def form_X(self):

        # TODO: Extract info related to table schema to replace dummies
        self.X = []
        for nlq in self.natural_lang_queries:
            x_str = self.build_x_str(
                nl_query=nlq,
                table_name="table",
                headers=["col0", "col1"]
            )
            self.X.append(self.in_tokenizer.encode(x_str))

    def form_Y(self):

        self.Y = []
        for cq in self.cypher_queries:
            y_str = self.build_y_str(c_query=cq)
            self.Y.append(self.out_tokenizer.encode(y_str))

    def build_x_str(self, nl_query, table_name, headers):
        '''
        [SOS][..NLQ..][SEP][...table_name...][SEP][...headers...][EOS]
        '''
        x_str = f'[SOS]{nl_query}[SEP]{table_name}'
        for header in headers:
            x_str += f'[SEP]{header}'
        x_str += "[EOS]"

        return x_str

    def build_y_str(self, c_query):
        '''
        [SOS][...cypher_query...][EOS]
        '''
        y_str = f'[SOS]{c_query}[EOS]'

        return y_str

    def build_io_tokenizers(self):
        # data sources
        train_src = [
            self.DATA_DIR / "train_tok_cypher.txt",
            self.DATA_DIR / "dev_tok_cypher.txt"
        ]
        test_src = [
            self.DATA_DIR / "test_tok_cypher.txt"
        ]

        # building corpora
        input_corpus_files = self.generate_input_corpus(data_files=train_src)
        output_corpus_files = self.generate_output_corpus(data_files=test_src)

        # training WordPiece tokenizers
        self.in_tokenizer = self.train_tokenizer_from_corpus(
            corpus_files=input_corpus_files
        )
        self.in_tokenizer.add_special_tokens(self.special_tokens)

        self.out_tokenizer = self.train_tokenizer_from_corpus(
            corpus_files=output_corpus_files
        )
        self.out_tokenizer.add_special_tokens(self.special_tokens)

    def load_data(self):

        nlq_path = self.input_corpus_dir / "natural_lang_queries.txt"
        self.natural_lang_queries = nlq_path.read_text().splitlines()

        cq_path = self.output_corpus_dir / "cypher_queries.txt"
        self.cypher_queries = cq_path.read_text().splitlines()

        with open(self.tables_schema) as in_:
            self.tables_columns = json.load(fp=in_)

    @staticmethod
    def _custom_parser(lines, version):
        '''
        Tags each line in lines as natural_lang_query/ sql_query/ cypher_query
        '''
        for count, line in enumerate(lines, 1):

            if count % version == 2:
                nl_query = line.strip()

            elif count % version == 3:
                cypher_query = line.strip()

            elif count % version == 1:
                sql_query = line.strip()

            elif count % version == 0:
                yield (nl_query, sql_query, cypher_query)

    def _proc_file(self, file_path, verbose=True, version=5):
        '''
        Parses file_path and picks natural language query & cypher query
        Returns nlq and cq lists of equal lengths
        '''
        lines = file_path.read_text().splitlines()

        natural_lang_queries = []
        cypher_queries = []
        sql_queries = []

        stream = self._custom_parser(lines=lines, version=version)
        table_id_extractor = "match\(alias:([0-9\-]*)\)"

        for nlq, sq, cq in stream:

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

    def generate_input_corpus(self, data_files):
        '''
        Includes all content that would form a meaningful input for this dataset
        without data leakage
        * natural_lang_queries
        * table_names
        * each table's headers (column names)
        '''

        out_dir = self.input_corpus_dir

        # Get all natural lang queries
        natural_lang_queries = []
        for file in data_files:
            nq, _, _ = self._proc_file(file, verbose=False)
            natural_lang_queries += nq

        natural_lang_queries_file_path = out_dir / "natural_lang_queries.txt"
        with open(natural_lang_queries_file_path, "w+") as out:
            out.write('\n'.join(natural_lang_queries))

        input_corpus = [
            str(natural_lang_queries_file_path),
            str(self.tables_schema)
        ]

        return input_corpus

    def generate_output_corpus(self, data_files):
        '''
        Includes all contents that could be in generated sequences
        * cypher_queries
        * table_names (expecting one-to-one map from input)
        Column names are encoded as "col{n}", so just adding a
        positional embedding to the input columns should do it
        '''
        out_dir = self.output_corpus_dir
        # Get all natural lang queries
        cypher_queries = []
        for file in data_files:
            _, cq, _ = self._proc_file(file, verbose=False)
            cypher_queries += cq

        cypher_queries_file_path = out_dir / "cypher_queries.txt"
        with open(cypher_queries_file_path, "w+") as out:
            out.write('\n'.join(cypher_queries))

        output_corpus = [
            str(cypher_queries_file_path),
            str(self.tables_schema)
        ]

        return output_corpus

    @staticmethod
    def train_tokenizer_from_corpus(corpus_files):
        '''
        Returns a custom word-piece tokenizer tranined on given corpus
        '''
        tokenizer = BertWordPieceTokenizer()
        # tokenizer.enable_padding()
        tokenizer.train(corpus_files)

        return tokenizer


if __name__ == "__main__":
    dataset = WikiSQL_S2S(data_dir="./data")
    dataset_iterator = torch.utils.data.DataLoader(dataset, batch_size=1)

    import ipdb
    ipdb.set_trace()
