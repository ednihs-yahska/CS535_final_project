# Natural Language to Cypher Query Generation

### Dataset
* The dataset is automatically generated from [WikiSQL](https://github.com/salesforce/WikiSQL)
* Data ETL for training/ evaluating model can be found in `data_prep.py`

### Modeling
* A sequence to sequence model is being used to generate Cypher queries from Natural language text and table metadata. 
* `a.gru_encoder_decoder.py` is GRU based encoder-decoder trainer
* `b.gru_encoder_decoder_attn.py` uses attention on encoder outputs while decoding the Cypher query

### Team
Akshay Shinde, Rahul Damineni, Sheekha Jariwala
