# How to do inference with extra test sets

We assume that you have previously trained an X-Transformers model.

## Clone repository

```bash
git clone https://github.com/mvsjober/X-Transformer
cd X-Transformers
```

## Generate test set input files

We need to generate the test set [input files for X-Transformer in the correct format](https://github.com/mvsjober/X-Transformer#running-x-transformer-on-customized-datasets).

First we need the full original training set in order to recreate the dictionary, it should be copied into `datasets/DATABASE` where `DATABASE` is the name for the database. For example:

```bash
mkdir datasets/yso-en
cp /path/to/original/datasets/yso-en/train_raw_texts.txt datasets/yso-en/
```

Generate `X.tsx.npz`, `Y.tsx.npz`, and `testx_raw_texts.txt` from a given [Annif-style full-text document corpus directory](https://github.com/NatLibFi/Annif/wiki/Document-corpus-formats#full-text-document-corpus-directory).

```bash
./xbert_generate_test.py /path/to/annif-data/test/ datasets/yso-en/ eng
```

## Copy X-Transformer model files

The following (or similar) files/directories need to be copied in place from the `save_models/MODEL_NAME` directory where the model was originally trained:

```
pifa-tfidf-s0/indexer/code.npz
proc_data/C.trn.pifa-tfidf-s0.npz
pifa-tfidf-s0/matcher/bert-large-cased-whole-word-masking/{config.json,pytorch_model.bin}
pifa-tfidf-s0/ranker/bert-large-cased-whole-word-masking/param.json 
pifa-tfidf-s0/ranker/bert-large-cased-whole-word-masking/0.model 
```

You can try the helper script, to copy from a local directory:


```bash
./xbert_copy_model.sh ../X-Transformer/ yso-en pifa-tfidf bert -30000
```

or a remote directory:

```bash
./xbert_copy_model.sh puhti:projappl/hpd/X-Transformer/ yso-en pifa-tfidf bert -30000 yso-kirjaesittely
```

## Run evaluation script

```bash
./run_transformer_extra_test.sh yso-en pifa-tfidf bert x 128 0 -30000
```
