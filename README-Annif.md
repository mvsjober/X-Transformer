# How to use X-Transformer with Annif datasets

First, we need to use this customized fork, which includes:

- helper scripts to generate X-Transformer input files from Annif-formatted datasets
- minimal changes to X-Transformer to allow for additional test sets (original code assumes there is exactly one test set)

Clone the repository:

```bash
git clone https://github.com/mvsjober/X-Transformer
cd X-Transformer
```

[Install dependencies according to the instructions in the main repository](https://github.com/OctoberChang/X-Transformer#depedencies-via-conda-environment). In addition, in the same conda environment, run:

```bash
pip install stop-words gensim pandas
```

For Swedish (at least), you also need to upgrade the version of Transformers:

```bash
pip install transformers==2.5.1 -U
```

## Reproduce some HPD project results

### English

Download and extract original dictionary and TF-IDF model:

```bash
wget https://a3s.fi/hpd-data/yso-en-dict-tfidf.zip
unzip yso-en-dict-tfidf.zip
```

Unzipping will create some files under `datasets/yso-en`. 

Download and extract pre-trained model:

```bash
wget https://a3s.fi/hpd-models/yso-en-pifa-tfidf-bert-30000.zip
unzip yso-en-pifa-tfidf-bert-30000.zip
```

Unzipping will create several files under `save_models/yso-en`.

Generate test set input files:

```bash
./xbert_generate_test.py ~/data/hpd/test/kirjaesittelyt/yso/eng/all/ ~/data/hpd/Annif-corpora/vocab/yso-ysoplaces-cicero-fi.tsv datasets/yso-en eng --extra_test kes
```

Run inference and evaluation:

```bash
./run_transformer_extra_test.sh yso-en pifa-tfidf bert bert-large-cased-whole-word-masking kes 128 0 -30000
```

### Finnish

Download and extract dictionary, TF-IDF and model:

```bash
wget https://a3s.fi/hpd-data/yso-fi-dict-tfidf.zip
wget https://a3s.fi/hpd-models/yso-fi-pifa-tfidf-bert-50000.zip
unzip yso-fi-dict-tfidf.zip
unzip yso-fi-pifa-tfidf-bert-50000.zip
```

Generate test set input files:

```bash
./xbert_generate_test.py ~/data/hpd/test/kirjaesittelyt/yso/fin/test/ ~/data/hpd/Annif-corpora/vocab/yso-ysoplaces-cicero-fi.tsv datasets/yso-fi fi --extra_test kes --ext tnpp
```

Note that this requires that we have generated `.tnpp.txt` versions of all the test set text files. First, the [Turku neural parser pipeline](http://turkunlp.org/Turku-neural-parser-pipeline/) tools are used to generate `.txt.conllu` files. Next:

```bash
for i in *.txt.conllu ; do grep -v "^#" $i | cut -f3 | tr "#" " " | tr "\n" " " | tr [:upper:] [:lower:] > ${i/.txt.conllu/.tnpp.txt} ; done
```

Run inference and evaluation:

```bash
./run_transformer_extra_test.sh yso-fi pifa-tfidf bert-base-finnish-uncased TurkuNLP/bert-base-finnish-uncased-v1 kes 128 0 -50000
```

### Swedish

Very similar to English:

```bash
wget https://a3s.fi/hpd-data/yso-sv-dict-tfidf.zip
wget https://a3s.fi/hpd-models/yso-sv-pifa-tfidf-bert-30000.zip
unzip yso-sv-dict-tfidf.zip
unzip yso-sv-pifa-tfidf-bert-30000.zip

./xbert_generate_test.py ~/data/hpd/test/kirjaesittelyt/yso/swe/all/ ~/data/hpd/Annif-corpora/vocab/yso-ysoplaces-cicero-fi.tsv datasets/yso-sv sv --extra_test kes

./run_transformer_extra_test.sh yso-sv pifa-tfidf bert-large-swedish-uncased af-ai-center/bert-large-swedish-uncased kes 128 0 -30000
```

## Generate training set input files

First, we need to generate the training set [input files for X-Transformer in the correct format](https://github.com/OctoberChang/X-Transformer#running-x-transformer-on-customized-datasets). This can be done with the `xbert_generate_train.py` script, for example:

```bash
./xbert_generate_train.py /path/to/yso-cicero-finna-eng.tsv /path/to/yso-ysoplaces-cicero-fi.tsv datasets/yso-en eng
```

## Generate test set input files

Next, we generate the test set input files. In order to create files compatible with the training set, we need two files created when generating the training set: `train.dict` and `train.tfidf.mm`, which can both be found in the corresponding datasets directory.

Generate `X.tst.npz`, `Y.tst.npz`, and `test_raw_texts.txt` from a given [Annif-style full-text document corpus directory](https://github.com/NatLibFi/Annif/wiki/Document-corpus-formats#full-text-document-corpus-directory).

```bash
./xbert_generate_test.py /path/to/annif-data/test/ /path/to/yso-ysoplaces-cicero-fi.tsv datasets/yso-en eng
```

We can also create additional test sets by specifying an extra suffix to add to the created files. For example:

```bash
./xbert_generate_test.py /path/to/annif-data/test/ /path/to/yso-ysoplaces-cicero-fi.tsv datasets/yso-en eng --extra_test x
```

would generate the files `X.tsx.npz`, `Y.tsx.npz`, and `testx_raw_texts.txt`.

## Run training

[See instructions in main repository](https://github.com/OctoberChang/X-Transformer#indexer).

## Run inference

### Copy X-Transformer model files

If a model has been trained elsewhere (such as an HPC cluster), we need to copy all the files related to the model to where we are performing the inference.

The following (or similar) files/directories need to be copied in place from the `save_models/MODEL_NAME` directory where the model was originally trained:

```
pifa-tfidf-s0/indexer/
proc_data/C.trn.pifa-tfidf-s0.npz
pifa-tfidf-s0/matcher/bert-large-cased-whole-word-masking/{config.json,pytorch_model.bin}
pifa-tfidf-s0/ranker/bert-large-cased-whole-word-masking/param.json 
pifa-tfidf-s0/ranker/bert-large-cased-whole-word-masking/0.model/
```

You can try the helper script, to copy from a local directory:


```bash
./xbert_copy_model.sh ../X-Transformer/ yso-en pifa-tfidf bert bert-large-cased-whole-word-masking -30000
```

or a remote directory:

```bash
./xbert_copy_model.sh puhti:projappl/hpd/X-Transformer/ yso-fi pifa-tfidf bert  TurkuNLP/bert-base-finnish-uncased-v1 -50000 yso-fi-new
```

### Run inference script

Example:

```bash
./run_transformer_extra_test.sh yso-en pifa-tfidf bert bert-large-cased-whole-word-masking x 128 0 -30000
```
