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
./xbert_generate_test.py /path/to/kirjaesittelyt/yso/eng/all/ /path/to/Annif-corpora/vocab/yso-ysoplaces-cicero-fi.tsv datasets/yso-en eng --extra_test kes
```

Another example with splitting of long text to max 128 words:

```bash
./xbert_generate_test.py /path/to/jyu-theses/eng-test /path/to/Annif-corpora/vocab/yso-ysoplaces-cicero-fi.tsv datasets/yso-en en --extra_test jyu --split 128
```

Run inference and evaluation:

```bash
./run_transformer_extra_test.sh yso-en pifa-tfidf bert bert-large-cased-whole-word-masking kes 128 0 -30000
```

For splitted datasets, the final results are obtained by averaging over all the parts. The `-l N` option limits the averaging to the `N` first parts:

```bash
./run_transformer_extra_test.sh yso-en pifa-tfidf bert bert-large-cased-whole-word-masking jyu 128 0 -30000 -l 3
```

The last summing step can also be rerun on its own, e.g., if you just tried with `-l 3` and want to retry with `-l 10` without having to rerun the whole inference pipeline:

```bash 
./xbert_evaluator.py -y datasets/yso-en/Y.tsjyu.npz -p save_models/yso-en/pifa-tfidf-s0/ranker-30000/bert-large-cased-whole-word-masking/tsjyu.pred.npz -i datasets/yso-en/testjyu_indices.txt -l 10
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
./xbert_generate_test.py /path/to/Annif-corpora/fulltext/kirjastonhoitaja/test/ /path/to/Annif-corpora/vocab/yso-ysoplaces-cicero-fi.tsv datasets/yso-fi fi --extra_test kh --ext tnpp
```

Note that this requires that we have generated `.tnpp.txt` versions of all the test set text files. First, the [Turku neural parser pipeline](http://turkunlp.org/Turku-neural-parser-pipeline/) tools are used to generate `.txt.conllu` files. Next:

```bash
for i in *.txt.conllu ; do grep -v "^#" $i | cut -f3 | tr "#" " " | tr "\n" " " | tr [:upper:] [:lower:] > ${i/.txt.conllu/.tnpp.txt} ; done
```

Another example with splitting of long text to max 128 words (note that we can specify separately to use tnpp or raw input for the TFIDF (`--ext`) and the input text for the network (`--raw_ext`)):

```bash
./xbert_generate_test.py /path/to/jyu-theses/fin-test /path/to/Annif-corpora/vocab/yso-ysoplaces-cicero-fi.tsv datasets/yso-fi fi --extra_test jyu --split 128 --ext tnpp --raw_ext tnpp
```

Run inference and evaluation:

```bash
./run_transformer_extra_test.sh yso-fi pifa-tfidf bert-base-finnish-uncased TurkuNLP/bert-base-finnish-uncased-v1 kh 128 0 -50000 -l 3
```

### Swedish

Very similar to English. Getting the dictionary, TF-IDF and model:

```bash
wget https://a3s.fi/hpd-data/yso-sv-dict-tfidf.zip
wget https://a3s.fi/hpd-models/yso-sv-pifa-tfidf-bert-30000.zip
unzip yso-sv-dict-tfidf.zip
unzip yso-sv-pifa-tfidf-bert-30000.zip
```

Inference and evaluation examples (uses splitting):

```bash
./xbert_generate_test.py /path/to/jyu-theses/swe/ /path/to/Annif-corpora/vocab/yso-ysoplaces-cicero-fi.tsv datasets/yso-sv sv --extra_test jyu --split 128

./run_transformer_extra_test.sh yso-sv pifa-tfidf bert-large-swedish-uncased af-ai-center/bert-large-swedish-uncased jyu 128 0 -30000 -l 10
```

### Tri-lang

Language model trained on multi-language (Finnish, Swedish, English) data.

```bash
wget https://a3s.fi/hpd-data/yso-trilang-dict-tfidf.zip
wget https://a3s.fi/hpd-models/yso-trilang-pifa-tfidf-bert-50000.zip
unzip yso-trilang-dict-tfidf.zip
unzip yso-trilang-pifa-tfidf-bert-50000.zip
```

Example run:

```bash
./xbert_generate_test.py ~/data/hpd/test/jyu-theses/swe/ ~/data/hpd/Annif-corpora/vocab/yso-ysoplaces-cicero-fi.tsv datasets/yso-trilang sv --extra_test jyu_swe --split 128
./run_transformer_extra_test.sh yso-trilang pifa-tfidf bert-multilingual bert-base-multilingual-uncased jyu_swe 128 0 -50000
```

## Generate training set input files

If you want to train your own models, you first need to generate the training set [input files for X-Transformer in the correct format](https://github.com/OctoberChang/X-Transformer#running-x-transformer-on-customized-datasets). This can be done with the `xbert_generate_train.py` script, for example:

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
./xbert_generate_test.py /path/to/annif-data/test/ /path/to/yso-ysoplaces-cicero-fi.tsv datasets/yso-en eng --extra_test foo
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

or a remote directory (note last argument if the remote folder is named differently):

```bash
./xbert_copy_model.sh puhti:projappl/hpd/X-Transformer/ yso-fi pifa-tfidf bert TurkuNLP/bert-base-finnish-uncased-v1 -50000 yso-fi-new
```

### Run inference script

Example:

```bash
./run_transformer_extra_test.sh yso-en pifa-tfidf bert bert-large-cased-whole-word-masking foo 128 0 -30000
```
