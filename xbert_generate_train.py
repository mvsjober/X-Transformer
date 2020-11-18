#!/usr/bin/env python3

# HOWTO INSTALL
# pip install stop-words gensim

import argparse
import gensim
import numpy as np
import os
import sys
from gensim.matutils import corpus2csc
from scipy.sparse import csr_matrix
from stop_words import get_stop_words


def filter_words(texts, language):
    stop_words = get_stop_words(language)
    res = []
    for text in texts:
        if isinstance(text, list):
            text = " ".join(text)
        tokens = gensim.utils.simple_preprocess(text)
        res.append([token for token in tokens if token not in stop_words])
    return res


def write_text_data(texts, out_fname):
    with open(out_fname, 'w') as fp:
        for t in texts:
            fp.write(" ".join(t)+"\n")
        print('Wrote', out_fname)


def write_csr(csr, out_fname):
    np.savez(out_fname, indices=csr.indices, indptr=csr.indptr,
             format=csr.format, shape=csr.shape, data=csr.data)
    print('Wrote', out_fname)


def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


def read_subjects_map(vocab_file):
    subj_map = {}

    with open(vocab_file) as fp:
        for i, line in enumerate(fp):
            parts = line.split()
            subj = parts[0].replace("<", "").replace(">", "")
            subj_map[subj] = i

    print('Read {} subjects from [{}].'.format(len(subj_map), vocab_file))
    return subj_map


def y_csr_matrix(test_labels, subj_map):
    rows, cols = [], []
    n_total, n_notfound = 0, 0

    for i, labs in enumerate(test_labels):
        for l in list(set(labs)):  # eliminate duplicates
            n_total += 1
            try:
                cols.append(subj_map[l])
                rows.append(i)
            except KeyError:
                n_notfound += 1

    assert len(rows) == len(cols)

    y_csr = csr_matrix(([1]*len(rows), (rows, cols)),
                       shape=(len(test_labels), len(subj_map)))
    print('Generated instance-to-label matrix for test with {} entries, '
          '{}/{} subjects not found'.format(len(rows), n_notfound, n_total))
    return y_csr


def read_tsv(fname):
    texts = []
    labels = []
    with open(fname) as fp:
        for line in fp:
            parts = line.split('\t')
            texts.append(parts[0])
            labs = []
            for p in parts[1:]:
                for pp in p.split():
                    labs.append(pp.replace("<", "").replace(">", ""))
            labels.append(labs)
    print('Read {} documents from [{}].'.format(len(texts), fname))
    return texts, labels


def parse_language(lang):
    la = lang[:2]
    if la == 'fi':
        return 'finnish'
    elif la == 'sw' or la == 'sv':
        return 'swedish'
    elif la == 'en':
        return 'english'
    return None


def main(args):
    ds_path = args.dataset
    os.makedirs(ds_path, exist_ok=True)

    print('Reading training data ...')
    train_texts, train_labels = read_tsv(args.train_data)

    print('Filtering words ...')
    train_texts = filter_words(train_texts, args.language)

    write_text_data(train_texts, os.path.join(ds_path, 'train_raw_texts.txt'))

    print('Generating dictionary ...')
    dictionary = gensim.corpora.Dictionary(train_texts)
    # n_features = len(dictionary.token2id)
    dict_fname = os.path.join(ds_path, 'train.dict')
    dictionary.save(dict_fname)
    print('Wrote dictionary to [{}].'.format(dict_fname))

    print('Converting train data to bag-of-words format ...')
    train_corpus = [dictionary.doc2bow(t) for t in train_texts]

    print('Creating TF-IDF matrix ...')
    tfidf = gensim.models.TfidfModel(train_corpus)
    train_tfidf = tfidf[train_corpus]
    tfidf_fname = os.path.join(ds_path, 'train.tfidf.mm')
    gensim.corpora.MmCorpus.serialize(tfidf_fname, train_tfidf)
    print('Wrote TF-IDF to [{}].'.format(tfidf_fname))

    x_csr = corpus2csc(train_tfidf).transpose().tocsr()

    empty_text_ind = []
    for i, t in enumerate(train_texts):
        if len(t) == 0:
            empty_text_ind.append(i)
    print('Removing {} empty train texts.'.format(len(empty_text_ind)))

    x_csr = delete_rows_csr(x_csr, empty_text_ind)
    write_csr(x_csr, os.path.join(ds_path, 'X.trn.npz'))

    subj_map = read_subjects_map(args.vocab)

    y_csr = y_csr_matrix(train_labels, subj_map)
    y_csr = delete_rows_csr(y_csr, empty_text_ind)
    write_csr(y_csr, os.path.join(ds_path, 'Y.trn.npz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate train input files for X-Transformer in the '
        'correct format. For format specification, see:\n'
        'https://github.com/mvsjober/X-Transformer#running-x-transformer-on-customized-datasets')
    parser.add_argument('train_data',
                        help='train data set in Annif TSV format')
    parser.add_argument('vocab', help='vocabulary file in TSV format')
    parser.add_argument('dataset', help='path to X-BERT dataset directory, '
                        'e.g., datasets/yso-en')
    parser.add_argument('language', help='fin|swe|eng')
    args = parser.parse_args()
    args.language = parse_language(args.language)
    if args.language is None:
        print('ERROR: language [{}] not supported'.format(args.language))
        sys.exit(1)

    main(args)
