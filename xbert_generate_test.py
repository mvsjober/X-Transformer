#!/usr/bin/env python3

# HOWTO INSTALL
# pip install stop-words gensim

import argparse
import gensim
import numpy as np
import os
from gensim.matutils import corpus2csc
from scipy.sparse import csr_matrix
from stop_words import get_stop_words

default_vocab_file = "/media/data/data/hpd/Annif-corpora/vocab/yso-ysoplaces-cicero-fi.tsv"

train_txt_fname = 'train_raw_texts.txt'
out_test_txt = 'testx_raw_texts.txt'
out_x_npz = 'X.tsx.npz'
out_y_npz = 'Y.tsx.npz'

label_prefix = '<http://www.yso.fi/onto/yso/'


def filter_words(texts, language):
    stop_words = get_stop_words(language)
    res = []
    for text in texts:
        if isinstance(text, list):
            text = " ".join(text)
        tokens = gensim.utils.simple_preprocess(text)
        res.append([token for token in tokens if token not in stop_words])
    return res


def read_text_datadir(data_path):
    texts = []
    labels = []
    files = sorted(os.listdir(data_path))
    for fn in files:
        if ".tnpp" in fn:
            continue
        if fn.endswith(".txt"):
            with open(os.path.join(data_path, fn), encoding='utf-8') as fp:
                doc = fp.read().replace("\n", " ")
                texts.append(doc)
            tsvfn = fn.replace(".txt", ".tsv")
            with open(os.path.join(data_path, tsvfn)) as fp:
                labs = []
                for line in fp:
                    parts = line.split()
                    if parts[0].startswith(label_prefix):
                        labs.append(parts[0].replace("<", "").replace(">", ""))
                labels.append(labs)
    print('Read {} documents from [{}]'.format(len(texts), data_path))
    return texts, labels


def read_text_data(in_fname):
    texts = []
    with open(in_fname) as fp:
        for line in fp:
            texts.append(line.rstrip().split())
    print('Read {} documents from [{}]'.format(len(texts), in_fname))
    return texts


def write_text_data(texts, out_fname):
    with open(out_fname, 'w') as fp:
        for t in texts:
            fp.write(" ".join(t)+"\n")
        print('Wrote', out_fname)


def write_csr(csr, out_fname):
    np.savez(out_fname, indices=csr.indices, indptr=csr.indptr,
             format=csr.format, shape=csr.shape, data=csr.data)
    print('Wrote', out_fname)


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


def main(args):
    ds_path = args.dataset
    train_texts = read_text_data(os.path.join(ds_path, train_txt_fname))

    print('Generating dictionary...')
    dictionary = gensim.corpora.Dictionary(train_texts)
    n_features = len(dictionary.token2id)

    print('Converting train data to bag-of-words format...')
    train_corpus = [dictionary.doc2bow(t) for t in train_texts]

    print('Creating TF-IDX matrix...')
    tfidf = gensim.models.TfidfModel(train_corpus)

    test_texts, test_labels = read_text_datadir(args.test_data)
    test_texts = filter_words(test_texts, args.language)
    write_text_data(test_texts, os.path.join(ds_path, out_test_txt))

    print('Converting test data to bag-of-words format...')
    test_corpus = [dictionary.doc2bow(t) for t in test_texts]
    test_tfidf = tfidf[test_corpus]

    x_csr = corpus2csc(test_tfidf, num_terms=n_features).transpose().tocsr()
    write_csr(x_csr, os.path.join(ds_path, out_x_npz))

    subj_map = read_subjects_map(args.vocab)

    y_csr = y_csr_matrix(test_labels, subj_map)
    write_csr(y_csr, os.path.join(ds_path, out_y_npz))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('test_data', help='path to test data directory, '
                        'e.g., ~/data/hpd/test/kirjaesittelyt/yso/eng/all')
    parser.add_argument('dataset', help='path to X-BERT dataset directory, '
                        'e.g., datasets/yso-en')
    parser.add_argument('language', help='fin|swe|eng')
    parser.add_argument('--vocab', default=default_vocab_file,
                        help='vocabulary file')
    args = parser.parse_args()

    la = args.language[:2]
    if la == 'fi':
        args.language = 'finnish'
    elif la == 'sw' or la == 'sv':
        args.language = 'swedish'
    elif la == 'en':
        args.language = 'english'

    main(args)
