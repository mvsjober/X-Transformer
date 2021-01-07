#!/usr/bin/env python3

import argparse
import gensim
import os
import sys
from gensim.matutils import corpus2csc

from xbert_generate_train import (parse_language, write_text_data,
                                  filter_words, write_csr, read_subjects_map,
                                  y_csr_matrix)


label_prefix = '<http://www.yso.fi/onto/yso/'


def read_text_datadir(data_path, ext):
    texts = []
    labels = []
    files = sorted(os.listdir(data_path))
    for fn in files:
        if fn.endswith(ext):
            tsvfn = fn.replace(ext, ".tsv")
            tsvfnp = os.path.join(data_path, tsvfn)
            if not os.path.exists(tsvfnp):
                print('Could not find {}, skipping {}...'.format(tsvfn, fn))
                continue
            with open(os.path.join(data_path, fn), encoding='utf-8') as fp:
                doc = fp.read().replace("\n", " ")
                texts.append(doc)
            with open(tsvfnp) as fp:
                labs = []
                for line in fp:
                    parts = line.split()
                    if parts[0].startswith(label_prefix):
                        labs.append(parts[0].replace("<", "").replace(">", ""))
                labels.append(labs)
    print('Read {} documents from [{}]'.format(len(texts), data_path))
    return texts, labels


def main(args):
    ds_path = args.dataset
    os.makedirs(ds_path, exist_ok=True)

    # train_texts = read_text_data(os.path.join(ds_path, train_txt_fname))

    dict_fname = os.path.join(ds_path, 'train.dict')
    print('Loading dictionary from [{}] ...'.format(dict_fname))
    dictionary = gensim.corpora.Dictionary.load(dict_fname)
    n_features = len(dictionary.token2id)
    print('n_features =', n_features)

    tfidf_fname = os.path.join(ds_path, 'train.tfidf.mm')
    print('Loading TF-IDF from [{}]'.format(tfidf_fname))
    loaded_tfidf = gensim.corpora.MmCorpus(tfidf_fname)
    tfidf = gensim.models.TfidfModel(loaded_tfidf)

    test_texts, test_labels = read_text_datadir(args.test_data, args.ext)
    test_texts = filter_words(test_texts, args.language)
    out_test_txt = 'test{}_raw_texts.txt'.format(args.extra_test)
    write_text_data(test_texts, os.path.join(ds_path, out_test_txt))

    print('Converting test data to bag-of-words format...')
    test_corpus = [dictionary.doc2bow(t) for t in test_texts]
    test_tfidf = tfidf[test_corpus]

    test_x = args.extra_test if args.extra_test != '' else 't'
    out_x_npz = 'X.ts{}.npz'.format(test_x)
    x_csr = corpus2csc(test_tfidf, num_terms=n_features).transpose().tocsr()
    write_csr(x_csr, os.path.join(ds_path, out_x_npz))

    subj_map = read_subjects_map(args.vocab)

    out_y_npz = 'Y.ts{}.npz'.format(test_x)
    y_csr = y_csr_matrix(test_labels, subj_map)
    write_csr(y_csr, os.path.join(ds_path, out_y_npz))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate test input files for X-Transformer in the '
        'correct format. For format specification, see:\n'
        'https://github.com/mvsjober/X-Transformer#running-x-transformer-on-customized-datasets')
    parser.add_argument('test_data', help='path to test data in Annif '
                        'full-text document corpus format')
    parser.add_argument('vocab', help='vocabulary file in TSV format')
    parser.add_argument('dataset', help='path to X-BERT dataset directory, '
                        'e.g., datasets/yso-en')
    parser.add_argument('language', help='fin|swe|eng')
    parser.add_argument('--extra_test', default='', nargs='?')
    parser.add_argument('--ext', default='.txt', nargs='?')
    args = parser.parse_args()
    args.language = parse_language(args.language)
    if args.language is None:
        print('ERROR: language [{}] not supported'.format(args.language))
        sys.exit(1)
    if args.ext[0] != '.':
        args.ext = '.' + args.ext

    main(args)
