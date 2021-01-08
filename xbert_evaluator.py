#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import pickle
import numpy as np
import scipy as sp
import scipy.sparse as smat
from xbert.rf_util import (
    smat_util,
)

def sigmoid(v, inplace=False):
    if inplace:
        out = v
    else:
        out = np.zeros_like(v)
    out[:] = 1.0 / (1.0 + np.exp(-v))
    return out

def generate(tY, pY, topk=10):
    assert isinstance(tY, smat.csr_matrix), type(tY)
    assert isinstance(pY, smat.csr_matrix), type(pY)
    assert tY.shape == pY.shape, "tY.shape = {}, pY.shape = {}".format(tY.shape, pY.shape)
    pY = smat_util.sorted_csr(pY)
    total_matched = np.zeros(topk, dtype=sp.uint64)
    recall = np.zeros(topk, dtype=sp.float64)
    for i in range(tY.shape[0]):
        truth = tY.indices[tY.indptr[i] : tY.indptr[i + 1]]
        matched = np.isin(pY.indices[pY.indptr[i] : pY.indptr[i + 1]][:topk], truth)
        cum_matched = np.cumsum(matched, dtype=sp.uint64)
        total_matched[: len(cum_matched)] += cum_matched
        recall[: len(cum_matched)] += cum_matched / (len(truth) + 1e-9)
        if len(cum_matched) != 0:
            total_matched[len(cum_matched) :] += cum_matched[-1]
            recall[len(cum_matched) :] += cum_matched[-1] / (len(truth) + 1e-9)
    prec = total_matched / tY.shape[0] / np.arange(1, topk + 1)
    recall = recall / tY.shape[0]
    return prec, recall

def main(args):
    Y_true = smat.load_npz(args.ytrue)
    Y_pred = smat.load_npz(args.ypred)
    Y_pred.data = sigmoid(Y_pred.data)

    if args.indices is None:
        pr, rc = generate(Y_true, Y_pred)

    else:
        Y_ind = np.loadtxt(args.indices, dtype="int")
        #print(Y_ind.shape)

        indmax = np.max(Y_ind)
        #print(indmax, type(indmax))
        Y_true_new = np.zeros((indmax+1, Y_true.shape[1]), dtype="int")
        Y_pred_new = np.zeros((indmax+1, Y_pred.shape[1]), dtype="float32")

        for i in range(indmax+1):
            mask = Y_ind==i
            Ytm = Y_true[mask]
            Y_true_new[i,:]=Ytm.getrow(0).toarray()
            Ypm = Y_pred[mask]
            if args.limit is not None:
                Ypm = Ypm[:(args.limit),:]
            #nrow = np.minimum(0, Ytm.shape[0]-1)
            #Y_pred_new[i,:]=Ypm.getrow(nrow).toarray()
            Ysum = smat.csr_matrix.sum(Ypm, axis=0)
            Y_pred_new[i,:]=Ysum
    
        #print(Y_true_new.max(), Y_pred_new.max())

        Y_true_new = smat.csr_matrix(Y_true_new)
        Y_pred_new = smat.csr_matrix(Y_pred_new)
        #print(Y_true_new.shape, Y_pred_new.shape)

        #Y_pred_new.data = sigmoid(Y_pred_new.data)
        pr, rc = generate(Y_true_new, Y_pred_new)

    print('PRECISION:', pr)
    print('RECALL:   ', rc)

if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y", "--ytrue", type=str, required=True,
        help="path to the npz file of the truth label matrix (CSR) for computing metrics",
    )
    parser.add_argument(
        "-p", "--ypred", type=str, required=True,
        help="path to the npz file of the sorted prediction (CSR)",
    )
    parser.add_argument(
        "-i", "--indices", type=str, 
        help="path to the indices file for summing predictions",
    )
    parser.add_argument(
        "-l", "--limit", type=int, metavar='N',
        help="limit the averaging to N first predictions",
    )
    args = parser.parse_args()
    print(args)
    main(args)
