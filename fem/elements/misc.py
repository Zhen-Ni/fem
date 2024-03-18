#!/usr/bin/env python3

import numpy as np

def repmat(mat, i, j):
    M, N = mat.shape
    res = np.zeros([M * i, N * j], dtype=mat.dtype)
    for ii in range(i):
        for jj in range(j):
            res[M * ii:M * ii + M, N * jj:N * jj + N] = mat
    return res


def blkdiag(_arrays):
    arrays = []
    for i in range(len(_arrays)):
        if len(_arrays[i].shape) == 1:
            arrays.append(_arrays[i].reshape(1, -1))
        else:
            arrays.append(_arrays[i])
    Ms = [i.shape[0] for i in arrays]
    Ns = [i.shape[1] for i in arrays]
    res = np.zeros([np.sum(Ms), np.sum(Ns)])
    M_start = 0
    N_start = 0
    for i in range(len(arrays)):
        M, N = Ms[i], Ns[i]
        res[M_start:M_start + M, N_start:N_start + N] = arrays[i]
        M_start += M
        N_start += N
    return res
