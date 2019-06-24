#!/usr/bin/env python3


import numpy as np
from scipy.sparse import coo_matrix


def get_elements_list(assembly):
    if hasattr(assembly,'elements'):
            return [assembly.elements]
    res = []
    for p in assembly.parts:
        res.extend(get_elements_list(p))
    return res

def shrink(assembly):
    nodes = assembly.nodes
    elements_list = get_elements_list(assembly)
    M, K = assembly.M, assembly.K
    used_nodes = np.zeros(len(nodes))
    for elements in elements_list:
        for element in elements:
            for idx in element:
                used_nodes[idx] += 1
    used_dof = np.array([used_nodes]*6).T.reshape(-1)
    idx = used_dof!=0
    M = M.tocsc()[idx][:,idx]
    K = K.tocsc()[idx][:,idx]
    return M, K


def make_nonsingular(M, eps=1e-10):
    """Note that this function will modify M"""
    for i in range(min(M.shape)):
        if M[i,i] == 0:
            M[i,i] = eps

def write_matrix(M, filename):
    """按 “行标 列标 值” 向文件写入稀疏矩阵。"""
    M = M.tocoo()
    with open(filename, 'w') as f:
        f.write('# Matrix size:\n{m} {n}\n'
                '# Order: index_row index_col data\n'.format(m=M.shape[0],n=M.shape[1]))
        for i in range(len(M.data)):
            f.write('{r} {c} {d}\n'.format(r=M.row[i]+1, c=M.col[i]+1,
                    d=M.data[i]))

def read_matrix(filename):
    """读取write_matrix写入的稀疏矩阵。"""
    with open(filename,'r') as f:
        data = []
        i, j = [], []
        for line in f:
            if line.startswith('#'):
                continue
            M, N = [int(i) for i in line.split()]
            break
        for line in f:
            if line.startswith('#'):
                continue
            ix_i, ix_j, data_ij = [eval(i) for i in line.split()]
            data.append(data_ij)
            i.append(ix_i-1)
            j.append(ix_j-1)
        mat = coo_matrix((data, (i, j)), shape=(M, N))
    return mat