#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm, inv
from scipy.sparse import csc_matrix, coo_matrix

from .misc import repmat, blkdiag


def get_mass_matrix(density, thick, elements, coordinates):
    elements = elements + 1
    M = np.zeros([576, elements.shape[0]])

    # 按照单元循环
    for ii in range(elements.shape[0]):
        # 获取当前单元的节点整体坐标
        X1, Y1, Z1 = coordinates[elements[ii, 0] - 1]
        X2, Y2, Z2 = coordinates[elements[ii, 1] - 1]
        X3, Y3, Z3 = coordinates[elements[ii, 2] - 1]
        X4, Y4, Z4 = coordinates[elements[ii, 3] - 1]
        # 将四边形拆成两个三角形计算面积
        L1 = ((X1 - X2)**2 + (Y1 - Y2)**2 + (Z1 - Z2)**2)**.5
        L2 = ((X3 - X2)**2 + (Y3 - Y2)**2 + (Z3 - Z2)**2)**.5
        L3 = ((X1 - X3)**2 + (Y1 - Y3)**2 + (Z1 - Z3)**2)**.5
        L = (L1 + L2 + L3) / 2
        Area1 = (L * (L - L1) * (L - L2) * (L - L3))**.5

        P1 = ((X1 - X3)**2 + (Y1 - Y3)**2 + (Z1 - Z3)**2)**.5
        P2 = ((X3 - X4)**2 + (Y3 - Y4)**2 + (Z3 - Z4)**2)**.5
        P3 = ((X1 - X4)**2 + (Y1 - Y4)**2 + (Z1 - Z4)**2)**.5
        P = (P1 + P2 + P3) / 2
        Area2 = (P * (P - P1) * (P - P2) * (P - P3))**.5

        Area = Area1 + Area2
        # 形成单元的集中质量矩阵，坐标变换并存入整体质量矩阵
        emass = Area * thick * density
        flag = [emass / 4, emass / 4, emass / 4,
                emass * thick**2 / 48, emass * thick**2 / 48, emass * thick**2 / 48]
        Me = np.diag(flag * 4)
        M[:, ii] = Me.reshape(-1)

    DOF = np.array([6 * elements[:, 0] - 5, 6 * elements[:, 0] - 4, 6 * elements[:, 0] - 3,
                    6 * elements[:, 0] - 2, 6 *
                    elements[:, 0] - 1, 6 * elements[:, 0],
                    6 * elements[:, 1] - 5, 6 * elements[:, 1] -
                    4, 6 * elements[:, 1] - 3,
                    6 * elements[:, 1] - 2, 6 *
                    elements[:, 1] - 1, 6 * elements[:, 1],
                    6 * elements[:, 2] - 5, 6 * elements[:, 2] -
                    4, 6 * elements[:, 2] - 3,
                    6 * elements[:, 2] - 2, 6 *
                    elements[:, 2] - 1, 6 * elements[:, 2],
                    6 * elements[:, 3] - 5, 6 * elements[:, 3] -
                    4, 6 * elements[:, 3] - 3,
                    6 * elements[:, 3] - 2, 6 * elements[:, 3] - 1, 6 * elements[:, 3]])
    I = repmat(DOF, 24, 1)
    J = np.kron(DOF, np.ones([24, 1]))

#    dof1 = np.array(list(set(elements.reshape(-1).tolist())))
#    dof2 = np.array([6 * dof1 - 5, 6 * dof1 - 4, 6 * dof1 - 3,
#                     6 * dof1 - 2, 6 * dof1 - 1, 6 * dof1]).T
#    dof = dof2.reshape(-1) - 1

    m = coo_matrix((M.reshape(-1), (I.reshape(-1) - 1, J.reshape(-1) - 1)),
                   shape=(6 * coordinates.shape[0], 6 * coordinates.shape[0]))
    m = m.tocsc()
#    m = m[dof][:, dof]
    m = (m + m.T) / 2
    return m


def get_stiffness_matrix(E, po, thick, elements, coordinates):
    elements = elements + 1
    W1 = np.ones(4)
    Q1 = 0.577350269189626 * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

    W0 = 0.01
    Walpha = 1 - W0 / 4
    alpha = (1 / (3 * Walpha))**0.5
    W2 = np.array([W0, Walpha, Walpha, Walpha, Walpha])
    Q2 = np.array([[0, 0], [-alpha, -alpha], [alpha, -alpha],
                   [-alpha, alpha], [alpha, alpha]])

    Dm = E * thick / (1 - po**2) * \
        np.array([[1, po, 0], [po, 1, 0], [0, 0, (1 - po) / 2]])
    Db = E * thick**3 / (12 * (1 - po**2)) * \
        np.array([[1, po, 0], [po, 1, 0], [0, 0, (1 - po) / 2]])
    Ds = (5 / 6) * thick * E / (2 * (1 + po)) * np.eye(2)
    gamma = E / (2000 * (1 + po))  # 罚函数因子取剪切模量的千分之一

    K = np.zeros([576, elements.shape[0]])

    # 按照单元循环
    for ii in range(elements.shape[0]):
        # 获取当前单元的节点整体坐标
        X1, Y1, Z1 = coordinates[elements[ii, 0] - 1]
        X2, Y2, Z2 = coordinates[elements[ii, 1] - 1]
        X3, Y3, Z3 = coordinates[elements[ii, 2] - 1]
        X4, Y4, Z4 = coordinates[elements[ii, 3] - 1]
    # 初始化矩阵
        kmb = np.zeros([26, 26])
        kmh = np.zeros([26, 26])
        kb = np.zeros([24, 24])
        ks = np.zeros([24, 24])
    # 计算当前单元的节点局部坐标并形成坐标变换矩阵
        v12 = np.array([X2 - X1, Y2 - Y1, Z2 - Z1])
        v13 = np.array([X3 - X1, Y3 - Y1, Z3 - Z1])
        v14 = np.array([X4 - X1, Y4 - Y1, Z4 - Z1])
        n3 = np.cross(v12, v14)
        n2 = np.cross(n3, v12)
        e1 = v12 / norm(v12)
        e2 = n2 / norm(n2)
        e3 = n3 / norm(n3)
        t = np.array([e1, e2, e3])
        T = blkdiag([t, t, t, t, t, t, t, t])

        x1 = 0
        y1 = 0
        x2 = e1.dot(v12)
        y2 = e2.dot(v12)
        x3 = e1.dot(v13)
        y3 = e2.dot(v13)
        x4 = e1.dot(v14)
        y4 = e2.dot(v14)
        x_e = np.array([x1, x2, x3, x4])
        y_e = np.array([y1, y2, y3, y4])

        x21 = x2 - x1
        y21 = y2 - y1
        x34 = x3 - x4
        y34 = y3 - y4
        x41 = x4 - x1
        y41 = y4 - y1
        x32 = x3 - x2
        y32 = y3 - y2
        y43 = y4 - y3
        y14 = y1 - y4
        x12 = x1 - x2
        x23 = x2 - x3
        # 计算弯曲和剪切刚度矩阵
        for jj in range(Q1.shape[0]):

            s = Q1[jj, 0]
            t = Q1[jj, 1]
            w = W1[jj]

            dnds = 1 / 4 * np.array([t - 1, 1 - t, 1 + t, -1 - t])
            dndt = 1 / 4 * np.array([s - 1, -1 - s, 1 + s, 1 - s])

            J11 = x_e.dot(dnds)
            J12 = y_e.dot(dnds)
            J21 = x_e.dot(dndt)
            J22 = y_e.dot(dndt)
            Jdet = J11 * J22 - J21 * J12
            Jv11 = J22 / Jdet
            Jv12 = -J12 / Jdet
            Jv21 = -J21 / Jdet
            Jv22 = J11 / Jdet
            Jv = np.array([[Jv11, Jv12], [Jv21, Jv22]])

            dndx = Jv11 * dnds + Jv12 * dndt
            dndy = Jv21 * dnds + Jv22 * dndt

            dn1dx = dndx[0]
            dn2dx = dndx[1]
            dn3dx = dndx[2]
            dn4dx = dndx[3]
            dn1dy = dndy[0]
            dn2dy = dndy[1]
            dn3dy = dndy[2]
            dn4dy = dndy[3]

            Bb = np.array([[0, 0, 0, 0, -dn1dx, 0, 0, 0, 0, 0,
                            -dn2dx, 0, 0, 0, 0, 0, -dn3dx, 0, 0, 0, 0, 0, -dn4dx, 0],
                           [0, 0, 0, dn1dy, 0, 0, 0, 0, 0, dn2dy, 0, 0,
                            0, 0, 0, dn3dy, 0, 0, 0, 0, 0, dn4dy, 0, 0],
                           [0, 0, 0, dn1dx, -dn1dy, 0, 0, 0, 0, dn2dx, -dn2dy,
                            0, 0, 0, 0, dn3dx, -dn3dy, 0, 0, 0, 0, dn4dx, -dn4dy, 0]])

            Bs = Jv.dot(np.array([[0, 0, -(1 - t) / 4, -y21 * (1 - t) / 8, x21 * (1 - t) / 8, 0, 0, 0, (1 - t) / 4, -y21 * (1 - t) / 8, x21 * (1 - t) / 8, 0,
                                   0, 0, (1 + t) / 4, -y34 * (1 + t) / 8, x34 * (1 + t) / 8, 0, 0, 0, -(1 + t) / 4, -y34 * (1 + t) / 8, x34 * (1 + t) / 8, 0],
                                  [0, 0, -(1 - s) / 4, -y41 * (1 - s) / 8, x41 * (1 - s) / 8, 0, 0, 0, -(1 + s) / 4, -y32 * (1 + s) / 8, x32 * (1 + s) / 8, 0,
                                   0, 0, (1 + s) / 4, -y32 * (1 + s) / 8, x32 * (1 + s) / 8, 0, 0, 0, (1 - s) / 4, -y41 * (1 - s) / 8, x41 * (1 - s) / 8, 0]]))
            kb = kb + Bb.T.dot(Db).dot(Bb) * Jdet * w
            ks = ks + Bs.T.dot(Ds).dot(Bs) * Jdet * w

        # 计算膜刚度矩阵
        for jj in range(Q2.shape[0]):

            s = Q2[jj, 0]
            t = Q2[jj, 1]
            w = W2[jj]

            n1 = (1 - s) * (1 - t) / 4
            n2 = (1 + s) * (1 - t) / 4
            n3 = (1 + s) * (1 + t) / 4
            n4 = (1 - s) * (1 + t) / 4

            dn1ds = (t - 1) / 4
            dn2ds = (1 - t) / 4
            dn3ds = (1 + t) / 4
            dn4ds = -(1 + t) / 4
            dn5ds = s * (t - 1)
            dn6ds = (1 - t**2) / 2
            dn7ds = -s * (1 + t)
            dn8ds = (t**2 - 1) / 2
            dn9ds = 2 * s * (t**2 - 1)
            dn1dt = (s - 1) / 4
            dn2dt = -(1 + s) / 4
            dn3dt = (1 + s) / 4
            dn4dt = (1 - s) / 4
            dn5dt = (s**2 - 1) / 2
            dn6dt = -t * (1 + s)
            dn7dt = (1 - s**2) / 2
            dn8dt = t * (s - 1)
            dn9dt = 2 * t * (s**2 - 1)

            dnds = np.array([dn1ds, dn2ds, dn3ds, dn4ds,
                             dn5ds, dn6ds, dn7ds, dn8ds, dn9ds])
            dndt = np.array([dn1dt, dn2dt, dn3dt, dn4dt,
                             dn5dt, dn6dt, dn7dt, dn8dt, dn9dt])
            dndsJ = np.array([dn1ds, dn2ds, dn3ds, dn4ds])
            dndtJ = np.array([dn1dt, dn2dt, dn3dt, dn4dt])
            J11 = x_e.dot(dndsJ)
            J12 = y_e.dot(dndsJ)
            J21 = x_e.dot(dndtJ)
            J22 = y_e.dot(dndtJ)
            Jdet = J11 * J22 - J21 * J12
            Jv11 = J22 / Jdet
            Jv12 = -J12 / Jdet
            Jv21 = -J21 / Jdet
            Jv22 = J11 / Jdet

            dndx = Jv11 * dnds + Jv12 * dndt
            dndy = Jv21 * dnds + Jv22 * dndt
            dn1dx = dndx[0]
            dn2dx = dndx[1]
            dn3dx = dndx[2]
            dn4dx = dndx[3]
            dn5dx = dndx[4]
            dn6dx = dndx[5]
            dn7dx = dndx[6]
            dn8dx = dndx[7]
            dn9dx = dndx[8]
            dn1dy = dndy[0]
            dn2dy = dndy[1]
            dn3dy = dndy[2]
            dn4dy = dndy[3]
            dn5dy = dndy[4]
            dn6dy = dndy[5]
            dn7dy = dndy[6]
            dn8dy = dndy[7]
            dn9dy = dndy[8]

            Bmb = np.array([[dn1dx, 0, dn1dy],
                            [0, dn1dy, dn1dx],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [(y14 * dn8dx - y21 * dn5dx) / 8, (x41 * dn8dy - x12 * dn5dy) / 8,
                             (y14 * dn8dy - y21 * dn5dy) / 8 + (x41 * dn8dx - x12 * dn5dx) / 8],
                            [dn2dx, 0, dn2dy],
                            [0, dn2dy, dn2dx],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [(y21 * dn5dx - y32 * dn6dx) / 8, (x12 * dn5dy - x23 * dn6dy) /
                             8, (y21 * dn5dy - y32 * dn6dy) / 8 + (x12 * dn5dx - x23 * dn6dx) / 8],
                            [dn3dx, 0, dn3dy],
                            [0, dn3dy, dn3dx],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [(y32 * dn6dx - y43 * dn7dx) / 8, (x23 * dn6dy - x34 * dn7dy) / 8,
                             (y32 * dn6dy - y43 * dn7dy) / 8 + (x23 * dn6dx - x34 * dn7dx) / 8],
                            [dn4dx, 0, dn4dy],
                            [0, dn4dy, dn4dx],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [(y43 * dn7dx - y14 * dn8dx) / 8, (x34 * dn7dy - x41 * dn8dy) / 8,
                             (y43 * dn7dy - y14 * dn8dy) / 8 + (x34 * dn7dx - x41 * dn8dx) / 8],
                            [dn9dx, 0, dn9dy],
                            [0, dn9dy, dn9dx]]).T

            Bmh = np.array([[-dn1dy / 2, dn1dx / 2, 0, 0, 0, (x41 * dn8dx - x12 * dn5dx) / 16 - (y14 * dn8dy - y21 * dn5dy) / 16 - n1,
                             -dn2dy / 2, dn2dx / 2, 0, 0, 0, (x12 * dn5dx - x23 * dn6dx) / 16 - (
                                 y21 * dn5dy - y32 * dn6dy) / 16 - n2,
                             -dn3dy / 2, dn3dx / 2, 0, 0, 0, (x23 * dn6dx - x34 * dn7dx) / 16 - (
                                 y32 * dn6dy - y43 * dn7dy) / 16 - n3,
                             -dn4dy / 2, dn4dx /
                             2, 0, 0, 0, (x34 * dn7dx - x41 * dn8dx) /
                             16 - (y43 * dn7dy - y14 * dn8dy) / 16 - n4,
                             -dn9dy / 2, dn9dx / 2]])

            kmb = kmb + Bmb.T.dot(Dm).dot(Bmb) * Jdet * w
            kmh = kmh + Bmh.T.dot(gamma).dot(Bmh) * Jdet * w

    # 内部自由度静力减缩、形成膜刚度矩阵
        kmbh = kmb + kmh
        kmii = kmbh[0:24, 0:24]
        kmoi = kmbh[24:26, 0:24]
        kmoo = kmbh[24:26, 24:26]
        km = kmii - kmoi.T.dot(inv(kmoo)).dot(kmoi)
    # 形成单元刚度矩阵、坐标变换并装入整体刚度矩阵
        ke = km + kb + ks
        Ke = T.T.dot(ke).dot(T)
        K[:, ii] = Ke.reshape(-1)

    DOF = np.array([6 * elements[:, 0] - 5, 6 * elements[:, 0] - 4, 6 * elements[:, 0] - 3, 6 * elements[:, 0] - 2, 6 * elements[:, 0] - 1, 6 * elements[:, 0],
                    6 * elements[:, 1] - 5, 6 * elements[:, 1] - 4, 6 * elements[:, 1] -
                    3, 6 * elements[:, 1] - 2, 6 *
                    elements[:, 1] - 1, 6 * elements[:, 1],
                    6 * elements[:, 2] - 5, 6 * elements[:, 2] - 4, 6 * elements[:, 2] -
                    3, 6 * elements[:, 2] - 2, 6 *
                    elements[:, 2] - 1, 6 * elements[:, 2],
                    6 * elements[:, 3] - 5, 6 * elements[:, 3] - 4, 6 * elements[:, 3] - 3, 6 * elements[:, 3] - 2, 6 * elements[:, 3] - 1, 6 * elements[:, 3]])
    I = repmat(DOF, 24, 1)
    J = np.kron(DOF, np.ones([24, 1]))

#    dof1 = np.array(list(set(elements.reshape(-1).tolist())))
#    dof2 = np.array([6 * dof1 - 5, 6 * dof1 - 4, 6 * dof1 - 3,
#                     6 * dof1 - 2, 6 * dof1 - 1, 6 * dof1]).T
#    dof = dof2.reshape(-1) - 1

    k = coo_matrix((K.reshape(-1), (I.reshape(-1) - 1, J.reshape(-1) - 1)),
                   shape=(6 * coordinates.shape[0], 6 * coordinates.shape[0]))
    k = k.tocsc()
#    k = k[dof][:, dof]
    k = (k + k.T) / 2
    return k


if __name__ == '__main__':
    from inp_processor import Inp_processor
    with open('Job-1.inp') as file:
        ip = Inp_processor(file)
    elements = ip.elements[0]
    coordinates = ip.nodes
    density = 7800
    thick = 0.01
    E = 210e9
    po = 0.3
    M = get_mass_matrix(7800, 0.01, elements, coordinates)
    K = get_stiffness_matrix(210e9, 0.3, 0.01, elements, coordinates)
