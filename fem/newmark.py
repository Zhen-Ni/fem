#!/usr/bin/env python3

import numpy as np
from scipy.sparse.linalg import inv

class IntegratorNewmark:
    """《有限单元法》P480"""
    def __init__(self, M, C, K, dt, t0=0,func_force=0, alpha=0.25, delta=0.5,
                 x0=0, dx0=0):
        self._N = M.shape[0]
        self._M = M
        self._C = C
        self._K = K
        self._t0 = t0
        self._dt = dt
        self.set_force(func_force)
        self.set_initial_conditions(x0, dx0)
        self.set_coefficients(alpha, delta)
        self._is_initialized = False

    def set_coefficients(self, alpha, delta):
        self._alpha = alpha
        self._delta = delta

    def set_force(self, func_force=0):
        if not callable(func_force):
            f = np.zeros(self._N)
            f[:] = func_force
            def func_force(t):
                return f
        self._func_force = func_force
        return self

    def set_initial_conditions(self, x0=None, dx0=None):
        N = self._N
        if x0 is 0:
            x0 = np.zeros(N)
        if dx0 is 0:
            dx0 = np.zeros(N)
        if x0 is not None:
            self._x0 = x0.copy()
        if dx0 is not None:
            self._dx0 = dx0.copy()
        return self

    def _initialize(self):
        if self._is_initialized:
            return
        self._t = self._t0
        ddx0 = inv(self._M).dot(self._func_force(self._t) - self._C.dot(self._dx0) -
                                self._K.dot(self._x0))
        alpha = self._alpha
        delta = self._delta
        dt = self._dt
        self._c0 = 1 / (alpha *dt**2)
        self._c1 = delta / (alpha*dt)
        self._c2 = 1 / (alpha * dt)
        self._c3 = 1 / (2*alpha) - 1
        self._c4 = delta / alpha - 1
        self._c5 = dt / 2 * (delta / alpha - 2)
        self._c6 = dt * (1-delta)
        self._c7 = delta * dt
        Ke = self._K + self._c0 * self._M + self._C
        self._invKe = inv(Ke)
        self._x = self._x0
        self._dx= self._dx0
        self._ddx = ddx0
        self._is_initialized = True


    def integrate(self):
        self._initialize()
        dt = self._dt
        self._t = self._t + dt
        t = self._t
        c0 = self._c0
        c1 = self._c1
        c2 = self._c2
        c3 = self._c3
        c4 = self._c4
        c5 = self._c5
        c6 = self._c6
        c7 = self._c7
        x = self._x
        dx = self._dx
        ddx = self._ddx
        M = self._M
        C = self._C
        invKe = self._invKe
        Qe = self._func_force(t)
        Qe = Qe + M.dot(c0*x+c2*dx+c3*ddx) + C.dot(c1*x+c4*dx+c5*ddx)
        x = invKe.dot(Qe)
        ddx = c0 * (x-self._x) - c2*self._dx - c3*self._ddx
        dx = self._dx+c6*self._ddx+c7*ddx
        self._x = x
        self._dx = dx
        self._ddx = ddx
        return t, x, dx, ddx





