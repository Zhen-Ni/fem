#!/usr/bin/env python3


import numpy as np
from scipy.sparse.linalg import eigsh, spsolve, inv

from .misc import make_nonsingular
from .newmark import IntegratorNewmark
from .dataset import VectorField

class Solver:
    def __init__(self, model):
        self._model = model
        self._result = None

    def __repr__(self):
        return "Solver object"

    @property
    def result(self):
        return self._result
    @property
    def model(self):
        return self._model

    def solve(self):
        assert 0, "overwrite it with your own function here"


class SolverModal(Solver):
    def __init__(self, model, order=20, frequency_shift=0.0):
        super().__init__(model)
        self._order = order
        self._frequency_shift = frequency_shift

    def __repr__(self):
        return "SolverModal object"

    def set_order(self, order):
        self._order = order
    def set_frency_shift(self, frequency_shift):
        self._frequency_shift = frequency_shift

    def solve(self):
        K = self._model.K
        M = self._model.M
        freq, mode_shape = eigsh(K, self._order, M, self._frequency_shift,)
#                                v0=np.ones(K.shape[0]))
        freq = np.sqrt(freq) / 2 / np.pi
        index_array = np.argsort(freq.real)
        step = Step()
        for idx in index_array:
            f = freq[idx]
            x = mode_shape.T[idx]
            step.append(FrameModal(self._model, {'displacement':x}, f))
        self._result = step
        return step


class SolverStatic(Solver):
    def __init__(self, model):
        super().__init__(model)

    def solve(self):
        res = spsolve(self._model.K, self._model.F)
        return Step([FrameStatic(self._model, {'displacement':res}, None)])


class SolverHarmonic(Solver):
    def __init__(self, model,freqs):
        super().__init__(model)
        self._freqs = freqs

    def set_frequency(self, freqs):
        self._freqs = freqs

    def solve(self):
        M = self._model.M
        K = self._model.K
        step = Step()
        for f in self._freqs:
            w = 2*np.pi*f
            A = - w**2 * M + K
            res = spsolve(A, self._model.F)
            step.append(FrameHarmonic(self._model, {'displacement':res}, f))
        return step


class SolverDynamic(Solver):
    def __init__(self, model, t,func_force=None, x0=0, dx0=0):
        super().__init__(model)
        self._t = t
        self._method = 'newmark'
        self._func_force = func_force
        self._x0 = None
        self._dx0 = None
        self.set_initial_conditions(x0, dx0)
        self._func_callback = None

    def set_t(self, t):
        self._t = t

    def set_method(self,name):
        if name.lower() == 'newmark':
            self._method = 'newmark'
        else:
            raise ValueError('unknown method {name}'.format(name=name))

    def set_force(self, func_force=None):
        self._func_force = func_force

    def set_initial_conditions(self, x0=None, dx0=None):
        N = len(self.model.nodes) * 6
        if x0 is 0:
            x0 = np.zeros(N)
        if dx0 is 0:
            dx0 = np.zeros(N)
        if x0 is not None:
            self._x0 = x0
        if dx0 is not None:
            self._dx0 = dx0

    def set_callback(self, func=None):
        self._func_callback = func


    def _solve_newmark(self, alpha=0.25, delta=0.5):
        eps_dt = 1e-10
        step = Step()
        M = self._model.M.copy().tocsc()
        K = self._model.K
        C = self._model.C
        make_nonsingular(M)
#        # 检查时间间隔是否相等
#        assert (np.abs(np.diff(self._t,2)) < 1e-10).all()
#        dt = self._t[1] - self._t[0]
#
#        func_force = self._func_force
#        if func_force is None:
#            func_force = self._model.F
#        integrator = IntegratorNewmark(M,C,K,dt,func_force,
#                                       alpha,delta, self._x0, self._dx0)
#        step.append(FrameDynamic(self.model,0.0,self._x0))
#
#        for i in range(len(self._t)-1):
#            t, x, dx, ddx = integrator.integrate()
#            step.append(FrameDynamic(self.model,t,x))
#        return step
        func_force = self._func_force
        if func_force is None:
            def func_force(t):
                return self._model.F

        x = self._x0
        dx = self._dx0
        t = self._t[0]
        dt = self._t[1] - t
        integrator = IntegratorNewmark(M,C,K,dt,t,func_force, alpha,delta,
                                               x, dx)
        step.append(FrameDynamic(self.model,{'displacement':x,'velocity':dx},t))
        for i,t in enumerate(self._t[1:],1):
            if not abs(t - self._t[i-1] - dt) < eps_dt:
                dt = t - self._t[i-1]
                integrator = IntegratorNewmark(M,C,K,dt,self._t[i-1],
                                               func_force, alpha,delta, x, dx)
            t, x, dx, ddx = integrator.integrate()
            frame = FrameDynamic(self.model,{'displacement':x,'velocity':dx},t)
            if self._func_callback is not None:
                self._func_callback(i,frame)
            step.append(frame)
        return step



    def solve(self, **kwargs):
        if self._method == 'newmark':
            step = self._solve_newmark(**kwargs)

        return step

class Step:
    def __init__(self, frames=[]):
        self._frames = list(frames)

    def append(self, frame):
        self._frames.append(frame)

    def __repr__(self):
        return "Step object with {n} frames".format(n=len(self._frames))

    def __getitem__(self, *args, **kwargs):
        return self._frames.__getitem__(*args, **kwargs)

    def __getattr__(self, name):
        result = []
        for f in self._frames:
            result.append(getattr(f, name))
        return np.array(result)

    def to_dataset(self, *args, **kwargs):
        """Convert each frame to datasets."""
        dss = [f.to_dataset(*args, **kwargs) for f in self._frames]
        return dss
        


class Frame:
    def __init__(self, model, fields, label=None, label_name=''):
        self._model = model
        self._label = label
        self._fields = fields
        self._label_name = label_name

    def __repr__(self):
        return "Frame object"
        

    def __getattr__(self, name):
        if name.lower() == self._label_name:
            return self._label
        res =  self._fields.get(name)
        if res is not None:
            return res
        raise AttributeError("Frame object has no attriburte"
                             " '{name}'".format(name=name))

    @property
    def x(self):
        return self.displacement[::6]
    @property
    def y(self):
        return self.displacement[1::6]
    @property
    def z(self):
        return self.displacement[2::6]
    @property
    def vx(self):
        return self.velocity[::6]
    @property
    def vy(self):
        return self.velocity[1::6]
    @property
    def vz(self):
        return self.velocity[2::6]
    @property
    def ax(self):
        return self.acceleration [::6]
    @property
    def ay(self):
        return self.acceleration [1::6]
    @property
    def az(self):
        return self.acceleration [2::6]
    
    def to_dataset(self, which='all'):
        """Write frame information to dataset.
        
        Parameters
        ----------
        which: 'all' | 'mesh' | 'spring' | 'mass'
            Define which part of the assembly to export.
        
        Returns
        -------
        ds: DataSet
            A DataSet object containing geometry information of the assembly.
        """
        ds = self._model.to_dataset(which)
        ds.time = self._label
        displacement = self._fields.get('displacement')
        if displacement is not None:
            data = displacement.reshape(-1,6)[:,:3]
            field = VectorField('displacement', data)
            ds.point_data['displacement'] = field
            data = displacement.reshape(-1,6)[:,3:]
            field = VectorField('displacement-rotation', data)
            ds.point_data['displacement-rotation'] = field
        velocity = self._fields.get('velocity')
        if velocity is not None:
            data = velocity.reshape(-1,6)[:,:3]
            field = VectorField('velocity', data)
            ds.point_data['velocity'] = field
            data = velocity.reshape(-1,6)[:,3:]
            field = VectorField('velocity-rotation', data)
            ds.point_data['velocity-rotation'] = field
        acceleration = self._fields.get('acceleration')
        if acceleration is not None:
            data = acceleration.reshape(-1,6)[:,:3]
            field = VectorField('acceleration', data)
            ds.point_data['acceleration'] = field
            data = acceleration.reshape(-1,6)[:,3:]
            field = VectorField('acceleration-rotation', data)
            ds.point_data['acceleration-rotation'] = field
        return ds



class FrameModal(Frame):
    def __init__(self, model, fields, label):
        super().__init__(model, fields, label, 'frequency')

    def __repr__(self):
        return "Frame object at {freq}Hz".format(freq=self._label)

class FrameStatic(Frame):
    pass

class FrameHarmonic(Frame):
    def __init__(self, model, fields, label):
        super().__init__(model, fields, label, 'frequency')
        w = label * 2 * np.pi
        self._fields['velocity'] = self._fields['displacement'] * 1j * w
        self._fields['acceleration'] = self._fields['displacement'] * w**2

    def __repr__(self):
        return "Frame object at {freq}Hz".format(freq=self._label)

class FrameDynamic(Frame):
    def __init__(self, model, fields, label):
        super().__init__(model, fields, label, 'time')

    def __repr__(self):
        return "Frame object at {time}s".format(freq=self._label)



