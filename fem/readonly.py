#!/usr/bin/env python3


class Readonly:
    def __init__(self):
        self.__dict__['_readonly'] = False

    def _set_readonly(self):
        self._readonly = True

    def __setattr__(self, name, value):
        if self._readonly:
            raise AttributeError('cannot set attribute of Readonly object')
        else:
            super().__setattr__(name, value)