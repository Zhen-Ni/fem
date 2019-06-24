#!/usr/bin/env python3

import os
import sys
os.chdir('..')
sys.path.append(os.curdir)

from fem import *


def test_vtk():    
    points = [(Point(0,0,0)),(Point(1,0,0)),(Point(2,0,0)),
              (Point(0,1,0)),(Point(1,1,0)),(Point(2,1,0)),
              (Point(1,2,0)),(Point(2,2,0))]
    cells = [Cell([0,1,4,3],9),Cell([1,2,5,4],9),Cell([4,5,7,6],9)]
    
    temp = ScalarField('temprature', [0,1,2,1,2,3,3,4])
    temp2 = ScalarField('temprature2', [0,-1,2,-1,2,3,-3,4])
    disp = VectorField('displacement',[[0,0,0],[0,0,1],[0,0,0],
                                       [0,0,1],[0,0,2],[0,0,1],
                                       [0,0,1],[0,0,0]])
    coor = VectorField('coordinate', [[0,0],[1,0],[2,0],
                                      [0,1],[1,1],[2,1],
                                      [1,2],[2,2]], 2)
    dataset = DataSet(points, cells, 'test',1.2)
    dataset.point_data['temp'] = temp
    dataset.point_data['temp2'] = temp2
    dataset.point_data['disp'] = disp
    dataset.point_data['coor'] = coor
    
    density = ScalarField('density', [1,1.5,2])
    dataset.cell_data['rho'] = density
    
    dataset2 = DataSet(points, cells, 'test',2.5)
    dataset2.point_data['temp'] = temp
    dataset2.point_data['disp'] = disp
    dataset2.point_data['coor'] = coor
    
    density = ScalarField('density', [1,1.5,2])
    dataset2.cell_data['rho'] = density
    
    slf = dumpvtk([dataset, dataset2], 'generated/dataset_test_vtk')

def test_vtu():    
    points = [(Point(0,0,0)),(Point(1,0,0)),(Point(2,0,0)),
              (Point(0,1,0)),(Point(1,1,0)),(Point(2,1,0)),
              (Point(1,2,0)),(Point(2,2,0))]
    cells = [Cell([0,1,4,3],9),Cell([1,2,5,4],9),Cell([4,5,7,6],9)]
    
    temp = ScalarField('temprature', [0,1,2,1,2,3,3,4])
    temp2 = ScalarField('temprature2', [0,-1,2,-1,2,3,-3,4])
    disp = VectorField('displacement',[[0,0,0],[0,0,1],[0,0,0],
                                       [0,0,1],[0,0,2],[0,0,1],
                                       [0,0,1],[0,0,0]])
    coor = VectorField('coordinate', [[0,0],[1,0],[2,0],
                                      [0,1],[1,1],[2,1],
                                      [1,2],[2,2]], 2)
    dataset = DataSet(points, cells, 'test',1.2)
    dataset.point_data['temp'] = temp
    dataset.point_data['temp2'] = temp2
    dataset.point_data['disp'] = disp
    dataset.point_data['coor'] = coor
    
    density = ScalarField('density', [1,1.5,2])
    dataset.cell_data['rho'] = density
    
    dataset2 = DataSet(points, cells, 'test',2.5)
    dataset2.point_data['temp'] = temp
    dataset2.point_data['disp'] = disp
    dataset2.point_data['coor'] = coor
    
    density = ScalarField('density', [1,-1.5,2])
    dataset2.cell_data['rho'] = density
    
    slf = dumpvtu([dataset, dataset2], 'dataset_test_vtu', 'generated')

def test_pvd():    
    points = [(Point(0,0,0)),(Point(1,0,0)),(Point(2,0,0)),
              (Point(0,1,0)),(Point(1,1,0)),(Point(2,1,0)),
              (Point(1,2,0)),(Point(2,2,0))]
    cells = [Cell([0,1,4,3],9),Cell([1,2,5,4],9),Cell([4,5,7,6],9)]
    
    temp = ScalarField('temprature', [0,1,2,1,2,3,3,4])
    temp2 = ScalarField('temprature2', [0,-1,2,-1,2,3,-3,4])
    disp = VectorField('displacement',[[0,0,0],[0,0,1],[0,0,0],
                                       [0,0,1],[0,0,2],[0,0,1],
                                       [0,0,1],[0,0,0]])
    coor = VectorField('coordinate', [[0,0],[1,0],[2,0],
                                      [0,1],[1,1],[2,1],
                                      [1,2],[2,2]], 2)
    dataset = DataSet(points, cells, 'test',1.2)
    dataset.point_data['temp'] = temp
    dataset.point_data['temp2'] = temp2
    dataset.point_data['disp'] = disp
    dataset.point_data['coor'] = coor
    
    density = ScalarField('density', [1,1.5,2])
    dataset.cell_data['rho'] = density
    
    dataset2 = DataSet(points, cells, 'test',2.5)
    dataset2.point_data['temp'] = temp
    dataset2.point_data['disp'] = disp
    dataset2.point_data['coor'] = coor
    
    density = ScalarField('density', [1,-1.5,2])
    dataset2.cell_data['rho'] = density
    
    slf = dumppvd([dataset, dataset2], 'dataset_test_pvd', 'generated')

if __name__ == '__main__':
    test_vtk()
    test_vtu()
    test_pvd()
    