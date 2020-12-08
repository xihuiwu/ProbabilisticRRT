import numpy as np
import math
from sklearn.gaussian_process import GaussianProcessRegressor
import pickle

import os
os.chdir( os.path.dirname( os.path.abspath(__file__) ) )

''' Map Parameters '''
lengthx = 1.4478
lengthy = 1.4478
nx,ny = (200,200)
dx = lengthx/nx
dy = lengthy/ny
x = np.linspace(0,lengthx,nx)
y = np.linspace(0,lengthy,ny)
xx,yy = np.meshgrid(x,y,sparse=False)
'''
mu = np.array([[0.8509,1.4478]]) # straight for RC car
mu = np.array([[0.0,0.8509]]) # left for RC car
mu = np.array([[1.4478,0.5969]]) # right for RC car
'''

mu = np.array([[0.7239,1.4478]]) # straight for wifibot
#mu = np.array([[0.0,0.7239]]) # left for wifibot
#mu = np.array([[1.4478,0.7239]]) # right for wifibot
sigma = np.array([[0.001,0.001]])
lamb = 1e8

''' RC car Parameters '''
L = 0.256
lf = L/2
lr = L/2
deltaMax = math.pi/6
vMax = 0.5
brake = 1 # m/s^2

''' Wifibot Parameters '''
W = 0.3302

''' Optimal Steer Parameters '''
dt = 0.1
w = [3,3,2,800,100]
bounds = [(-deltaMax,deltaMax), (0,vMax)]
length_scale = math.sqrt( lengthx**2 + lengthy**2 )

''' Tree Parameters '''
it = 80
thresh = 0.05

''' GPR Model '''
gprXv = pickle.load( open("gprXv_model.sav", "rb") )
gprYv = pickle.load( open("gprYv_model.sav", "rb") )

''' Involved Vehicle States '''
involved_states = np.loadtxt('involved_states.txt')
