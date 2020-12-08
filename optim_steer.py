import numpy as np
import math
from scipy import optimize
import time
import sympy
from matplotlib import pyplot as plt

import config

'''
u is 1x2 vector
'''
def ackermann_to_differential(u):
	speed_control = np.empty(2)
	if u[0] == 0:
		speed_control[0] = u[1]/2
		speed_control[1] = u[1]/2
	else:
		omega = u[1]/math.sqrt( config.lr**2 + (config.L*sympy.cot(u[0]))**2 )
		radius = abs(config.L/math.tan(u[0]))
		if u[0] > 0: # turn left
			speed_control[0] = omega*(radius-config.W/2)
			speed_control[1] = omega*(radius+config.W/2)
		else: # turn right
			speed_control[0] = omega*(radius+config.W/2)
			speed_control[1] = omega*(radius-config.W/2)
	return speed_control
	

'''
state is 1x4 vector
control input is 1x2 vector
'''
def kinematic(state, u):
	# u[0] = delta, u[1] = a
	# state[0] = x, state[1] = y, state[2] = psi, state[3] = v
	if abs(u[0]) > config.deltaMax:
		u[0] = np.sign(u[0])*config.deltaMax

	newState = np.zeros_like(state)
	beta = math.atan(config.lr * math.tan(u[0]) / config.L)
	newState[0] = state[0] + state[3] * math.cos(state[2] + beta) * config.dt
	newState[1] = state[1] + state[3] * math.sin(state[2] + beta) * config.dt
	newState[2] = state[2] + state[3] / config.lr * math.sin(beta) * config.dt
	newState[3] = u[1]

	return newState

'''
state, 1x4 vector
u is the variable that needs minimization, 1x2 vector
u_prev is the previous control input, 1x2 vector
sample is sampled position, 1x2 vector
cars is list of involving vehicles
w is the weight matrix
'''
def optimControl(u, u_prev, state, sample, cars):
	newState = kinematic(state, u)
	refState = kinematic(state, [0,u_prev[1]])
	refDist = math.sqrt((refState[0] - sample[0])**2 + (refState[1] - sample[1])**2)

	if u[0] == 0:
		radius2 = float("inf")
	else:
		radius2 = config.lr**2 + (config.L*sympy.cot(u[0]))**2

	cost = config.w[0] * ( ((u[0] - u_prev[0])/(config.deltaMax*2))**2 + ((u[1] - u_prev[1])/(config.vMax/2))**2 ) + \
			config.w[1] * state[3]**4/radius2 + \
			config.w[2] * ( (newState[3] - config.vMax)/config.vMax )**2 + \
			config.w[3] * ( math.sqrt( (newState[0] - sample[0])**2 + (newState[1] - sample[1])**2 ) - refDist)

	temp = 0
	if len(cars) != 0:
		for i in range(len(cars)):
			temp += math.sqrt( (newState[0]-cars[i].state[0])**2 + (newState[1]-cars[i].state[1])**2 )
		cost += config.w[4]*config.length_scale*len(cars) / temp
	else:
		cost += 0

	return cost

def optimSteer(u0, u_prev, state, sample, cars):
	res = optimize.minimize(optimControl, u0, args=(u_prev,state,sample,cars), method='SLSQP', bounds=config.bounds, options={'disp':False})
	#res = optimize.minimize(optimControl, u0, args=(u_prev,state,sample,positionT), method='L-BFGS-B', bounds=bounds)
	newState = kinematic(state, res.x)
	return res.x,newState

if __name__ == '__main__':
	u0 = np.array([0.0, 0.0])
	u_prev = np.array([0.0, 0.32])
	state = np.array([0.8509, 0.0, math.pi/2, 0.32])
	sample = np.array([0.8509,1.4478])
	positionT = []
	start_time = time.time()
	u,newState = optimSteer(u0, u_prev, state, sample, cars)
	print("--- %s seconds ---"%(time.time()-start_time))
	print(u)
	print(state)
	print(newState)

