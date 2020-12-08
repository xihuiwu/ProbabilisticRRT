#!/usr/bin/env python
#import rospy
#from std_msgs.msg import Float64MultiArray, Bool

from treelib import Node, Tree
import math
import numpy as np
import copy
import time

from car import Car
from optim_steer import kinematic
from grow_tree import growTree, find_plan, move_root
from collision_check import prune
from probability_map import generateMap, sample
import config

import matplotlib.pyplot as plt

def predict(car,enter_loc,enter_angle):
	theta = math.pi/2 - enter_angle
	x = car.state[0]*math.cos(enter_angle) - car.state[1]*math.sin(enter_angle) + enter_loc[0]
	y = car.state[0]*math.sin(enter_angle) + car.state[1]*math.cos(enter_angle) + enter_loc[1]
	print(x,y)
	
	xv_mean, xv_var = config.gprXv.predict([[x]], return_cov=True)
	yv_mean, yv_var = config.gprYv.predict([[y]], return_cov=True)
	
	return np.array([[car.state[0]+np.squeeze(xv_mean)*config.dt,car.state[1]+np.squeeze(yv_mean)*config.dt]]), np.array([[np.squeeze(xv_var)*config.dt,np.squeeze(yv_var)*config.dt]])

'''
def execute(command,pub,move_pub,rate):
	msg = Float64MultiArray()
	move_msg = Bool()
	move_msg.data = True
	move_pub.publish(move_msg)
	n = 1
	while n <= 10:
		msg.data = []
		msg.data.append(float(command[0]))
		msg.data.append(float(command[1]))
		pub.publish(msg)
		rate.sleep()
		n += 1
'''

if __name__ == '__main__':
	'''
	rospy.init_node('prrt',anonymous=True)
	pub = rospy.Publisher('/wifibot108/command',Float64MultiArray,queue_size=1)
	move_pub = rospy.Publisher('/wifibot107/move_flag',Bool,queue_size=1)
	rate = rospy.Rate(100) # 100 Hz
	'''
	
	''' Initialize ego vehicle '''
	#ego = Car(np.array([0.8509,0,math.pi/2,0]),1,0,0.17,np.array([0,0])) # RC car
	ego = Car(np.array([0.7239,0,math.pi/2,0]),1,0,0.17,np.array([0,0])) # wifibot
	current_speed = ego.state[3]

	''' Load involving vehicle position '''
	car1_state = config.involved_states

	''' Initialize involving vehicles, heading angle is only used for  '''
	cars = []
	car1 = Car(car1_state[0,:],1,0,0.17,[])
	cars.append(car1)
	enter_locs = [car1_state[0,0:2]] # left enterence relative to ego vehicle
	enter_angles = [car1_state[0,2]]
	
	''' Initialize tree '''
	tree = Tree()
	tree.create_node("0",0,data=copy.deepcopy(ego))
	
	''' Algorithm '''
	goal_idx = -1
	pass_flag = False
	init_flag = True
	state_count = 0
	#move_msg = Bool()
	start_time = time.time()
	control_commands = []
	while True:
		################################################################################
		if tree.get_node(goal_idx) != None and init_flag == False: # Execute for dt seconds
			#print('Executing plan!')
			control_plan,nodes = find_plan(tree, goal_idx, True)
			if len(control_plan) == 1:
				pass_flag = True
			#execute(control_plan[0],pub,rate)
			control_commands.append(control_plan[0])
			#print('executed for 0.1 sec')
			current_speed = (control_plan[0][0] + control_plan[0][1])/2
			ego.state = tree[nodes[0]].data.state
			plt.scatter(ego.state[0],ego.state[1])
			tree = move_root(tree,nodes[0])
			if pass_flag == True:
				print('Passed the intersection!')
				print('Used time %s'%(time.time()-start_time))
				break
		################################################################################	
		elif tree.get_node(goal_idx) == None and init_flag == False: # Brake for dt seconds
			print('Finding Plan!')
			current_speed -= config.brake*config.dt
			if current_speed <= 0:
				current_speed = 0
			#execute([current_speed,current_speed],pub,rate)
			control_commands.append(np.array([current_speed,current_speed]))
			#print('brake for 0.1 sec')
			u = [0,current_speed]
			state = kinematic(ego.state,ego.u_prev)
			ego.update(state,u)
			# Create a new tree
			tree = Tree()
			tree.create_node("0",0,data=copy.deepcopy(ego))
			goal_idx = -1
		
		################################################################################
		if len(cars) != 0:
			cars[0].update(car1_state[state_count,:],[0,0])
			plt.scatter(cars[0].state[0],cars[0].state[1],marker='x')
			state_count += 1
			
		if len(cars) == 0:
			pdfMap = generateMap(config.mu,config.sigma,config.lamb,False)
		else:
			mu = config.mu
			sigma = config.sigma
			for i in range(len(cars)):
				tempCar = copy.deepcopy(cars[i])
				tempMu,tempVar = predict(cars[i],enter_locs[i],enter_angles[i])
				tempCar.state[0] = tempMu[0,0]
				tempCar.state[1] = tempMu[0,1]
				mu = np.append(mu,tempMu,axis=0)
				sigma = np.append(sigma,tempVar,axis=0)
				tree = prune(tree,cars[i],tempCar)
				if tree.get_node(goal_idx) == None:
					goal_idx = -1
			pdfMap = generateMap(mu,sigma,config.lamb,False)
			
		# Remove involved vehicle if it passed the intersection
		if len(cars) != 0:
			if state_count == car1_state.shape[0]:
				cars.pop(0)
		
		################################################################################
		if goal_idx == -1:
			if init_flag == True:
				init_flag = False
			tree, goal_idx = growTree(tree,pdfMap,config.mu,cars,False,False)
	
	#plt.show()
	print(control_commands)
	
	'''
	for i in control_commands:
		execute(i,pub,move_pub,rate)
	'''

