from treelib import Node, Tree
import numpy as np
import math
import copy
from matplotlib import pyplot as plt
import time

import config
from car import Car
from optim_steer import optimSteer, ackermann_to_differential
from probability_map import generateMap, sample

def diff_kinematic(state,u):
	newState = np.empty(4)
	newState[3] = (u[0]+u[1])/2
	omega = (u[1]-u[0])/config.L
	theta = omega*config.dt
	if u[0] == u[1]:
		radius = float('inf')
	else:
		radius = u[1]/omega
		
	iccx = state[0] - radius*math.sin(state[2])
	iccy = state[1] + radius*math.cos(state[2])
	
	newState[0] = math.cos(theta)*(state[0]-iccx) - math.sin(theta)*(state[1]-iccy) + iccx
	newState[1] = math.sin(theta)*(state[0]-iccx) + math.cos(theta)*(state[1]-iccy) + iccy
	newState[2] = state[2] + theta
	return newState

def growTree(tree,pdfMap,mu,cars,print_control=False,plot=False):
	# here mu is just the destination coordinate
	tree_backup = Tree(tree.subtree(tree.root),deep=True)
	n = 0
	goal_idx = -1
	plan_flag = False
	mu = np.squeeze(mu) # chang mu from matrix type to array type
	while n < config.it:
		''' Sample '''
		pt = sample(pdfMap)
		if plot == True:
			plt.scatter(pt[0],pt[1],marker='x')

		''' Find the nearest node '''
		minDist = float("inf")
		for i in range(n+1):
			dist = math.sqrt( (pt[0]-tree[i].data.state[0])**2 + (pt[1]-tree[i].data.state[1])**2 )
			if dist < minDist:
				idx = i
				minDist = dist

		''' Optimal Steer '''
		u,newState = optimSteer([0.0,0.0], tree[idx].data.u_prev, tree[idx].data.state, pt, cars)
		newEgo = copy.deepcopy(tree[idx].data)
		newEgo.update(newState,u)
		if print_control == True:
			print(u)

		''' Check whether the new node is out of the intersection area '''
		if newState[0] >= 0 and newState[0] <= config.lengthx and newState[1] >= 0 and newState[1] <= config.lengthy:
			''' Check whether the new node and the nearest nodehas the same coordinate '''
			if np.array_equal(tree[idx].data.state[0:2],newState[0:2]):
				tree[idx].data = newEgo
			else:
				n += 1
				tree.create_node(str(n),n,parent=idx,data=newEgo)

			''' Threshold Check '''
			if math.sqrt( (newEgo.state[0]-mu[0])**2 + (newEgo.state[1]-mu[1])**2 ) < config.thresh:
				plan_flag = True
				break
		else:
			break

	if plan_flag == False:
		print('Not able to find a motion plan!!!')
		#tree = tree_backup
	else:
		print('Successfully find a motion plan!!!')
		''' Steer to destination '''
		u,newState = optimSteer(tree[n].data.u_prev, tree[n].data.u_prev, tree[n].data.state, mu, cars)
		newEgo = copy.deepcopy(tree[idx].data)
		newEgo.update(newState,u)
		tree.create_node(str(n+1),n+1,parent=n,data=newEgo)
		goal_idx = n+1
	
	if plot == True:
		for j in tree.all_nodes_itr():
			plt.scatter(j.data.state[0]-config.lr*math.cos(j.data.state[2]),j.data.state[1]-config.lr*math.sin(j.data.state[2]))
		plt.scatter(mu[0],mu[1],marker='*')
		if len(cars) != 0:
			for k in range(len(cars)):
				plt.scatter(cars[k].state[0],cars[k].state[1],marker='v')
		plt.xlim(0,config.lengthx)
		plt.ylim(0,config.lengthy)
		plt.show()

	return tree, goal_idx

def find_plan(tree,goal_idx,convert=False):
	current_idx = goal_idx
	parent_idx = -1
	nodes = []
	while parent_idx != tree.root:
		parent_idx = tree.parent(current_idx).identifier
		nodes.append(current_idx)
		current_idx = parent_idx
	nodes.append(tree.root)
	nodes.reverse()
	
	control_plan = []
	if convert == True:
		for i in range(1,len(nodes)):
			node = nodes[i]
			control_input = np.array([tree[nodes[i]].data.u_prev[0],tree[nodes[i-1]].data.state[3]])
			control_plan.append(ackermann_to_differential(control_input))
	else:
		control_plan = []
		
	return control_plan, nodes[1:]
	
def move_root(tree,node):
	newTree = Tree(tree.subtree(node),deep=True)
	'''
	for i in newTree.all_nodes_itr():
		newIdx = i.identifier - node
		newTree.update_node(i.identifier,identifier=newIdx,tag=str(newIdx))
	'''
	return newTree

if __name__ == '__main__':
	''' Initialize ego vehicle '''
	#ego = Car(np.array([0.8509,0,math.pi/2,0]),1,0,0.17,np.array([0,0])) # RC car
	ego = Car(np.array([0.7239,0,math.pi/2,0]),1,0,0.17,np.array([0,0])) # wifibot

	''' Initialize tree '''
	tree = Tree()
	tree.create_node("0",0,data=ego)

	''' Generate Probability Position Map '''
	pdfMap = generateMap(config.mu,config.sigma,config.lamb,False)

	''' Initialize involving vehicles '''
	cars = []
	car1 = Car(np.array([0.5,0.7239,math.pi,0]),1,0,0.17,[])
	cars.append(car1)

	''' Grow tree '''
	start_time = time.time()
	tree, goal_idx = growTree(tree,pdfMap,config.mu,cars,False,True)
	print("Goal Node is %s"%goal_idx)
	print("--- %s seconds used for finding a plan---"%(time.time()-start_time))
	
	if goal_idx != -1:
		control_plan = find_plan(tree, goal_idx, True)
		print(control_plan)

		''' Plot teh Differential Robot State '''
		
		state = ego.state
		state[1] = state[1] - config.lr
		#plt.figure()
		plt.scatter(state[0],state[1],marker='x')
		for i in range(len(control_plan)):
			state = diff_kinematic(state,control_plan[i])
			plt.scatter(state[0],state[1],marker='x')
		plt.show()
		
	
