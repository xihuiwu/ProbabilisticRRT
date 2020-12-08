from treelib import Node, Tree
from car import Car
import numpy as np
import time

def sumsqr(vector1,vector2):
	value = (vector1[0]-vector2[0])**2 + (vector1[1]-vector2[1])**2
	return value

'''
This function is solely a collision check between two cars
'''
def collisionCheck(car1,car2):
	collision_flag = False
	for i in range(car1.centers.shape[0]):
		for j in range(car2.centers.shape[0]):
			dist2 = sumsqr(car1.centers[i,:],car2.centers[j,:])
			if dist2 <= (car1.radius+car2.radius)**2:
				collision_flag = True
				return collision_flag
	return collision_flag

'''
This function combines collision check and prune together
'''
def prune(tree,*cars):
	num_of_nodes = tree.size()
	for i in range(num_of_nodes):
		if tree.get_node(i) != None:
			ego = tree[i].data
			for x in range(ego.centers.shape[0]):
				for y in range(len(cars)):
					for z in range(cars[y].centers.shape[0]):
						dist2 = sumsqr(ego.centers[x,:],cars[y].centers[z,:])
						if dist2 <= (ego.radius+cars[y].radius)**2:
							tree.remove_node(i)
	return tree
			
if __name__ == '__main__':
	tree = Tree()
	tree.create_node("0",0,data=Car(np.array([0,0,0,0]),1,0,0.17,[]))
	tree.create_node("1",1,parent=0,data=Car(np.array([3,3,0,0]),1,0,0.17,[]))
	
	car1 = Car(np.array([3,3,0,0]),1,0,0.17,[])

	start =int(round(time.time()*1000))
	tree.show()
	tree = prune(tree,car1)
	tree.show()
	print("--- %s milliseconds ---"%(int(round(time.time()*1000))-start))
