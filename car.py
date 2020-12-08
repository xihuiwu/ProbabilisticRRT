import numpy as np
import math

# Number of circle centers are odd numbers, i.e., 1 or 3
# state is 4x1 vector
# centers is nx2 matrix
class Car():
	def __init__(self,state,n,radius,dist,u_prev):
		if state.shape[0] == 1:
			self.state = np.squeeze(np.transpose(state))
		else:
			self.state = np.squeeze(state)
		self.radius = radius
		self.dist = dist
		self.centers = np.empty([n,2])
		self.calCenter()
		self.u_prev = u_prev

	def update(self,state,u):
		self.state = np.squeeze(state)
		self.u_prev = u
		self.calCenter()

	def calCenter(self):
		position = np.array([self.state[0],self.state[1]])
		heading = self.state[2]
		displacement = np.array([self.dist,0])
		if self.centers.shape[0] == 1:
			self.centers = np.array([[self.state[0],self.state[1]]])
		else:
			R = np.array([[math.cos(heading),-math.sin(heading)],[math.sin(heading),math.cos(heading)]])

			self.centers[0,:] = np.matmul(R,np.transpose(position-displacement))
			self.centers[1,:] = position
			self.centers[2,:] = np.matmul(R,np.transpose(position+displacement))

if __name__ == '__main__':
	init_state = np.array([0,0,math.pi/2,0])
	ego = Car(init_state,3,1,0.7)

