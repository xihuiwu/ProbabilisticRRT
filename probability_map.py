import numpy as np
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import config

def gauss2D(xx,yy,mu,sigma,rho):
	z = np.zeros_like(xx)
	z = np.exp( (((xx-mu[0])/sigma[0])**2+((yy-mu[1])/sigma[1])**2-2*rho*np.multiply((xx-mu[0]),(yy-mu[1]))/sigma[0]/sigma[1])/(-2*(1-rho**2)) )/(2*math.pi*sigma[0]*sigma[1])
	maxValue = 1/(2*math.pi*sigma[0]*sigma[1])
	return z,maxValue


def generateMap(mu,sigma,lamb,plot=False):
	assert mu.shape[0] == sigma.shape[0], "Number of mu and sigma pairs are wrong"
	if mu.shape[0] == 1:
		pdf,maxValue = gauss2D(config.xx,config.yy,mu[0,:],sigma[0,:],0)
		pdfMap = pdf*lamb + 1/(config.lengthx*config.lengthy)
	else:
		pdfs = []
		maxs = []
		for i in range(mu.shape[0]):
			pdf,maxValue = gauss2D(config.xx,config.yy,mu[i,:],sigma[i,:],0)
			pdfs.append(pdf)
			maxs.append(maxValue)
		minValue = min(maxs[1:])
		scale = maxs/minValue
		pdfMap = pdfs[0]/scale[0]*lamb
		for i in range(1,mu.shape[0]):
			pdfMap -= pdfs[i]/scale[i]
		pdfMap += minValue
		
	pdfMap = np.true_divide(pdfMap,np.max(pdfMap)) #normalize
		
	if plot == True:
		fig = plt.figure()
		ax = fig.add_subplot(111,projection='3d')
		ax.plot_surface(config.xx,config.yy,pdfMap)
		plt.show()
	
	return pdfMap


def sample(pdf):
	row = np.sum(pdf,axis=1)
	col = np.sum(pdf,axis=0)
	x_cdf = np.cumsum(col)/np.sum(col) # normalize x position cdf
	y_cdf = np.cumsum(row)/np.sum(row) # normalize y position cdf

	prob = np.random.uniform(0,1,2)
	x_ind = np.digitize(prob[0],x_cdf)
	y_ind = np.digitize(prob[1],y_cdf)

	x = x_ind*config.dx
	y = y_ind*config.dy

	return np.array([x,y])


if __name__ == '__main__':
	plot_map = True
	plot_sample = True
	######################################
	#    Single Gaussian Distribution    #
	######################################
	'''
	mu = np.squeeze(config.mu)
	sigma = np.squeeze(config.sigma)
	dist,_ = gauss2D(config.xx,config.yy,mu,sigma,0)

	if plot_sample == True:
		plt.figure()
		start =int(round(time.time()*1000))
		for i in range(1000):
			(x,y) = sample(dist)
			plt.scatter(x,y)
		plt.show()
		print("--- %s milliseconds ---"%(int(round(time.time()*1000))-start))
	'''

	######################################
	#        With Involving Vehicle      #
	######################################
	mu = np.array([[0.8509,1.4478],[0.3,0.8809]])
	sigma = np.array([[0.01,0.01],[7.32e-1,3.33e-1]])
	lamb = 1e5
	pdfMap = generateMap(mu,sigma,lamb,True)
	
	if plot_sample == True:
		plt.figure()
		start =int(round(time.time()*1000))
		for i in range(1000):
			pt = sample(pdfMap)
			plt.scatter(pt[0],pt[1])
		print("--- %s milliseconds ---"%(int(round(time.time()*1000))-start))
		plt.show()
