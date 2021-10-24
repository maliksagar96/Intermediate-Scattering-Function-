#In this code or any other relevant code for that matter we are not working in simulation units but 
#real units. We are converting pixels to micrometer and doing all the calculations for that.

import time
start=time.time()
import numpy as np
import pandas as pd
from pandas import DataFrame
import trackpy as tp
import matplotlib.pyplot as plt
import warnings
from warnings import warn
import multiprocessing

warnings.filterwarnings('ignore')

start = time.time()
mpp = 145 / 512 		 # Microns per pixel, conversion factor
fps = 21 				 # Frame per second
conc = 0.55
k = 3.17
minus_one = -1

j = np.sqrt(np.complex(minus_one))

fname = '/home/sagar/Documents/codes/python/s4/fiveframes.npy'

inputdata = np.load(fname)   

frames = int(inputdata[:,2][len(inputdata[:,2])-1] - inputdata[:,2][0] + 1)

traj=DataFrame(inputdata,columns=['x','y','frame','particle'])

calcDrift = tp.compute_drift(traj)
driftRemoved = tp.subtract_drift(traj,calcDrift)

data = driftRemoved.to_numpy()

data[:, 0] *= mpp                # converting pixel values to micrometers
data[:, 1] *= mpp

b = 1

#Step function which just spits out 0 or 1 depending on whether the input particle has moved greater then some distance or not
#Overlap Function
def overlapFn(b, x_t1,x_t2,y_t1,y_t2):
	overlap = 0.3 * b - (((x_t2 - x_t1)**2+(y_t2 - y_t1)**2)**0.5)
	heaviside_overlap=np.where(overlap<0,0,1)
	return heaviside_overlap

#Fourier transform of overlap function
#q = Fourier variable
def FT(q,x_t1,x_t2,y_t1,y_t2):
	j = np.sqrt(np.complex(minus_one))
	return 	np.sum(np.exp(-j*(q*(x_t2 + y_t2 - x_t1 - y_t1))))

def calcISF(i, f0, isf):
	
	for ti in range(i, i+1):

		ovrlapsum = 0 				#Number of slow particles
		ft = 0 						#Fourier transform of overlap function  
		loopCounter = 0				#For time averaging
		for tj in range(int(inputdata[:,2][0]), frames):
			if(ti + tj < frames):
				ovrlap = overlapFn(b,x[:,tj],x[:,ti+tj],y[:,tj],y[:,ti+tj])
				ovrlapsum = ovrlapsum + sum(ovrlap) 
				ft = ft + np.real(FT(k, x[:,tj],x[:, ti+tj], y[:,tj], y[:, ti+tj]))
				loopCounter = loopCounter + 1
		if(loopCounter == 0):
			break
		isf[ti] = ft/(N * loopCounter) 
		f0[ti] = ovrlapsum/(N * loopCounter)


if __name__ == "__main__":
	        
	N = int(np.shape(np.where(data[:, 2] == inputdata[:,2][0]))[1])		#Total number of particles in frame 0
	print("Number of particles = ", N)
	
	x = np.zeros((N, frames), dtype = float)
	y = np.zeros((N, frames), dtype = float)

	for t in range(int(inputdata[:,2][0]), frames):
		frameIndex = np.where(data[:,2] == t)         # Index of the ith(or tth if you like) frame 
		
		datafrm = data[frameIndex]                        # Data of that frame
		datafrm = datafrm[datafrm[:, 3].argsort()]	
		x[:, t] = datafrm[:, 0]
		y[:, t] = datafrm[:, 1]

	f0 = multiprocessing.Array('d', frames)
	isf = multiprocessing.Array('d', frames)	
	
	processes = []

	for i in range(frames):	
		p = multiprocessing.Process(target = calcISF, args = (i, f0, isf))
		p.start()
		processes.append(p)
		
	for process in processes:
		process.join() 
	
	np.save('ISF'+str(conc)+str(k), isf)
	np.save('f0'+str(conc)+str(k), f0)