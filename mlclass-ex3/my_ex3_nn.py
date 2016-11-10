# -*- coding: utf-8 -*-
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from fmincg import *

data1=scio.loadmat("ex3data1.mat")
xdata=data1['X']
ydata=data1['y'][0::,0]
# print ydata
data2 = scio.loadmat("ex3weights.mat")   #the format is dictionary
theta1=data2['Theta1']
theta2=data2['Theta2']
# print xdata.shape,ydata.shape,theta1.shape,theta2.shape
M,N=xdata.shape
Xdata=np.hstack([np.array([[1.0]*M]).T,xdata])  #add column
def g(z):
	return 1.0/(1.0+np.exp(-z))

def predict(Theta1,Theta2,X):
	a1=X
	z2=np.dot(Theta1,X.T)    
	a2=g(z2)
	a2add=np.vstack([np.array([[1.0]*len(a2.T)]),a2])   #add a row
	z3=np.dot(Theta2,a2add)
	p=np.argmax(z3,axis=0)+1  #column maximum index
	return p

r=predict(theta1,theta2,Xdata)
num=(r==ydata)
print 1.0*sum(num)/M