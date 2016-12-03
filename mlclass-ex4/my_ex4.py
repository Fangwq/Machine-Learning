# -*- coding: utf-8 -*-
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from function import *
import random

data1 = scio.loadmat("ex4data1.mat")   #the format is dictionary
xdata=data1['X']	#(5000,400)
ydata=data1['y']    #(5000,1)
# print type(ydata),type(xdata),ydata.shape
# print ydata
data2 = scio.loadmat("ex4weights.mat")   #the format is dictionary
theta1=data2['Theta1']
theta2=data2['Theta2']
m1,n1=theta1.shape    #(25,401)
m2,n2=theta2.shape	  #(10,26)
M,N=xdata.shape      #(5000,401)
# print m1,n1,m2,n2,M,N
# ===============================#check if the cost function is coded right================
a=theta1.reshape(1,m1*n1)	#(1,10025)
b=theta2.reshape(1,m2*n2)	#(1,260)
test_theta=np.hstack([a,b])[0] #it is an array with shape(10028,)
# print np.hstack([theta1.reshape(1,m1*n1),theta2.reshape(1,m2*n2)]) #mearge two array
alpha=1.0   #the parameter lambda
print J(xdata,ydata,test_theta,alpha)  
#=======================initialize the theta weight between the layers=====================
#with random weight, the output is different at each time.
#The learning rate maybe very important, and sometimes this code 
#doesn't get the right answer. The random initial maybe the problem.
rand=0.12
initial_theta1=np.random.rand(m1,n1)*2.0*rand-rand
initial_theta2=np.random.rand(m2,n2)*2.0*rand-rand
print initial_theta1.shape,initial_theta2.shape
array_initial_theta1=initial_theta1.reshape(1,m1*n1)
array_initial_theta2=initial_theta2.reshape(1,m2*n2)
Theta_all=np.hstack([array_initial_theta1,array_initial_theta2])[0]
para=3.0
weight,Jlist,steplist=fmincg(xdata,ydata,Theta_all,para)
print 'the prediction accuracy is:%s' % predict(xdata,ydata,weight)
plt.figure()
plt.plot(xrange(len(Jlist)),Jlist,'ro')
plt.legend(("costJ",""),loc="best")
plt.grid(True)
plt.figure()
plt.plot(xrange(len(steplist)),steplist,'cd')
plt.legend(("stepsize",""),loc="best")
plt.grid(True)
plt.show()
#=======================

