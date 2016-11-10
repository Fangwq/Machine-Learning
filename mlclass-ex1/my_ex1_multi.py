# -*- coding: utf-8 -*-          
import numpy as np
import matplotlib.pyplot as plt

with open('ex1data2.txt','r') as f:
	data=f.readlines()
	N=len(data)
	M=len(data[0].split(','))
	# print N,M
	rdata=np.array([[0.0]*M for i in xrange(N)])
	for i in xrange(N):
		if ',' in data[i]:
			temp=data[i].split(',')
			for j in xrange(M):
				rdata[i][j]=float(temp[j])
tempdata=rdata
tempdata=np.hstack([np.array([[1.0]*N]).T,tempdata])  #add a column
std=np.array([0.0 for i in xrange(M)])
for i in xrange(M-1):
	rdata[0::,i]=rdata[0::,i]-rdata[0::,i].mean()
	std[i]=np.std(rdata[0::,i])
	rdata[0::,i]=rdata[0::,i]/std[i]

xdata=np.hstack([np.array([[1.0]*N]).T,rdata[0::,0:M-1]])  #add a column
ydata=rdata[0::,M-1]
def J(theta):
	return 1.0/2/N*sum((np.dot(xdata,theta)-ydata)**2)

def gradient_J(theta):
	return 1/2.0/N*np.dot(xdata.T,np.dot(xdata,theta)-ydata)+ \
     		1/2.0/N*np.dot((np.dot(xdata,theta)-ydata).T,xdata)

iter=100
theta=np.array([1.0,1.0,1.0])
Jlist=np.array([0.0 for i in xrange(iter)])
for  i in xrange(iter):
	alpha=0.03
	Jlist[i]=J(theta)
	theta=theta-alpha*gradient_J(theta)
#using exact formula
theta_exact=np.dot(np.mat(np.dot(tempdata[0::,0:M].T, \
			tempdata[0::,0:M])).I,np.dot(tempdata[0::,0:M].T,tempdata[0::,M]))
print theta,theta_exact
plt.figure()
plt.plot(xrange(iter),Jlist,'r-',linewidth=2)
plt.legend(("costJ",""),loc='best')
plt.xlabel("iteration number")
plt.ylabel("cost J")
plt.show()


