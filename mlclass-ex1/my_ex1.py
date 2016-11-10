# -*- coding: utf-8 -*-          
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  
from mpl_toolkits.mplot3d import Axes3D  

# a = np.loadtxt('ex1data1.txt')  

def J(a,b):
	# global N
	result=1/2.0/N*(a*xdata+b-ydata)**2
	return sum(result)

def gradient_J(a,b):
	# global N
	alpha=0.01;
	a=a-alpha*1/N*sum((a*xdata+b-ydata)*xdata)
	b=b-alpha*1/N*sum(a*xdata+b-ydata)
	return a,b

with open('ex1data1.txt', 'r') as f:  
    data = f.readlines()  
    N=len(data)
    xdata=np.array([0.0 for i in xrange(N)])
    ydata=np.array([0.0 for i in xrange(N)])
    #define how many columns and its type
    # print len(data[0].split(',')),data[0].split(',')[0],type(data[0].split(',')[0])
    for i in xrange(N):
    	if ',' in data[i]:
    		temp= data[i].split(',')
    		# print temp
    		xdata[i]=float(temp[0])
    		ydata[i]=float(temp[1])


iter=2000;
Jlist=np.array([0.0 for i in xrange(iter)])
dJlist=np.array([0.0 for i in xrange(iter)])
theta0=0.0;theta1=0.0
for i in xrange(iter):
	Jlist[i]=J(theta0,theta1)
	theta0,theta1=gradient_J(theta0,theta1)
print 'The Linear parameter:%s,%s' % (theta0,theta1)
predict1 = np.dot(np.array([3.5,1]),np.array([theta0,theta1]))
predict2 = np.dot(np.array([7,1]),np.array([theta0,theta1]))
print predict1, predict2
plt.figure()
plt.plot(xdata,ydata,'ro')
plt.plot(xdata,theta0*xdata+theta1,'b-',linewidth=2)
plt.legend(("Training data","Linear fit"),loc='best')
plt.grid(True)
plt.xlim(min(xdata),max(xdata))
plt.ylim(min(ydata),max(ydata))
plt.figure()
plt.plot(xrange(iter),Jlist,'g-',linewidth=2)
plt.legend(("costJ",''),loc='best')
plt.grid(True)

m=100;n=100
theta_value0=np.linspace(-1,4,m)
theta_value1=np.linspace(-10,10,n)
X, Y = np.meshgrid(theta_value0, theta_value1)
Z=np.array([[0.0] *m for i in xrange(n)])
for i in xrange (m):
	for k in xrange (n):
		Z[i][k]=J(X[i][k],Y[i][k])

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.figure()
Zmin=min(Z.flatten())
Zmax=max(Z.flatten())
cs= plt.contour(X,Y,Z,np.linspace(0,100,10))
plt.plot(theta0,theta1,'xr')
plt.show()

# print type(data[0]),data[0]