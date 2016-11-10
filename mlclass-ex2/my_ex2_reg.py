# -*- coding: utf-8 -*-          
import numpy as np
import matplotlib.pyplot as plt
from BFGS_reg import *

with open("ex2data2.txt","r") as f:
    data = f.readlines()  
    N=len(data)
    M=len(data[0].split(','))
    rdata=np.array([[0.0]*M for i in xrange(N)])
    for i in xrange(N):
		if ',' in data[i]:
			temp=data[i].split(',')
			for j in xrange(M):
				rdata[i][j]=float(temp[j])
xdata=rdata[rdata[0::,M-1]==0.0,0:M-1]
ydata=rdata[rdata[0::,M-1]==1.0,0:M-1]
#============================================
order=6
x1array=rdata[0::,0]
x2array=rdata[0::,1]
xlist=map_feature(x1array,x2array,order,N)
# print xlist[0,0::],xlist.shape
ylist=rdata[0::,M-1]
dimension=28   #column
var=1.0
theta,Jlist,steplist=BFGS_method(xlist,ylist,dimension,var)
# theta,Jlist,steplist=fmincg(xlist,ylist,dimension,var)
print theta,Jlist,predict(xlist,ylist,theta)
m=300;n=300
xmin=np.linspace(-1.0,1.5,m)
ymin=np.linspace(-1.0,1.5,n)
X, Y = np.meshgrid(xmin, ymin)
Z=np.zeros([m,n])
for i in xrange(m):
	for j in xrange(n):
		Z[i,j]=np.dot(map_feature(xmin[i],ymin[j],order,1),theta)
plt.figure()
plt.plot(xdata[0::,0],xdata[0::,1],'ro')
plt.plot(ydata[0::,0],ydata[0::,1],'bx')
cs=plt.contour(X,Y,Z,[0.0],colors='k', linewidths=2.0, linestyles='--')  
plt.clabel(cs, fontsize=10,fmt='%0.1f')
cs.collections[0]   #for contour plot legend
plt.legend(("0","1","Boundary"),loc="best")
plt.grid(True)
plt.figure()
plt.plot(xrange(len(Jlist)),Jlist,'ro')
plt.legend(("costJ",""),loc="best")
plt.grid(True)
plt.figure()
plt.plot(xrange(len(steplist)),steplist,'cd')
plt.legend(("stepsize",""),loc="best")
plt.grid(True)
plt.show()