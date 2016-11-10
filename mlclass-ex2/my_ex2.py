# -*- coding: utf-8 -*-          
import numpy as np
import matplotlib.pyplot as plt
from BFGS import *

with open("ex2data1.txt","r") as f:
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
xlist=np.hstack([np.array([[1.0]*N]).T,rdata[0::,0:M-1]])  #add a column
ylist=rdata[0::,M-1]
# theta,Jlist,steplist=BFGS_method(xlist,ylist,M)
theta,Jlist,steplist=fmincg(xlist,ylist,M)
print theta,len(Jlist),Jlist
xpoint=np.array([min(xdata[0::,0]),max(xdata[0::,0])])
ypoint=-1./theta[2]*(theta[1]*xpoint + theta[0])
plt.figure()
plt.plot(xdata[0::,0],xdata[0::,1],'ro')
plt.plot(ydata[0::,0],ydata[0::,1],'bx')
plt.plot(xpoint,ypoint,'k--',lw=2)
plt.legend(("0","1","Boundary"),loc="best")
plt.plot()
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