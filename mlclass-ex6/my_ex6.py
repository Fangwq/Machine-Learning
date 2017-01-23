# -*- coding: utf-8 -*-
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import random
from function import *

data1 = scio.loadmat('ex6data1.mat')  
xdata1=data1['X'] 
ydata1=data1['y'][:,0]
# print data1
print type(xdata1),type(ydata1)
print xdata1.shape
# print ydata1.shape,xdata1.shape

data2 = scio.loadmat('ex6data2.mat')   
xdata2=data2['X']
ydata2=data2['y'][:,0]
print xdata2.shape

data3 = scio.loadmat('ex6data3.mat')   
xdata3=data3['X']
ydata3=data3['y'][:,0]
Xval=data3['Xval']
yval=data3['yval'][:,0]
print xdata3.shape	
xtemp=xdata3
ytemp=ydata3
#=====draw the point with different classes=====
xx1=xdata1[ydata1==0]
yy1=xdata1[ydata1==1]
xx2=xdata2[ydata2==0]
yy2=xdata2[ydata2==1]
xx3=xdata3[ydata3==0]
yy3=xdata3[ydata3==1]

#=================turn y into -1 when y=0===============
ydata1=np.array(map(lambda x: 2*x-1,ydata1))
ydata2=np.array(map(lambda x: 2*x-1,ydata2))
ydata3=np.array(map(lambda x: 2*x-1,ydata3))

#================calculate the kernel function==================
# r1,c1=np.shape(xdata1)
# r2,c2=np.shape(xdata2)
# r3,c3=np.shape(xdata3)
# # print r2,c2
# linearK=np.zeros((r1,r1))
# gaussK1=np.zeros((r2,r2))
# gaussK2=np.zeros((r3,r3))
# for i in xrange(r1):
# 	for j in xrange(r1):
# 		linearK[i,j]=linearkernel(xdata1[i,:],xdata1[j,:])
# 		linearK[j,i]=linearK[i,j]
# for i in xrange(r2):
# 	for j in xrange(r2):
# 		gaussK1[i,j]=gausskernel(xdata2[i,:],xdata2[j,:])
# 		gaussK1[j,i]=gaussK1[i,j]
# for i in xrange(r3):
# 	for j in xrange(r3):
# 		gaussK2[i,j]=gausskernel(xdata3[i,:],xdata3[j,:])
# 		gaussK2[j,i]=gaussK2[i,j]
# print np.diag(linearK),np.diag(gaussK1)

#===============use the Simplified SMO algorithm=================
#====wx+b=0 edition=====
C1=100
alpha1,b1=SimplifiedSMO(xdata1,ydata1,C1,kernel='linearkernel')
print alpha1.shape,b1
print alpha1[alpha1>0],len(alpha1[alpha1>0])
print alpha1
linearW=np.dot(ydata1*alpha1,xdata1)
print linearW
wx1=min(xdata1[:,0]);wx2=max(xdata1[:,1])
wy=(-1.0/linearW[1])*(linearW[0]*np.array([wx1,wx2]) + b1)

# C2=10
# alpha2,b2=SimplifiedSMO(xdata2,ydata2,C2,kernel='linearkernel')

C3=0.6
alpha3,b3=SimplifiedSMO(xdata3,ydata3,C3,kernel='gausskernel')
print alpha3.shape,b3
print alpha3[alpha3>0],len(alpha3[alpha3>0])
#the alpha value may be smaller than zero,it need to be deleted
m=300;n=300
xmin=np.linspace(min(xdata3[:,0]),max(xdata3[:,0]),m)
ymin=np.linspace(min(xdata3[:,1]),max(xdata3[:,1]),n)
X, Y = np.meshgrid(xmin, ymin)
# xdata3=xdata3[(alpha3>0) & (alpha3<C2)]
# ydata3=ydata3[(alpha3>0) & (alpha3<C2)]
# alpha3=alpha3[(alpha3>0) & (alpha3<C2)]
xdata3=xdata3[alpha3>0]
ydata3=ydata3[alpha3>0]
alpha3=alpha3[alpha3>0]
M,=np.shape(alpha3)
Z=np.zeros([m,n])
for i in xrange(m):
	temp=np.array([X[:,i],Y[:,i]]).T    #it can be chosen [X[:,i],Y[:,0]]
	for j in xrange(n):
		for k in xrange(M):
			Z[i,j]=Z[i,j]+alpha3[k]*ydata3[k]*gausskernel(temp[j],xdata3[k])
Z=Z+b3

=====check the accuracy=====
value=np.zeros(len(xtemp))
for i in xrange(len(value)):
	for j in xrange(M):
		value[i]=value[i]+alpha3[j]*ydata3[j]*gausskernel(xtemp[i],xdata3[j])
value=value+b3
for i in xrange(len(value)):
	if value[i]>0:
		value[i]=1
	else:
		value[i]=0
accuracy=1.0*len(value[value==ytemp])/len(value)
print accuracy

#===============plot the figure===============
#=====figure one=====
plt.figure()
plt.plot(xx1[:,0],xx1[:,1],'ro')
plt.plot(yy1[:,0],yy1[:,1],'bx')
plt.plot(np.array([wx1,wx2]),wy,'g-',lw=2)
plt.grid(True)

#====figure two====
plt.figure()
plt.plot(xx2[:,0],xx2[:,1],'ro')
plt.plot(yy2[:,0],yy2[:,1],'bx')
plt.grid(True)

#====figure three====
plt.figure()
plt.plot(xx3[:,0],xx3[:,1],'ro')
plt.plot(yy3[:,0],yy3[:,1],'bx')
cs=plt.contour(X,Y,Z.T,[0.0],colors='k', linewidths=2.0, linestyles='--')  
plt.clabel(cs, fontsize=10,fmt='%0.1f')
cs.collections[0].set_label("Boundary")   #for contour plot legend
# plt.legend(("0","1","Boundary"),loc="best")
plt.grid(True)
plt.show()
