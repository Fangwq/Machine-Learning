# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib
from scipy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D


data1 = scio.loadmat('ex7data1.mat')  
xdata1=data1['X'] 
xall=xdata1
print xdata1.shape


data2 = scio.loadmat('ex7faces.mat')   
xdata2=data2['X']
print xdata2.shape

mean=np.mean(xdata1,axis=0)
print mean
std=np.std(xdata1-mean,axis=0)
xdata1=(xdata1-mean)/std
M,N=xdata1.shape
evals, evecs = LA.eigh(np.dot(xdata1.T,xdata1)/M)
print evals,evecs
pointx=mean+1.5*evals[0]*evecs[:,0]
pointy=mean+1.5*evals[1]*evecs[:,1]
#Compute the projection of the data using only the top K 
#eigenvectors in evecs(first K columns). 

#==============deal with data4==============
data4=scio.loadmat('bird_small.mat')
xdata4=data4['A']
xdata4=1.0*xdata4/max(xdata4.reshape(-1,))
print xdata4.shape
a,b,c=np.shape(xdata4)
K=6
xdata4=xdata4.reshape(a*b,c)
M,N=xdata4.shape
indx=np.zeros(M)
# sent=np.zeros(M)
initial_center=np.random.randint(M,size=K)
# print xdata4[initial_center]
centroids=np.zeros((K,N))
center=xdata4[initial_center]
error=np.ones((K,N))
count=0
while error.all()>1.0e-7:
	count=count+1
	for i in xrange(M):
		xpoint=np.array([xdata4[i]]*K)     #turn into 3*2 array
		distance=np.sum((xpoint-center)**2,axis=1)
		# sent[i]=min(distance)
		indx[i]=int(np.argmin(distance))
	for i in xrange(K):
		centroids[i,:]=np.mean(xdata4[indx==i],axis=0)
	error=centroids-center
	center=centroids
	# print error
print 'the loop times:',count
# print indx
# replace the data with centroids and show the image
num=np.random.randint(M,size=1000)
point=xdata4[num]
index=indx[num]
colors = ['red','green','blue','purple','pink','black']
print np.unique(index)
#===================PCA of data4===============
dmean=np.mean(xdata4,axis=0)
std=np.std(xdata4-dmean,axis=0)
xdata4=(xdata4-dmean)/std
M,N=xdata4.shape
evals, evecs = LA.eigh(np.dot(xdata4.T,xdata4)/M)
print evals
print evecs
U=evecs[:,1:3]
Z=np.dot(xdata4,U)
Z=Z[num]
# print Z[:,0]
# print point[:,0]

plt.figure()
plt.plot(xall[:,0],xall[:,1],'ro')
plt.plot([mean[0],pointx[0]],[mean[1],pointx[1]],'g-',lw=2)
plt.plot([mean[0],pointy[0]],[mean[1],pointy[1]],'g-',lw=2)
plt.grid(True)
plt.figure()
plt.imshow(xdata2[0:100],clim=(-1, 1), aspect=1,cmap=plt.cm.gray)
plt.xticks([])   #get rid of ticks
plt.yticks([])
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point[:,0],point[:,1],point[:,2],c=index, cmap=matplotlib.colors.ListedColormap(colors))
fig=plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Z[:,0],Z[:,1],c=index, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()
