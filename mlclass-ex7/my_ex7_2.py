# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

data4=scio.loadmat('bird_small.mat')
xdata4=data4['A']
xall=xdata4
xdata4=1.0*xdata4/max(xdata4.reshape(-1,))
print xdata4.shape
a,b,c=np.shape(xdata4)
K=50
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
		indx[i]=np.argmin(distance)
	for i in xrange(K):
		centroids[i,:]=np.mean(xdata4[indx==i],axis=0)
	error=centroids-center
	center=centroids
	# print error
print 'the loop times:',count
# print indx
# replace the data with centroids and show the image
for i in xrange(M):
	xdata4[i,:]=centroids[int(indx[i]),:]
xdata4=xdata4.reshape(a,b,c)
plt.figure()
plt.imshow(xdata4,clim=(-1, 1), aspect=1,cmap=plt.cm.gray)
plt.xticks([])   #get rid of ticks
plt.yticks([])
plt.figure()
plt.imshow(xall,clim=(-1, 1), aspect=1,cmap=plt.cm.gray)
plt.xticks([])   #get rid of ticks
plt.yticks([])
plt.show()
