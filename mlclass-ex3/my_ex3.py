# -*- coding: utf-8 -*-
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from fmincg import *

data = scio.loadmat("ex3data1.mat")   #the format is dictionary
xdata=data['X']
ydata=data['y']
# print ydata[0::,0],type(ydata[0::,0])
M,N=xdata.shape
# iterm=np.random.randint(0,M-1,100)  #get random 100 rows
iterm=np.random.permutation(M)[0:100]
rdata=xdata[iterm,0::]
m,n=rdata.shape
example_width = int(np.round(np.sqrt(n)))
example_height = int(n/ example_width)
display_rows = int(np.floor(np.sqrt(m)))
display_cols = int(np.ceil(1.0*m/ display_rows))
display_array = - np.ones([1+display_rows * (example_height + 1),1+display_cols * (example_width + 1)])
# print display_array.shape
# draw the image
col=0
for i in xrange(display_rows):
	for j in xrange(display_cols):
		if col>m:
			break
		max_val = max(abs(rdata[col, :]))
		# if j==9 and i==9:
			# print display_array[(j - 1)*(example_height+1):((j - 1)*(example_height + 1)+example_height), \
		    	          # (i - 1)*(example_width+1):((i - 1)*(example_width+ 1)+example_width)].shape
		display_array[(j - 1)*(example_height+1):((j - 1)*(example_height + 1)+example_height), \
		              (i - 1)*(example_width+1):((i - 1)*(example_width+ 1)+example_width)]= \
						rdata[col, :].reshape(example_height, example_width)/max_val
		col=col+1
	if col>m:
		break
#================================================================================
label=np.unique(ydata)
var=0.1
Xdata=np.hstack([np.array([[1.0]*M]).T,xdata])   #add a column
Ydata=ydata[0::,0]
theta_list=np.array([[0.0]*(N+1)])  #store the theta in each step
for i in label:
	theta,Jlist,steplist=fmincg(Xdata,Ydata,i,N+1,var)
	theta_list=np.vstack([theta_list,theta])   #collect all the theta
print theta_list[1::,0:10],Jlist,predict(Xdata,Ydata,theta_list[1::,:])
plt.figure()
plt.imshow(display_array.T,clim=(-1, 1), aspect=1,cmap=plt.cm.gray)
plt.figure()
plt.plot(xrange(len(Jlist)),Jlist,'ro')
plt.legend(("costJ",""),loc="best")
plt.grid(True)
plt.figure()
plt.plot(xrange(len(steplist)),steplist,'cd')
plt.legend(("stepsize",""),loc="best")
plt.grid(True)
plt.show()


