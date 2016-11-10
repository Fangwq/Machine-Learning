# -*- coding: utf-8 -*-
import numpy as np

def g(z):
	return 1.0/(1.0+np.exp(-z))

#cost function
def J(x,y,c,theta,para):
	y=define(y,c)
	N=len(x)
	z0=np.dot(theta,x.T)
	y=y.reshape(1,N)
	return 1.0/N*sum((-y*np.log(g(z0))-(1.0-y)*np.log(1.0-g(z0)))[0])+para/2.0/N*sum(theta[1:]**2)

# use dictionary to get one vs all y
def define(y,c):
	label=np.unique(y)
	y_dict={True:1.0, False:0.0}
	y=(y==c)
	y=map(lambda x: y_dict[x],y)   #list type
	return np.array(y)

#predict the accuracy of our model
def predict(x,y,theta):
	N=len(x)
	z0=np.dot(theta,x.T)
	result=g(z0).T
	# predict_y=np.amax(result, axis=1)  #the largest number in each row
	predict_y=np.argmax(result,axis=1)+1 #the 
	# print predict_y
	p=sum(predict_y==y)
	return 1.0*p/N

#gradient of costfunction
def gradient_J(x,y,c,theta,para):
	y=define(y,c)
	N=len(x)
	y=y.reshape(1,N)
	z0=np.dot(theta,x.T)
	temp_theta=np.hstack([0.0,theta[1:]])   #add 0.0
	return 1.0/N*(np.dot(g(z0)-y,x)[0]+para*temp_theta)


#====fmincg function written by Carl Edward Rasmussen====
#The nan value encounter in the Jlist is really annoying
#when the initial value of theta is improper since overflow 
#will happen.
def fmincg(x,y,c,dim,para):
	max_iter=10000
	theta=np.array([0.0 for i in xrange(dim)])
	costJ=np.zeros([1,max_iter])
	stepsize=np.zeros([1,max_iter])
	count1=0
	rho = 0.01                           
	sig = 0.5       
	init = 0.1    
	ext = 3.0                    
	largest = 20                         
	ratio = 100.0                                      
	i = 0                                           
	ls_failed = 0                         
	f1, df1 =J(x,y,c,theta,para),gradient_J(x,y,c,theta,para)
	# print f1   
	s = -df1                                      
	d1 = -np.dot(s,s)
	z1 = 1.0/(1.0-d1)
	z2=0  #It doesn't do anything, only for the stepsize in the below                         
	while i < max_iter:
	    i = i + 1
	    X0 = theta 
	    f0 = f1 
	    df0 = df1
	    theta = theta + z1*s
	    f2, df2 =J(x,y,c,theta,para),gradient_J(x,y,c,theta,para)
	    # print f2
	    d2 = np.dot(df2,s)
	    f3 = f1 
	    d3 = d1 
	    z3 = -z1
	    M = largest
	    success = 0 
	    limit = -1
	    while True:
	        while ((f2 > f1+z1*rho*d1) or (d2 > -sig*d1)) and (M > 0):
	            limit = z1                                         
	            if f2 > f1:
	                z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3)                
	            else:
	                A = 6.0*(f2-f3)/z3+3.0*(d2+d3)                                
	                B = 3.0*(f3-f2)-z3*(d3+2.0*d2)
	                z2 = (np.sqrt(B*B-A*d2*z3*z3)-B)/A
	            if np.isnan(z2) or np.isinf(z2):
	                z2 = z3/2.0                  
	            z2 = max(min(z2, init*z3),(1.0-init)*z3)
	            z1 = z1 + z2                                         
	            theta = theta + z2*s
	            f2, df2 =J(x,y,c,theta,para),gradient_J(x,y,c,theta,para)                
	            M = M - 1 
	            d2 = np.dot(df2,s)
	            z3 = z3-z2
	        if (f2 > f1+z1*rho*d1) or (d2 > -sig*d1):
	            break                                      
	        elif d2 > sig*d1:
	            success = 1 
	            break                                          
	        elif M == 0:
	            break                                                         
	        A = 6.0*(f2-f3)/z3+3.0*(d2+d3)                  
	        B = 3.0*(f3-f2)-z3*(d3+2.0*d2)
	        z2 = -d2*z3*z3/(B+np.sqrt(B*B-A*d2*z3*z3))       
	        if (not np.isreal(z2)) or np.isnan(z2) or np.isinf(z2) or z2 < 0:
	            if limit < -0.5:                       
	                z2 = z1 * (ext-1) 
	            else:
	                z2 = (limit-z1)/2.0                               
	        elif (limit > -0.5) and (z2+z1 > limit):
	            z2 = (limit-z1)/2.0                                
	        elif (limit < -0.5) and (z2+z1 > z1*ext):
	            z2 = z1*(ext-1.0)                         
	        elif z2 < -z3*init:
	            z2 = -z3*init
	        elif (limit > -0.5) and (z2 < (limit-z1)*(1.0-init)):
	            z2 = (limit-z1)*(1.0-init)
	        f3 = f2; d3 = d2; z3 = -z2                 
	        z1 = z1 + z2 
	        theta = theta + z2*s                      
	    	f2, df2 =J(x,y,c,theta,para),gradient_J(x,y,c,theta,para)   
	    	# print f2             
	        M = M - 1 
	        d2 = np.dot(df2,s)
	    if success:
	        count1=count1+1
	        f1 = f2
	        # print theta 
	        costJ[0,count1-1]=f1
	        stepsize[0,count1-1]=z2  
	        #when the initial theta is 0, the z2 value doesn't appear in initial loop,
	        #so I add "z2=0" in the initial setting
	        s = (np.dot(df2,df2)-np.dot(df1,df2))/np.dot(df1,df1)*s - df2      
	        tmp = df1; df1 = df2; df2 = tmp     
	        d2 = np.dot(df1,s)
	        if d2 > 0:                                   
	            s = -df1                              
	            d2 = -np.dot(s,s)   
	        z1 = z1 * min(ratio, d1/d2)       
	        d1 = d2
	        ls_failed = 0                           
	    else:
	        theta = X0 
	        f1 = f0 
	        df1 = df0  
	        if ls_failed or i > max_iter:      
	            break                             
	        tmp = df1 
	        df1 = df2 
	        df2 = tmp
	        s = -df1                                                   
	        d1 = -np.dot(s,s)
	        z1 = 1.0/(1.0-d1)                     
	        ls_failed = 1
	# print count1
	return  theta,costJ[0,0:count1],stepsize[0,0:count1]
	
