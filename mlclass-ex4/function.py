# -*- coding: utf-8 -*-
import numpy as np

m1,n1=25,401
m2,n2=10,26
def g(z):
	return 1.0/(1.0+np.exp(-z))

def grad_g(z):
	return g(z)*(1.0-g(z))

def J(x,y,theta,para):
	sym=np.unique(y)-1
	# print sym
	temp_theta1=theta[0:m1*n1].reshape(m1,n1)
	temp_theta2=theta[m1*n1:].reshape(m2,n2)
	N=len(x)
	a1=x 	#(5000,400)
	a1add=np.hstack([np.array([[1.0]*len(x)]).T,x])	#(5000,401)
	z2=np.dot(a1add,temp_theta1.T)	#(5000,25)
	a2=g(z2)	#(5000,25)
	a2add=np.hstack([np.array([[1.0]*len(a2)]).T,a2])   #(5000,26)
	z3=np.dot(a2add,temp_theta2.T)   #(5000,10)
	a3=g(z3)	#(5000,10)
	sum_temp=[0.0]*len(sym)
	for i in sym:
		ytemp=one_vs_all(y,i+1)		#the label is from 0-9, but the value of y is 1-10
		# print np.log(a3[:,i]).shape	#(5000,)
		sum_temp[i]=sum(-ytemp*np.log(a3[:,i])-(1.0-ytemp)*np.log(1.0-a3[:,i]))
	return 1.0/N*sum(sum_temp)+ para/2.0/N*(sum(sum(temp_theta1[:,1:n1]**2))+sum(sum(temp_theta2[:,1:n2]**2)))

def one_vs_all(y,c):
	y_dict={True:1.0, False:0.0}
	y=(y==c)
	# print y
	y=map(lambda x: y_dict[x[0]],y)   #list type 
	# print y
	# print np.shape(np.array(y))	#(5000,)
	return np.array(y)

def gradient_J(x,y,theta,para):
	N=len(x)
	temp_theta1=theta[0:m1*n1].reshape(m1,n1)
	temp_theta2=theta[m1*n1:].reshape(m2,n2)
	grad_theta1=np.zeros((m1,n1))
	grad_theta2=np.zeros((m2,n2))
	eta=1.0   #learning rate
	sym=np.unique(y)-1
	ytemp=np.zeros((N,len(sym)))
	for i in sym:
		ytemp[:,i]=one_vs_all(y,i+1)   #(5000,10) y matrix
	#=====Forward propagation======
	# a1=x 	#(5000,400)
	# a1add=np.hstack([np.array([[1.0]*len(x)]).T,x])	#(5000,401)
	# z2=np.dot(a1add,temp_theta1.T)	#(5000,25)
	# a2=g(z2)	#(5000,25)
	# a2add=np.hstack([np.array([[1.0]*len(a2)]).T,a2])   #(5000,26)
	# z3=np.dot(a2add,temp_theta2.T)   #(5000,10)
	# a3=g(z3)	#(5000,10)
	#=======Backpropagation========
	G1=np.zeros((m1,n1))
	G2=np.zeros((m2,n2))
	#=======not with matrix product:it's very slow======
	# for i in xrange(N):
	# 	ra1=x[i:i+1,0:]  #x[i:i+1,0:]: get (1,400) from (5000,400)
	# 	ra1add=np.hstack([np.array([[1.0]*len(ra1)]).T,ra1])  #(1,401) add a column
	# 	rz2=np.dot(ra1add,temp_theta1.T)	#(1,25)
	# 	ra2=g(rz2)  #(1,25)
	# 	ra2add=np.hstack([np.array([[1.0]*len(ra2)]).T,ra2])  #(1,26)
	# 	rz3=np.dot(ra2add,temp_theta2.T)  #(1,10)
	# 	ra3=g(rz3)   #(1,10)
	# 	err3=ra3-ytemp[i,:]	#(1,10)
	# 	temp=np.dot(err3,temp_theta2)		#temp[0:,1:]: get (1,25) from (1,26)
	# 	err2=temp[0:,1:]*grad_g(rz2)    #(1,25)
	# 	G1=G1+eta*np.dot(err2.T,ra1add)
	# 	G2=G2+eta*np.dot(err3.T,ra2add)
	
	#=======with matrix product: it's very fast=========
	ra1=x
	ra1add=np.hstack([np.array([[1.0]*len(ra1)]).T,ra1])
	rz2=np.dot(ra1add,temp_theta1.T)
	ra2=g(rz2)  
	ra2add=np.hstack([np.array([[1.0]*len(ra2)]).T,ra2])  
	rz3=np.dot(ra2add,temp_theta2.T)  
	ra3=g(rz3)   
	err3=ra3-ytemp
	temp=np.dot(err3,temp_theta2)
	err2=temp[0:,1:]*grad_g(rz2)
	G1=G1+eta*np.dot(err2.T,ra1add)
	G2=G2+eta*np.dot(err3.T,ra2add)
	grad_theta1=G1/N+para*np.hstack([np.array([[0.0]*len(temp_theta1[0:,1:])]).T,temp_theta1[0:,1:]])/N
	grad_theta2=G2/N+para*np.hstack([np.array([[0.0]*len(temp_theta2[0:,1:])]).T,temp_theta2[0:,1:]])/N
	return np.hstack([grad_theta1.reshape(1,m1*n1),grad_theta2.reshape(1,m2*n2)])[0]

#predit the accuracy
def predict(x,y,theta):
	N=len(x)
	temp_theta1=theta[0:m1*n1].reshape(m1,n1)
	temp_theta2=theta[m1*n1:].reshape(m2,n2)
	a1=x 	#(5000,400)
	a1add=np.hstack([np.array([[1.0]*len(x)]).T,x])	#(5000,401)
	z2=np.dot(a1add,temp_theta1.T)	#(5000,25)
	a2=g(z2)	#(5000,25)
	a2add=np.hstack([np.array([[1.0]*len(a2)]).T,a2])   #(5000,26)
	z3=np.dot(a2add,temp_theta2.T)   #(5000,10)
	a3=g(z3)	#(5000,10)
	result=a3
	predict_y=np.argmax(result,axis=1)+1 #the large nuber index in each row
	# print predict_y.shape
	p=sum(predict_y==y[:,0])
	return 1.0*p/N

#optimization algorithm to find best weight
def fmincg(x,y,initial_theta,para):
	max_iter=100
	theta=initial_theta
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
	f1, df1 =J(x,y,theta,para),gradient_J(x,y,theta,para)
	# print df1.shape,'==='
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
	    f2, df2 =J(x,y,theta,para),gradient_J(x,y,theta,para)
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
	            f2, df2 =J(x,y,theta,para),gradient_J(x,y,theta,para)                
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
	    	f2, df2 =J(x,y,theta,para),gradient_J(x,y,theta,para)   
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
	return  theta,costJ[0,0:count1],stepsize[0,0:count1]
