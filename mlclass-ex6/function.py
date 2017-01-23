# -*- coding: utf-8 -*-
import numpy as np
import random
from numba import jit

tol=1.0e-3
esp=1.0e-3

@jit
def linearkernel(class1,class2):
	return np.dot(class1,class2)

@jit
def gausskernel(class1,class2):
	sigma=0.1
	return np.exp(-sum((class1-class2)**2)/2./sigma**2)

#==================the code for platt's paper==============
# @jit
# def takeStep(X,Y,C,i1,i2,kernel='linearkernel'):
# 	if i1==i2:
# 		return 0
# 	else:
# 		M,N=np.shape(X)
# 		# alpha=2*np.random.rand(M)-1
# 		alpha1=alpha[i1]
# 		alpha2=alpha[i2]
# 		y1=Y[i1]
# 		y2=Y[i2]
# 		temp_kernel1=np.zeros(M)
# 		temp_kernel2=np.zeros(M)
# 		b=0.0
# 		if kernel=='linearkernel':
# 			for i in xrange(M):
# 				temp_kernel1[i]=linearkernel(X[i],X[i1])
# 				temp_kernel2[i]=linearkernel(X[i],X[i2])
# 		elif kernel=='gausskernel':
# 			for i in xrange(M):
# 				temp_kernel1[i]=gausskernel(X[i],X[i1])
# 				temp_kernel2[i]=gausskernel(X[i],X[i2])
# 		E1=sum(alpha*Y*temp_kernel1)-b-y1
# 		E2=sum(alpha*Y*temp_kernel2)-b-y2
# 		s=y1*y2
# 		if s==-1.0:
# 			L=max(0.0,alpha2-alpha1)
# 			H=min(C,C+alpha2-alpha1)
# 		else:
# 			L=max(0.0,alpha2+alpha1-C)
# 			H=min(C,alpha2+alpha1)
# 		if L==H:
# 			return 0
# 		if kernel=='linearkernel':
# 			K11=linearkernel(X[i1],X[i1])
# 			K12=linearkernel(X[i1],X[i2])
# 			K22=linearkernel(X[i2],X[i2])
# 		elif kernel=='gausskernel':
# 			K11=gausskernel(X[i1],X[i1])
# 			K12=gausskernel(X[i1],X[i2])
# 			K22=gausskernel(X[i2],X[i2])
# 		eta=2.0*K12-K11-K22
# 		if eta<0.0:
# 			a2=alpha2+y2*(E2-E1)/eta
# 			if a2<L:
# 				a2=L
# 			elif a2>H:
# 				a2=H
# 		else:
# 			# f1=y1*(E1+b)-alpha1*K11-s*alpha2*K12
# 			# f2=y2*(E2+b)-s*alpha1*K12-alpha2*K22
# 			# L1=alpha1+s*(alpha2-L)
# 			# H1=alpha1+s*(alpha2-H)
# 			# Lobj=L1*f1+L*f2+1/2.0*L1**2*K11+1/2.0*L**2*K22+s*L*L1*K12
# 			# Hobj=H1*f1+H*f2+1/2.0*H1**2*K11+1/2.0*H**2*K22+s*H*H1*K12
# 			c1=eta/2.0
# 			c2=y2*(E1-E2)-eta*alpha2
# 			Lobj=c1*L**2+c2*L
# 			Hobj=c1*H**2+c2*H
# 			if Lobj>Hobj+eps:
# 				a2=L
# 			elif Lobj<Hobj-eps:
# 				a2=H
# 			else:
# 				a2=alpha2
# 		if abs(a2-alpha2)<eps*(a2+alpha2+eps):
# 			return 0
# 		a1=alpha1+s*(alpha2-a2)
# 		if a1<0:
# 			a2=a2+s*a1
# 			a1=0.0
# 		elif a1>C:
# 			a2=a2+s*(a1-C)
# 		if a1>0.0 and a1<C:
# 			bnew=b+E1+y1*(a1-alpha1)*K11+y2*(a2-alpha2)*K12
# 		else:
# 			if a2>0.0 and a2<C:
# 				bnew=b+E2+y1*(a1-alpha1)*K12+y2*(a2-alpha2)*K22
# 			else:
# 				b1=b+E1+y1*(a1-alpha1)*K11+y2*(a2-alpha2)*K12
# 				b2=b+E2+y1*(a1-alpha1)*K12+y2*(a2-alpha2)*K22
# 				bnew=(b1+b2)/2.0
# 		detal_b=bnew-b
# 		b=bnew
# 		# if kernel=='linearkernel':
# 			# new_w=w+y1*(a1-alpha1)*X[i1]+y2*(a2-alpha2)*X[i2]    #just for linear case
# 		t1=y1*(a1-alpha1)
# 		t2=y2*(a2-alpha2)
# 		if kernel=='linearkernel':
# 			for i in xrange(M):
# 				if 0<alpha[i]<C:
# 					error[i]=error[i]+t1*linearkernel(X[i1],X[i])+t2*linearkernel(X[i2],X[i])-detal_b
# 		elif kernel=='gausskernel':
# 			for i in xrange(M):
# 				if 0<alpha[i]<C:
# 					error[i]=error[i]+t1*gausskernel(X[i1],X[i])+t2*gausskernel(X[i2],X[i])-detal_b
# 		error[i1]=0.0
# 		error[i2]=0.0
# 		alpha[i1]=a1
# 		alpha[i2]=a2
# 		return 1

# @jit
# def examineExample(X,Y,C,i1,kernel='linearkernel'):
# 	M,N=np.shape(X)
# 	y1=Y[i1]
# 	alpha1=alpha[i1]
# 	temp_kernel1=np.zeros(M)
# 	if kernel=='linearkernel':
# 		for i in xrange(M):
# 			temp_kernel1[i]=linearkernel(X[i],X[i1])
# 	elif kernel=='gausskernel':
# 		for i in xrange(M):
# 			temp_kernel1[i]=gausskernel(X[i],X[i1])
# 	E1=sum(alpha*Y*temp_kernel1)-b-y1
# 	r1=y1*E1
# 	if (r1< -tol and alpha1<C) or (r1 >tol and alpha1>0):
# 		i2=-1
# 		tmax=0
# 		for k in xrange(M):
# 			if alpha[k]>0 and alpha[k]<C:
# 				E2=error[k]
# 				temp=abs(E1-E2)
# 				if temp>tmax:
# 					tmax=temp
# 					i2=k
# 			if i2>=0:
# 				if takeStep(X,Y,C,i1,i2,kernel):
# 					return 1
# 		k0=random.randint(0,M-1)
# 		for k in xrange(k0,M+k0):
# 			i2=np.mod(k,M)
# 			if alpha[i2]>0 and alpha[i2]<C:
# 				if takeStep(X,Y,C,i1,i2,kernel):
# 					return 1
# 		k0=random.randint(0,M-1)		
# 		for k in xrange(k0,M+k0):
# 			i2=np.mod(k,M)
# 			if takeStep(X,Y,C,i1,i2,kernel):
# 				return 1		
# 	return 0


# @jit
# def SMO(X,Y,pointx,pointy,kernel='linearkernel'):
# 	'this is exactly main code for the algorithm'
# 	M,N=np.shape(X)
# 	numChanged=0
# 	examineAll=1
# 	while numChanged>0 or examineAll:
# 		numChanged=0
# 		if examineAll:
# 			for i in xrange(M):
# 				numChanged=numChanged+examineExample(i)
# 		else:
# 			for i in xrange(M):
# 				if alpha[i] !=0 and alpha[i] !=C:
# 					numChanged=numChanged+examineExample(i)
# 		if examineAll==1:
# 			examineAll=0
# 		elif numChanged==0:
# 			examineAll=1
#==========the code for platt's paper============


#=========================simplified SMO algorithm=============================
#no error_cache
@jit
def SimplifiedSMO(X,Y,C,kernel='linearkernel'):
	m,n=np.shape(X)
	# alpha=np.random.rand(m)
	alpha=np.zeros(m)
	b=0.0
	passes=0
	max_iter=200
	while passes<max_iter:
		num_changed_alpha=0
		# print '=========='
		for i in xrange(m):
			tempEi=0.0;tempEj=0.0
			temp=np.zeros(m)
			if kernel=='linearkernel':
				for k in xrange(m):
					temp[k]=linearkernel(X[k],X[i])
				tempEi=sum(alpha*Y*temp)
			elif kernel=='gausskernel':
				for k in xrange(m):
					temp[k]=gausskernel(X[k],X[i])
				tempEi=sum(alpha*Y*temp)
			Ei=tempEi-Y[i]+b
			# print Ei
			if (Y[i]*Ei<-tol and alpha[i]<C) or (Y[i]*Ei>tol and alpha[i]>0):
				j=random.randint(0,m-1)
				while j==i:
					j=random.randint(0,m-1)
				if kernel=='linearkernel':
					for k in xrange(m):
						temp[k]=linearkernel(X[k],X[j])
					tempEj=sum(alpha*Y*temp)
				elif kernel=='gausskernel':
					for k in xrange(m):
						temp[k]=gausskernel(X[k],X[j])
					tempEj=sum(alpha*Y*temp)
				Ej=tempEj-Y[j]+b
				alpha1=alpha[i];alpha2=alpha[j]
				if Y[i]!=Y[j]:
					L=max(0,alpha[j]-alpha[i])
					H=min(C,C+alpha[j]-alpha[i])
				if Y[i]==Y[j]:
					L=max(0,alpha[i]+alpha[j]-C)
					H=min(C,alpha[i]+alpha[j])
				if L==H:
					continue
				if kernel=='linearkernel':
					eta=2.0*linearkernel(X[i],X[j])-linearkernel(X[i],X[i])-linearkernel(X[j],X[j])
				elif kernel=='gausskernel':
					eta=2.0*gausskernel(X[i],X[j])-gausskernel(X[i],X[i])-gausskernel(X[j],X[j])				
				if eta>=0:
					continue
				alpha[j]=alpha[j]-Y[j]*(Ei-Ej)/eta
				if alpha[j]>H:
					alpha[j]=H
				elif L<=alpha[j]<=H:
					alpha[j]=alpha[j]
				elif alpha[j]<L:
					alpha[j]=L
				if abs(alpha[j]-alpha2)<1.0e-5:
					continue
				alpha[i]=alpha[i]+Y[i]*Y[j]*(alpha2-alpha[j])
				if kernel=='linearkernel':
					b1=b-Ei-Y[i]*(alpha[i]-alpha1)*linearkernel(X[i],X[i])-Y[j]*(alpha[j]-alpha2)*linearkernel(X[i],X[j])
					b2=b-Ej-Y[i]*(alpha[i]-alpha1)*linearkernel(X[i],X[j])-Y[j]*(alpha[j]-alpha2)*linearkernel(X[j],X[j])
				elif kernel=='gausskernel':
					b1=b-Ei-Y[i]*(alpha[i]-alpha1)*gausskernel(X[i],X[i])-Y[j]*(alpha[j]-alpha2)*gausskernel(X[i],X[j])
					b2=b-Ej-Y[i]*(alpha[i]-alpha1)*gausskernel(X[i],X[j])-Y[j]*(alpha[j]-alpha2)*gausskernel(X[j],X[j])
				if 0<alpha[i]<C:
					b=b1
				elif 0<alpha[j]<C:
					b=b2
				else:
					b=(b1+b2)/2.0
				num_changed_alpha=num_changed_alpha+1
		if num_changed_alpha==0:
			passes=passes+1
		else:
			passes=0
	return alpha,b
