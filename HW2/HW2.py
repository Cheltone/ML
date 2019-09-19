#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
import math
import sys
import statistics


# In[114]:


#MANUAL CALCULATIONS
mean1 = (0.8 + 0.9 + 1.2 + 1.1)/4
mean1_1 = (1.2 + 1.4 + 1.4 + 1.5)/4
mean2 = (0.8 + 0.6+ 0.65 + 0.75)/4
mean2_1 = (1.1 +1 + 1.1 + 0.0)/4

var1 = (((0.8-1)**2 + (0.9 -1)**2 + (1.2-1)**2 +(1.1-1)**2)/3)
var1_1 = (((1.2-1.375)**2 + (1.4 -1.375)**2 + (1.4-1.375)**2 +(1.5-1.375)**2)/3)
var2 = (((0.8-0.7)**2 + (0.6 -0.7)**2 + (0.65-0.7)**2 +(0.75-0.7)**2)/3)
var2_1 = (((1.1-1.025)**2 + (1 -1.025)**2 + (1.1-1.025)**2 +(0.9-1.025)**2)/3)

Cv1 = (((0.8-1)*(1.2-1.375) + (0.9-1)*(1.4-1.375) + (1.2-1)*(1.4-1.375) + (1.1-1)*(1.5-1.375))/3)
Cv2 = (((0.8-.7)*(1.1-1.025) + (0.6-.7)*(1-1.025) + (0.65-0.7)*(1.1-1.025)+(0.75-.7)*(.9-1.025)))/3

#


'''
0.033333333333333326
0.015833333333333335
0.008333333333333337
0.009166666666666674
0.01666666666666667
3.469446951953614e-18
'''
print(var1)
print(var1_1)
print(var2)
print(var2_1)
print(Cv1)
print(Cv2)


# In[115]:


data = pd.read_csv('hw2.txt',delim_whitespace = 1, header=None)
data.columns = ['X', 'Y', 'Class']
Class1, Class2 = data[data['Class']==0], data[data['Class']==1]

print(Class1)
print(Class2)

    


# In[116]:


print(statistics.variance(Class1['X']))
print(statistics.variance(Class1['Y']))
print(statistics.variance(Class2['X']))
print(statistics.variance(Class2['Y']))


# In[117]:


#1B
x1 = np.array([[0.85, 1.15]])
def C1(x, y):
    mu = np.array([1.0, 1.375])
    Cov_1 = np.array([[0.0333, 0.0167], [0.0167, 0.0158]])
    A1 = 1
    d = 2
    Cov_Inv = np.linalg.inv(Cov_1)
    xar = np.array([x,y])
    
    pa = (2*math.pi)**(d/2)
    pb = A1/((np.linalg.det(Cov_1))**(1/2)*pa )
    pc = np.exp((-1/2)*(np.dot(np.dot(np.transpose((xar - mu)),Cov_Inv),(xar-mu))))
    p1 = pb * pc
    return p1
    
def C2(x, y): 
    mu = np.array([0.7, 1.025])
    Cov_2 = np.array([[0.0083, 0], [0, 0.0092]])
    A1 = 1
    d = 2
    Cov_Inv = np.linalg.inv(Cov_2)
    xar = np.array([x,y])
    
    pa = (2*math.pi)**(d/2)
    pb = A1/((np.linalg.det(Cov_2))**(1/2)*pa )
    pc = np.exp((-1/2)*(np.dot(np.dot(np.transpose((xar - mu)),Cov_Inv),(xar-mu))))
    p1 = pb * pc
    return p1



fig = plt.figure(figsize=(12,12))
x = np.linspace(.5, 1.5, 50)
y = np.linspace(0.8, 2, 50)
A, B = np.meshgrid(x,y)
C = np.vectorize(C2)
D = np.vectorize(C1)
plt.scatter(0.85, 1.15, marker='D', Color = 'Red', label="Test point, [x = [0.85 1.15]T]")
plt.scatter(Class1['X'], Class1['Y'], label="Training Data, Class 1", marker = '*')
plt.scatter(Class2['X'], Class2['Y'], label="Training Data, Class 2", marker = '.')
plt.legend()
plt.contour(A,B, C(A,B))
plt.contour(A,B, D(A,B))


# In[118]:


#1eii
from scipy.spatial import distance

x1 = np.transpose(np.array([[0.85, 1.15]]))

Mu1 = np.array([1.0, 1.375])
Mu2 = np.array([0.7, 1.025])

#Euclidean 1
dist1 = math.sqrt((x1[0]-Mu1[0])**2 + (x1[1]-Mu1[1])**2)
dist2 = math.sqrt((x1[0]-Mu2[0])**2 + (x1[1]-Mu2[1])**2)

'''
From SCIPY
dist1 = distance.euclidean(x1, Mu1)
dist2 = distance.euclidean(x1, Mu2)
'''
print(dist1)
print(dist2)


# In[119]:


#1eiii
x1 = np.transpose(np.array([[0.85, 1.15]]))
Cov1 = np.array([[0.0333, 0.0167], [0.0167, 0.0158]])
Cov1_inv = np.linalg.inv(Cov1)
Cov2 = np.array([[0.0083, 0], [0, 0.0092]])
Cov2_inv = np.linalg.inv(Cov2)

#dist3 = np.dot(np.dot(Cov1_inv,Mu1),x1) + (np.dot(np.dot(Cov1_inv,Mu1), Mu1)) *-1/2

#SCIPY
dist1 = distance.mahalanobis(x1, Mu1 , Cov1_inv)
dist2 = distance.mahalanobis(x1, Mu2 , Cov2_inv)
print(dist1)
print(dist2)
print('X belongs to Class 1')


# In[125]:


#2a
x1 = np.array([[0.85, 1.15]])
def C1(x, y):
    mu = np.array([1.0, 1.375])
    Cov_1 = np.array([[0.1, 0], [0, 0.1]])
    A1 = 1
    d = 2
    Cov_Inv = np.linalg.inv(Cov_1)
    xar = np.array([x,y])
    
    pa = (2*math.pi)**(d/2)
    pb = A1/((np.linalg.det(Cov_1))**(1/2)*pa )
    pc = np.exp((-1/2)*(np.dot(np.dot(np.transpose((xar - mu)),Cov_Inv),(xar-mu))))
    p1 = pb * pc
    return p1
    




fig = plt.figure(figsize=(16,16))
x = np.linspace(0, 2.5, 50)
y = np.linspace(0, 2.5, 50)
A, B = np.meshgrid(x,y)

D = np.vectorize(C1)



plt.contour(A,B, D(A,B))



# In[121]:


#2b
x1 = np.array([[0.85, 1.15]])
def C1(x, y):
    mu = np.array([1.0, 1.375])
    Cov_1 = np.array([[0.1, 0], [0, 0.3]])
    A1 = 1
    d = 2
    Cov_Inv = np.linalg.inv(Cov_1)
    xar = np.array([x,y])
    
    pa = (2*math.pi)**(d/2)
    pb = A1/((np.linalg.det(Cov_1))**(1/2)*pa )
    pc = np.exp((-1/2)*(np.dot(np.dot(np.transpose((xar - mu)),Cov_Inv),(xar-mu))))
    p1 = pb * pc
    return p1
    




fig = plt.figure(figsize=(16,16))
x = np.linspace(0, 2.5, 50)
y = np.linspace(0, 2.5, 50)
A, B = np.meshgrid(x,y)

D = np.vectorize(C1)



plt.contour(A,B, D(A,B))



# In[122]:


#2c
x1 = np.array([[0.85, 1.15]])
def C1(x, y):
    mu = np.array([1.0, 1.375])
    Cov_1 = np.array([[0.1, 0.2], [0.1, 0.3]])
    A1 = 1
    d = 2
    Cov_Inv = np.linalg.inv(Cov_1)
    xar = np.array([x,y])
    
    p1a = (2*math.pi)**(d/2)
    p1b = A1/((np.linalg.det(Cov_1))**(1/2)*p1a )
    p1c = np.exp((-1/2)*(np.dot(np.dot(np.transpose((xar - mu)),Cov_Inv),(xar-mu))))
    p1 = p1b * p1c
    return p1
    




fig = plt.figure(figsize=(16,16))
x = np.linspace(0, 2, 50)
y = np.linspace(0, 2.75, 50)
A, B = np.meshgrid(x,y)

D = np.vectorize(C1)



plt.contour(A,B, D(A,B))


# In[123]:


#2d
x1 = np.array([[0.85, 1.15]])
def C1(x, y):
    mu = np.array([1.0, 1.375])
    Cov_1 = np.array([[0.1, -0.2], [-0.1, 0.3]])
    A1 = 1
    d = 2
    Cov_Inv = np.linalg.inv(Cov_1)
    xar = np.array([x,y])
    
    p1a = (2*math.pi)**(d/2)
    p1b = A1/((np.linalg.det(Cov_1))**(1/2)*p1a )
    p1c = np.exp((-1/2)*(np.dot(np.dot(np.transpose((xar - mu)),Cov_Inv),(xar-mu))))
    p1 = p1b * p1c
    return p1
    




fig = plt.figure(figsize=(16,16))
x = np.linspace(0, 2, 50)
y = np.linspace(0, 2.75, 50)
A, B = np.meshgrid(x,y)

D = np.vectorize(C1)



plt.contour(A,B, D(A,B))


# In[124]:


#4a
from scipy.spatial import distance

x1 = np.transpose(np.array([[0.85, 1.15]]))

#Euclidean 1
eq1 = math.sqrt((x1[0]-data['X'][0])**2 + (x1[1]-data['Y'][0])**2)
eq2 = math.sqrt((x1[0]-data['X'][1])**2 + (x1[1]-data['Y'][1])**2)
eq3 = math.sqrt((x1[0]-data['X'][2])**2 + (x1[1]-data['Y'][2])**2)
eq4 = math.sqrt((x1[0]-data['X'][3])**2 + (x1[1]-data['Y'][3])**2)
eq5 = math.sqrt((x1[0]-data['X'][4])**2 + (x1[1]-data['Y'][4])**2)
eq6 = math.sqrt((x1[0]-data['X'][5])**2 + (x1[1]-data['Y'][5])**2)
eq7 = math.sqrt((x1[0]-data['X'][6])**2 + (x1[1]-data['Y'][6])**2)
eq8 = math.sqrt((x1[0]-data['X'][7])**2 + (x1[1]-data['Y'][7])**2)
'''
From SCIPY
dist1 = distance.euclidean(x1, Mu1)
dist2 = distance.euclidean(x1, Mu2)
'''
print(eq1)
print(eq2)
print(eq3)
print(eq4)
print(eq5)
print(eq6)
print(eq7)
print(eq8)

'''
eq1 = 0.07071067811865474
eq2 = 0.25495097567963926
eq3 = 0.4301162633521313
eq4 = 0.4301162633521315
eq5 = 0.07071067811865459
eq6 = 0.291547594742265
eq7 = 0.20615528128088292
eq8 = 0.2692582403567251
'''


# In[ ]:





# In[ ]:




