#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Problem 1:2:b
#Analytical method for finding decision boundaries
import numpy as np
import matplotlib as plt

def intersection(mean1,mean2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = mean2/(std2**2) - mean1/(std1**2)
  c = mean1**2 /(2*std1**2) - mean2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

µ1, σ1 = 4, 2   
µ2, σ2 = 6, 3
µ3, σ3 = 5, 2

result_1_3 = intersection(µ1,µ3,σ1,σ3)[0]
result_1_2 = intersection(µ1,µ2,σ1,σ2)[1]
result_2_3 = intersection(µ2,µ3,σ2,σ3)[0]

print("The decision boundaries are -∞ to -1.0056, -1.0056 to 4.5, 4.5 to 6.8979, and 6.8979 to ∞.")


# The boundary for ω1 and ω3 is x =4.5.(also, see attached image of hand written solution.)


# Problem 1:3:b

import matplotlib.pyplot as plt
import scipy.stats
import numpy as np


x_min = -7
x_max = 16.0

µ1, σ1 = 4, 2   
µ2, σ2 = 6, 3
µ3, σ3 = 5, 2



x = np.linspace(x_min, x_max, 100)

y1 = scipy.stats.norm.pdf(x,µ1,σ1)
y2 = scipy.stats.norm.pdf(x,µ2,σ2)
y3 = scipy.stats.norm.pdf(x,µ3,σ3)

plt.plot(x,y1, color='blue')
plt.plot(x,y2, color='green')
plt.plot(x,y3, color='purple')

plt.xlim(x_min,x_max)
plt.ylim(0,0.25)
plt.title('Problem 1:3:b')
plt.xlabel("ω1=blue, ω2=green, ω3=purple")
plt.show()






# In[17]:


#Problem 1:2:c
# Solve for overall probability error.
import numpy as np
from scipy.integrate import quad

def gaus(x, mu, sig):
     return ((1/(sig * np.sqrt(2 * np.pi)) *np.exp( - (x - mu)**2 / (2 * sig**2))))*(1/3)

mu = 4
sig = 2

a = quad(gaus,-np.inf,-1.00568606, args=(mu,sig))[0]
b = quad(gaus, 4.5, 6.89790614, args=(mu,sig))[0]
c = quad(gaus,6.89790614,np.inf, args=(mu,sig))[0]

mu2 = 6
sig2 = 3

d = quad(gaus,-1.00568606, 4.5, args=(mu2,sig2))[0]
e = quad(gaus, 4.5, 6.89790614, args=(mu2,sig2))[0]

mu3 = 5
sig3 = 2
 
f = quad(gaus,-np.inf, -1.00568606, args=(mu3,sig3))[0]
g = quad(gaus,-1.00568606, 4.5, args=(mu3,sig3))[0]
h = quad(gaus,6.89790614, np.inf, args=(mu3,sig3))[0]

Overall_Probability_Error = (a+b+c+d+e+f+g+h)
print(Overall_Probability_Error)


# In[18]:


# Probem 1:3:a & b
#P(ω1), P(ω2), P(ω3) = 0.6, 0.2, 0.2
###################
#(A)

x_min = -10
x_max = 15.0

ω1 = ((1/(2 * np.sqrt(2 * np.pi)) *np.exp( - (x - 4)**2 / (2 * 2**2))))
ω2 = ((1/(3 * np.sqrt(2 * np.pi)) *np.exp( - (x - 6)**2 / (2 * 3**2))))
ω3 = ((1/(2 * np.sqrt(2 * np.pi)) *np.exp( - (x - 5)**2 / (2 * 2**2))))

p1 = 0.6
p2_3 = 0.2

post1 = (ω1*p1)/((ω1*p1) + (ω2*p2_3) + (ω3*p2_3))
post2 = (ω2*p2_3)/((ω1*p1) + (ω2*p2_3) + (ω3*p2_3))
post3 = (ω3*p2_3)/((ω1*p1) + (ω2*p2_3) + (ω3*p2_3))


x = np.linspace(x_min, x_max, 100)


plt.plot(x,post1, color='blue')
plt.plot(x,post2, color='green')
plt.plot(x,post3, color='purple')
plt.title('Problem 1:3:a & b')
plt.xlabel("ω1=blue, ω2=green, ω3=purple")
plt.show()

#(B) 
print("Using MAP, an approximate decision boundaries can be observed from --∞ to ~-3.0, ~-3.0 to ~7.0, and ~7.0 to ∞. X = 4.7 is found in within the decision boundary, ~-3.0 to ~7.0.")


# In[61]:


#Problem 2

import matplotlib.pyplot as plt
from scipy.stats import uniform
import numpy as np
from scipy.integrate import quad

#Negative
class_1 = uniform(loc = 0, scale = 1)
#Positive
class_2 = uniform(loc = 0.95, scale = 3.95)

x = np.linspace(-10, 15, 1000)
b, ax = plt.subplots(1, 1)
ax.plot(x, class_2.pdf(x), color='blue')
ax.plot(x, class_1.pdf(x), color='green')
plt.title('Uniform PDF of Class 1 and Class 2.')
plt.show()
print(b)

#Part a & B
#Positive prob/error  0.95, 0.97
#Negative prob/error 0.97, 1.0

a = 0
b = 1
c = 0.95
d = 3.95
arbx = 0.97

def uni(x,b,a):
    return ((1/(b-a)))

def uni2(x,d,c):
    return ((1/(d-c)))
print(uni)
FN = (quad(uni,arbx, b, args=(b,a)))[0]*(1/2)
FP = (quad(uni,c,arbx, args=(d,c)))[0]*(1/2)

FN = round(FN,4)
FP = round(FP,4)

print("The probability of a false negative classification is",FN,".")
print("The probability of a false positve classification is",FP,".")
print("The overall error is", FP +FN)
#False Negative=0.015
#False Positive=0.0033
#Overall error=0.0183


# In[55]:


#Problem 2-3
#The arbitrary decision boundary of 0.97 is not the optimal decision boundary for minimization of probability error. 
#Ideally, the decision boundary for Class 1 and Class 2 would be at 1, and 0.95 respectively would produce and error of zero.
#It should be mentioned that an error of zero is unrealistic and would be considered a skeptical result.
#Lowering the value of prior probability would cause the overall error value to decrease as the prior probability approached zero.


a = 0
b = 1
c = 0.95
d = 3.95
eqprob = 0.5

idealHigh = 1
idealLow = 0.95
def uni(x,b,a):
    return ((1/(b-a)))

def uni2(x,d,c):
    return ((1/(d-c)))
print(uni)
Ideal1 = round((quad(uni,1, idealHigh, args=(b,a)))[0]*(1/2), 4)
Ideal2 = round((quad(uni,idealLow, 0.95, args=(d,c)))[0]*(1/2), 4)

print(Ideal1)
print(Ideal2)


# In[ ]:





# In[ ]:





# In[ ]:




