#!/usr/bin/env python
# coding: utf-8

# In[158]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Training data


synth_train = pd.read_csv('synth.tr.txt',delim_whitespace = 1, header=None)
synth_train.columns = ['x1', 'x2', 'y']
tr0, tr1 = synth_train[synth_train['y']==0], synth_train[synth_train['y']==1]



synth_test = pd.read_csv('synth.te.txt',delim_whitespace = 1, header=None)
synth_test.columns = ['x1', 'x2', 'y']
te0, te1 = synth_test[synth_test['y']==0], synth_test[synth_test['y']==1]

def gaussian_mle(synth):                                                                                                                                                                               
    mu = synth.mean()                                                                                                                                                                            
    var = (synth-mu).T @ (synth-mu) / synth.shape[0] #  this is slightly suboptimal, but instructive

    return mu, var
#tr0Co = [gaussian_mle(tr0)[1]['x1'], gaussian_mle(tr0)[1]['x2']]

tr0mu = [gaussian_mle(tr0)[0][0], gaussian_mle(tr0)[0][1]]
gaussian_mle(tr0) #tr0Cov
tr0Cov = np.array([[0.274595, 0.011139],[0.011139, 0.035830]])
tr1mu = [gaussian_mle(tr1)[0][0], gaussian_mle(tr1)[0][1]]
gaussian_mle(tr1) #tr1Cov 
tr1Cov = np.array([[0.15847, -0.015450],[-0.01545, 0.029719]])
#print(tr0mu)
#print(tr1mu)
#print(tr1Cov)
#print(tr0Cov)

tr0x1, tr0x2, tr0y = tr0['x1'].tolist(), tr0['x2'].tolist(), tr0['y'].tolist()
tr1x1, tr1x2, tr1y = tr1['x1'].tolist(), tr1['x2'].tolist(), tr1['y'].tolist() 
te0x1, te0x2, te0y = te0['x1'].tolist(), te0['x2'].tolist(), te0['y'].tolist()
te1x1, te1x2, te1y = te1['x1'].tolist(), te1['x2'].tolist(), te1['y'].tolist()
#Plot tr0,tr1,te0,te1, y=mx+b

 
# use the function regplot to make a scatterplot

#sns.plt.show()
 
# Without regression fit:
#sns.regplot(x=df["sepal_length"], y=df["sepal_width"], fit_reg=False)
#sns.plt.show()

##################################################################
#I NEED TO PLOT THESE.
plt.figure(figsize=(16, 16))
plt.scatter(tr0x1, tr0x2, label="Training Data, Class 0")
plt.scatter(tr1x1, tr1x2, label="Training Data, Class 1")
plt.scatter(te0x1, te0x2, label="Testing Data, Class 0")
plt.scatter(te1x1, te1x2, label="Testing Data, Class 1")
plt.title("CASE 1")
plt.show()


# In[152]:


|


# In[ ]:


plt.figure(figsize=(16, 16))
plt.scatter(tr0x1, tr0x2, label="Training Data, Class 0")
plt.scatter(tr1x1, tr1x2, label="Training Data, Class 1")
plt.scatter(te0x1, te0x2, label="Testing Data, Class 0")
plt.scatter(te1x1, te1x2, label="Testing Data, Class 1")
plt.title("CASE 2")
plt.show()


# In[168]:


plt.figure(figsize=(16, 16))
plt.scatter(tr0x1, tr0x2, label="Training Data, Class 0")
plt.scatter(tr1x1, tr1x2, label="Training Data, Class 1")
plt.scatter(te0x1, te0x2, label="Testing Data, Class 0")
plt.scatter(te1x1, te1x2, label="Testing Data, Class 1")



plt.title("CASE 3")
plt.show()


# In[ ]:





# In[ ]:





# In[166]:


import matplotlib.pyplot as plt
import numpy as np

bill = [.34,.108,.64,.88,.99,.51]
tip =  [.5,.17,.11,.8,.14,.5]  
plt.scatter(bill, tip)

#fit function
f = lambda x: 0.1462*x - 0.8188
# x values of line to plot
x = np.array([-0.5,1.5])
# plot fit
plt.plot(x,f(x),lw=1, c="k",label="fit line between 0 and 100")

#better take min and max of x values
x = np.array([min(bill),max(bill)])
plt.plot(x,f(x), c="orange", label="fit line between min and max")

plt.legend()
plt.show()


# In[ ]:




