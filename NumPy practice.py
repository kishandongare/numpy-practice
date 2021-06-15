#!/usr/bin/env python
# coding: utf-8

# # practice NumPy

# numpy is the core library for scientific and numerical computing in python
# 
# it provide high performance multidimensional array object and tools for working with arrays
# 
# Numpy main object is the multidimentional arrays
# 
# it is a table of element all of the same type,indexed by  a tuple of positive integers
# 
# in Numpy,dimensions are called axes

# In[1]:


import numpy as np


# In[2]:


a = np.array([1,2,3]) #1D
print(a)


# In[3]:


b = np.array([[4,5,6],[7,8,9]]) #2D
print(b)


# In[4]:


#get dimension
a.ndim


# In[5]:


b.ndim


# In[6]:


#get shape
a.shape


# In[7]:


b.shape


# In[8]:


#get type
a.dtype


# In[9]:


b.dtype


# In[10]:


#get size 8+8+8+8 int32
a.itemsize


# In[11]:


b.itemsize


# In[12]:


#get array size
a.size


# In[13]:


b.size


# In[14]:


#get total size
a.nbytes


# In[15]:


b.nbytes


# # Accessing/Changing specific elements,rows,columns,etc

# In[16]:


a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a)


# In[17]:


a.shape


# In[18]:


#get a specific element
a[1,5]


# In[19]:


a[1,-2]


# In[20]:


#getting a specific column
a[0,:]


# In[21]:


#get a specific column
a[:,5]


# In[22]:


#getting a little more fancy [startindex:endindex:stepsize]
a[0,1:7:2]


# In[23]:


a[0,1:-1:2]


# In[24]:


a[1,5] = 20
print(a)


# In[25]:


a[:,2] = 5
print(a)


# In[26]:


a[:,2] = [1,4]
print(a)


# # 3D example

# In[27]:


b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(b)


# In[28]:


b[0,1,1]


# In[29]:


b[1,1,1]


# In[30]:


b[:,1,:]


# In[31]:


b[:,0,:]


# In[32]:


#replace
b[:,1,:] = [[9,9],[8,7]]
print(b)


# # Initialization different Type of Array

# In[33]:


#all 0s matrix
np.zeros((2,3))


# In[39]:


np.zeros((3,2,3,3))


# In[42]:


#all 1s matrix
np.ones((4,2,2),dtype = 'int32')


# In[44]:


#all other nunber
np.full((2,2),99,dtype ='float32')


# In[45]:


#All othe full like
np.full_like(a,4)


# In[46]:


#random decimal number
np.random.rand(4,2)


# In[47]:


np.random.rand(4,2,3)


# In[49]:


#random integer value
np.random.randint(7,size = (3,3))


# In[52]:


np.random.randint(3,7,size = (3,3)) #in between 3,7


# In[53]:


#the identity matrix
np.identity(5)


# In[58]:


#Repeat an array
arr = np.array([[1,2,3,4]])
r = np.repeat(arr,3)
print(r)


# In[60]:


arr = np.array([[1,2,3,4]])
r = np.repeat(arr,3,axis = 0)
print(r)


# In[62]:


out = np.ones((5,5))
print(out)


# In[65]:


z = np.zeros((3,3))
print(z)


# In[67]:


z[1,1] = 9
print(z)


# In[75]:


out = np.ones((5,5))
print(out)
z = np.zeros((3,3))
print(z)
z[1,1] = 9
print(z)
out[1:4,1:4] = z
print(out)


# In[77]:


#copy array
a = np.array([1,2,3])
b = a.copy()
print(a)
b[0] = 23
print(b)


# # mathematics
# 

# In[78]:


a  = np.array([1,2,3,4])
print(a)


# In[79]:


a+2


# In[80]:


a-2


# In[81]:


a*3


# In[82]:


a/3


# In[83]:


b = np.array([1,0,1,0])
a+b


# In[84]:


a**2


# In[85]:


np.cos(a)


# # Linear Algebra
# 

# In[90]:


a = np.full((2,3),2)
print(a)
b = np.full((3,2),2)
print(b)
np.matmul(a,b)


# In[92]:


c = np.identity(3)
np.linalg.det(c)


# In[ ]:


#we can also de 
#(https://docs.scipy.org/doc/numpy/reference/routines.linalg.html)


#determinant
#Traces
#singular vector decomposition
#Eigenvalues
#matrix Norm
#Inverse
#Etc


# # Statistics
# 

# In[94]:


stats = np.array([[1,2,3],[4,5,6]])
stats


# In[96]:


np.min(stats)


# In[97]:


np.max(stats)


# In[98]:


np.sum(stats)


# In[99]:


np.sum(stats,axis = 0)


# # Reorganizing arrays

# In[100]:


before = np.array([[1,2,3],[4,5,6]])
print(before)

after = before.reshape((2,3))
print(after)


# In[101]:


#Vertically stacking vectors
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])
np.vstack([v1,v2,v1])


# In[106]:


#Horizontal stacks
h1 = np.ones((2,4))
h2 = np.zeros((2,5))
np.hstack((h1,h2,h1))
    


# # Miscellaneous things

# In[ ]:


#load data from file


# In[ ]:


filedat = np.genfromtxt('data.txt',delimites = ',') #file data in the form of matrix
filedata = filedata.astype('int32')
print(filedata)


# In[ ]:


#boolean masking and indexing


# In[ ]:


filedata[filedata>50]


# In[ ]:


np.any(filedata>50,axis = 0)


# In[ ]:


((filedata > 50 & filedata < 100))


# In[ ]:




