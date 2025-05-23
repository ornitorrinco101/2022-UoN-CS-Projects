# -- coding: utf-8 --
"""
Created on Tue Oct 25 12:00:35 2016

@author: ppzfrp, modified by RJAH
"""
# set up random 3-d positions
#
import numpy as np
import time
import errno
N=10000
seed=1234
np.random.seed(seed)
pos=np.random.random((3,N))
start_time=time.time()
# deliberately slow code to find nearest neighbours within periodic unit cube
#                                                                            
#  You may only change the code between here and the line "end_time=time.time()")

'''
r=[0,1,2]
for i in r:
    for a in range(N):                                  
        for b in range(N):
            d=abs(pos[i,a]-pos[i,b])
            mn=min(d,1-d)
            s[a,b]+=mn**2
'''

'''
r=[0,1,2]
for i in r:
    A1=pos[i,:]
    A2=np.reshape(A1,(N,1))
    d=abs(np.repeat(A2,N,axis=-1)-np.full((N,N),A1))
    s+=(np.minimum(d,ones-d))**2
'''

'''
A1=np.array([np.array([pos[0,:]]),np.array([pos[1,:]]),np.array([pos[2,:]])])
A2=np.reshape(A1,())
A=np.repeat(A2,N,axis=-1)
B=np.full((N,N),A1)
d=abs(A-B)
s=np.sum(np.minimum(d,ones-d)**2)
'''
#code that worksafter here
ones=np.ones((N,N))
A1=np.array([[pos[0,:]],[pos[1,:]],[pos[2,:]]])
A2=np.reshape(A1,(3,N,1))
A=(np.repeat(A2,N,axis=-1))
B=(np.full_like(A,A1))
dist=np.abs(A-B)
s=np.sum((np.minimum(dist,ones-dist))**2,axis=0)

A2=s+np.diagflat(ones[1,:])
matchedIndices=np.argmin(A2,axis=0)
end_time = time.time()
print('Elapsed time = ', repr(end_time - start_time))

# generate filename from N and seed
filename = 'pyneigh' + str(N) + '_' + str(seed)
# if a file with this name already exists read in the nearest neighbour
# list found before and compare this to the current list of neighbours,
# else save the neighbour list to disk for future reference
try:
    fid = open(filename,'rb')
    matchedIndicesOld = np.loadtxt(fid)
    fid.close()
    if (matchedIndicesOld == matchedIndices).all(): 
        print('Checked match')
    else:
        print('Failed match')
except OSError as e:
    if e.errno == errno.ENOENT:
        print('Saving neighbour list to disk')
        fid = open(filename,'wb')
        np.savetxt(fid, matchedIndices, fmt="%8i")
        fid.close()
    else:
        raise