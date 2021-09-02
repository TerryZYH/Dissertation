#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import json
import os
import random
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import shuffle
from skmultiflow.drift_detection.adwin import ADWIN
import math


# # iForest ADWIN

# In[2]:


class Node:
    def __init__(self,internal=True,left=None,right=None,sAtt=None,sVal=None,size=None):
        self.internal = internal
        self.size = size
        self.left = left
        self.right = right
        self.sAtt = sAtt
        self.sVal = sVal
        return
            


# In[3]:


def c(n):
#     print('3333',n)
    return 2*H(n-1)-(2*(n-1)/n)

def H(i):
    return math.log(i)+0.5772156649


# In[4]:


class iTree:
    def __init__(self,X,e,l,X_cati=None):
        """
        INPUTS:
        X: input data
        e: current tree height
        l: height limit

        OUTPUT:
        """
        self.X = X
        self.l = l
        self.root = self.build(X,e,l,X_cati)
        return

    def build(self,X,e,l,X_cati=None):
#         print('e:',e,'l：',l)
        if e>=l or len(X)<=1:
#             print(e)
            return Node(internal=False,size=len(X))
        else:
            q = random.randint(0,len(X[0])-1)
            if X_cati and X_cati[q]:
                p = random.choice(list(set(X[:,q])))
                Xl = X[X[:,q]==p]
                Xr = X[X[:,q]!=p]
                return Node(internal=True,
                            left=self.build(Xl,e+1,l,X_cati),
                            right=self.build(Xr,e+1,l,X_cati),
                            sAtt=q,
                            sVal=p)
            else:
                p = random.uniform(min(X[:,q]),max(X[:,q]))
                Xl = X[X[:,q]<p]
                Xr = X[X[:,q]>=p]
                return Node(internal=True,
                            left=self.build(Xl,e+1,l,X_cati),
                            right=self.build(Xr,e+1,l,X_cati),
                            sAtt=q,
                            sVal=p)
            
    def pathLength(self,x,N=None,e=0,X_cati=None):
        """
        INPUTS:
        x: an instance
        N: a Node in the iTree
        e: current path length
        """
        if not N:
            N = self.root
        if not N.internal:
            if N.size>1:
#                 print(1)
                return e+(N.size>1)*c(N.size)
            else:
#                 print(0)
                return e
        a = N.sAtt
        if X_cati and X_cati[a]:
            if x[a] == N.sVal:
                return self.pathLength(x,N.left,e+1,X_cati)
            else:
                return self.pathLength(x,N.right,e+1,X_cati)
        else:
            if x[a] < N.sVal:
                return self.pathLength(x,N.left,e+1,X_cati)
            else:
                return self.pathLength(x,N.right,e+1,X_cati)
        
    def draw(self):
        if len(self.X[0])!=2:
            print('iTree.draw() function only support 2D data')
            return
        plt.figure()
        plt.scatter(self.X[:,0],self.X[:,1])
        limits = [[min(self.X[:,0]),max(self.X[:,0])],[min(self.X[:,1]),max(self.X[:,1])]]
        self._helper(self.root,limits)
                
    def _helper(self,node,limits):
        if not node.internal:
            return
        att = node.sAtt
        val = node.sVal
        point1 = [(1-att)*val+att*limits[1-att][0],att*val+(1-att)*limits[1-att][0]]
        point2 = [(1-att)*val+att*limits[1-att][1],att*val+(1-att)*limits[1-att][1]]
        plt.plot([point1[0],point2[0]],[point1[1],point2[1]])
        limitL = [[limits[0][0],(1-att)*val+att*limits[0][1]],[limits[1][0],(1-att)*limits[1][1]+att*val]]
        limitR = [[(1-att)*val+att*limits[0][0],limits[0][1]],[(1-att)*limits[1][0]+att*val,limits[1][1]]]
        self._helper(node.left,limitL)
        self._helper(node.right,limitR)


# In[6]:


import math
import random
class iForest:
    def __init__(self,X,t,phi,X_cati=None):
        """
        INPUTS:
        X: input data
        X_type: list of data type in each dimension of X
        t: number of trees
        phi: subsampling size
        
        OUTPUT:
        Forest: a set of t iTrees
        """
        self.size = t
        self.n = phi
        self.forest = []
        self._Train(X,t,phi,X_cati)
        return
        
    def _Train(self,X,t,phi,X_cati):
        # l: height limit of iTrees
        l = math.ceil(math.log2(phi))
#         l = phi/2
#         l = min(math.ceil(math.log2(phi))*2,phi/2)
        for i in range(t):
            # X_prime: subsample of X, used for training the ith iTree
            X_prime = X[np.random.choice(X.shape[0], phi, replace=False), :]
#             print(X_prime.shape)
            self.forest.append(iTree(X_prime,0,l,X_cati))
    
    def predict(self,x):
        h_sum = 0
        for i in range(self.size):
#             print(self.forest[i].pathLength(x))
            h_sum += self.forest[i].pathLength(x)
        E = h_sum/self.size
#         print('E',E)
#         print('c',c(self.n))
        return 2**(-E/c(self.n))
        


# In[7]:


class ADWIN1:
    def __init__(self,delta=0.02):
        self.W = []
        self.delta = delta
        self.change = False
        
    
    def add_element(self,x):
        self.change = False
        self.W.append(x)
        while len(self.W)>1 and self.driftTest():
            self.change = True
            del self.W[0]
            
    def detected_change(self):
        return self.change
            
    def driftTest(self):
        for i in range(1,len(self.W)-1):
            W0 = self.W[:i]
            W1 = self.W[i:]
            n0 = len(W0)
            n1 = len(W1)
            n = len(self.W)
#             print('n0',n0,'n1',n1)
            m = 1/(1/n0+1/n1)
            delta_prime = self.delta/n
            epsilon_cut = (1/(2*m)*np.log(4/delta_prime))**(1/2)
            mu0 = np.mean(W0)
            mu1 = np.mean(W1)
            if abs(mu0-mu1)>=epsilon_cut:
                return True
        return False
            
        


# In[8]:


class MADWIN:
    def __init__(self,delta=0.02,min_win_size=50,max_size=250):
        self.W = []
        self.delta = delta
        self.min_win_size = min_win_size
        self.max_size = max_size
        self.change = False
        
    
    def add_element(self,x):
        if len(self.W)>=self.max_size:
            del self.W[0]
        self.change = False
        self.W.append(x)
        while len(self.W)>1 and self.driftTest():
            self.change = True
            del self.W[-1]
            
    def detected_change(self):
        ans = self.change
        self.change = False
        return ans
            
    def driftTest(self):
        for i in range(self.min_win_size,len(self.W)-self.min_win_size):
            W0 = self.W[:i]
            W1 = self.W[i:]
            n0 = len(W0)
            n1 = len(W1)
            n = len(self.W)
#             print('n0',n0,'n1',n1)
            m = 1/(1/n0+1/n1)
            delta_prime = self.delta/n
            epsilon_cut = (1/(2*m)*np.log(4/delta_prime))**(1/2)
            mu0 = np.mean(W0)
            mu1 = np.mean(W1)
            if abs(mu0-mu1)>=epsilon_cut:
                return True
        return False
            
        


# In[9]:


class Detector:
    def __init__(self,t,phi,delta=0.02,min_win_size=50):
        """
        INPUTS:
        t: iforest number of trees
        phi: iforest subsampling size
        delta: adwin threshold
        min_win_size: adwin minimal window size
        """
        self.t = t
        self.phi = phi
        self.delta = delta
        self.selector = VarianceThreshold()
        self.madwin = MADWIN(delta=self.delta)
        self.adwin = ADWIN(delta=self.delta)
        self.feature_filter = None
        self.iforest = None
        return
    
    def train(self,X):
        X_prime = self.selector.fit_transform(X)
        self.feature_filter = self.selector.get_support()
        self.iforest = iForest(X_prime, self.t, self.phi)
        return
    
    def predict(self,x):
        x = x.reshape((1,-1))
        xp = self.selector.transform(x)
        xp = xp.reshape((-1,))
        s = self.iforest.predict(xp)
        self.madwin.add_element(s*4)
        self.adwin.add_element(s)
        adwins = 0
        if self.adwin.detected_change():
            adwins = 1
        if self.madwin.detected_change():
            return -1, adwins
        return s, adwins


# In[172]:


class algorithm:
    def __init__(self,e,window_size,t,phi,delta=0.02,min_win_size=50,thresh=0.7):
        """
        INPUT:
        e: number of detectors
        window_size: buffer size
        t: iforest number of trees
        phi: iforest subsampling size
        delta: adwin threshold
        min_win_size: adwin minimal window size
        thresh: abnormal threshold
        """
        self.e = e
        self.window_size = window_size
        self.t = t
        self.phi = phi
        self.delta = delta
        self.thresh = thresh
        self.min_win_size = min_win_size
        self.selector = VarianceThreshold()
        self.previous_window = []
        self.current_window = []
        self.ensemble = []
        return
    
    def predict(self,x):
        output = []
        var = []
        if len(self.current_window)==self.window_size:
            self.previous_window = self.current_window
            self.current_window = []
            # feature selection:
            self.selector.fit(self.previous_window)
            feature_filter = self.selector.get_support()
            if self._feature_drift_detection(feature_filter) or len(self.ensemble)==0:
                # feature drift occurs OR no existing detector in ensemble
                detector = Detector(self.t,self.phi,self.delta,self.min_win_size)
                detector.train(self.previous_window)
                if len(self.ensemble)>=self.e:
                    del self.ensemble[0]
                self.ensemble.append(detector)
                output = [-1]*len(self.previous_window)
                var = [-1]*len(self.previous_window)
                return output,var
            else:
                for sample in self.previous_window:
                    scores = np.zeros((len(self.ensemble),))
                    for i in range(len(self.ensemble)):
                        detector = self.ensemble[i]
                        s,_ = detector.predict(sample)
                        scores[i] = s
                    var.append(scores)
                    valid_scores = scores[scores>=0]
                    if len(valid_scores)>0:
                        score = np.mean(valid_scores)
                        if score>self.thresh:
                            output.append(0)
                        else:
                            output.append(1)
                    else:
                        detector = Detector(self.t,self.phi,self.delta,self.min_win_size)
                        detector.train(self.previous_window)
                        if len(self.ensemble)>=self.e:
                            del self.ensemble[0]
                        self.ensemble.append(detector)
                        output = [-1]*len(self.previous_window)
                        var = [-1]*len(self.previous_window)
                        return output,var
        self.current_window.append(x)
        return output,var
        
    def _feature_drift_detection(self,feature_filter):
        for detector in self.ensemble:
            if np.array_equal(detector.feature_filter,feature_filter):
                return False
        return True
        


# In[190]:


def evaluation(prediction,label):
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(label)):
        if prediction[i] != -1:
            if prediction[i] and label[i]:
                TP += 1
            elif prediction[i] and not label[i]:
                FP += 1
            elif not prediction[i] and label[i]:
                FN += 1
            else:
                TN += 1
    ACC = (TP+TN)/(TP+TN+FP+FN)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1 = 2*Precision*Recall/(Precision+Recall)
    FPR = (FP+1)/(FP+TN+1)
    TPR = (TP+1)/(TP+FN+1)
    print('TP:',TP,'TN:',TN,'FP:',FP,'FN:',FN)
    print('ACC:',ACC,'Precision:',Precision,'Recall:',Recall,'F1:',F1)
    print('FPR:',FPR,'TPR:',TPR)
    return


# In[ ]:




