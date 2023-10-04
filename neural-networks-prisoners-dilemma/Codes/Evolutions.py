#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def swap_det(rateA, rateB):
    
    if rateA < rateB:
        return True
    else:
        return False
    

def swap_prob(rateA, rateB):
    
    pAtoB = 1.0 / (1.0 + 10.0**((rateA-rateB)/400))
        
    return np.random.choice([True, False], p=[pAtoB, 1-pAtoB])


def swap_prob_exp(rateA, rateB):
    
    pAtoB = 1.0 / (1.0 + np.exp((rateA-rateB)/400))
        
    return np.random.choice([True, False], p=[pAtoB, 1-pAtoB])


def swap_prob_50():
    
    return np.random.choice([True, False], p=[0.50, 0.50])