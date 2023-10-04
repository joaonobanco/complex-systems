#!/usr/bin/env python
# coding: utf-8

# In[109]:


import pandas as pd
import numpy as np

class StageMatrix:
    
    def __init__(self, num_rows, num_stages, index, init=0.0):
        
        idx_stages = [str(s) for s in np.arange(num_stages)]
        
        init_data = np.repeat(init, num_rows*num_stages).reshape(num_rows, num_stages)
        
        self._matrix = pd.DataFrame(init_data, index=index, columns=idx_stages)
        

    def get_row(self, row):
        
        return self.matrix.loc[row,:]
        
    def set_row(self, row, values):
        
        self.matrix.loc[row,:] = values
    
    
    def get_column(self, column):
        
        return self.matrix.loc[:,column]
        
    def set_column(self, column, values):
    
        self.matrix.loc[:,column] = values
        
    def add_column(self, column, values):
        
        if column == '0':
            old = np.repeat(0.0,len(values))
        else:
            old = self.get_column(str(int(column)-1))
        
        new = np.add(old,values)
        
        self.matrix.loc[:,column] = new
    
    
    def get_at(self, row, column):
        
        return self.matrix.loc[row,column]

    def set_at(self, row, column, value):
        
        self.matrix.at[row,column] = value
        
    
    @property
    def matrix(self):
        return self._matrix