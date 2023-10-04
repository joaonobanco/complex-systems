#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


class WinScoring:
    
    def __init__(self, players, tournament_file, repetitions=1, index_by_strat=False):
        
        self._file = pd.read_csv(tournament_file, index_col=0, header=0)
        
        self._reps = repetitions
        
        self.N = len(players)
        
        if index_by_strat == True:
            self.idx_players = [str(p) for p in players]
            self._flag_index = True
        else:
            self.idx_players = [str(n) for n in np.arange(self.N)]
            self._flag_index = False
            
    
        init_data = np.repeat(0.0, self.N*self.N).reshape(self.N, self.N)
        
        self._win_matrix = pd.DataFrame(init_data, index=self.idx_players, columns=self.idx_players)
        
                                
        self._calculate(repetitions)
    
        
    def _calculate(self, repetitions):
    
        for rep in np.arange(repetitions):
        
            for match in self._file.index.unique()[rep::repetitions]:

                # Win/Lose. One of (0,0), (1,0), (0,1), or (1,1).
                state_Player, state_Opponent = self._file.loc[match,"Win"].to_numpy()
    
                # Player/Opponent index
                if self._flag_index == True:
                    index_Player, index_Opponent = self._file.loc[match,"Player name"].to_numpy()
                else:
                    index_Player, index_Opponent = self._file.loc[match,"Player index"].to_numpy()
                    
                index_Player = str(index_Player)
                index_Opponent = str(index_Opponent)

                # Player wins
                if state_Player > state_Opponent:
                    self._win_matrix.at[index_Player, index_Opponent] += 1
                # Opponent wins
                elif state_Opponent > state_Player:
                    self._win_matrix.at[index_Opponent, index_Player] += 1
                
    @property
    def win_matrix(self):
        return self._win_matrix.to_numpy()

