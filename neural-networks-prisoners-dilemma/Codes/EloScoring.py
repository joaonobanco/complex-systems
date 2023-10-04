#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import elo


class EloScoring:
    
    def __init__(self, players, tournament_file, init_rate=1200.0, repetitions=1, elo_params=[10,200], index_by_strat=False):
        
        self._file = pd.read_csv(tournament_file, index_col=0, header=0)
        
        self._reps = repetitions
        
        self.N = len(players)
        
        if index_by_strat == True:
            self.idx_players = [str(p) for p in players]
            self._flag_index = True
        else:
            self.idx_players = [str(n) for n in np.arange(self.N)]
            self._flag_index = False
            
        
        try:
            if len(init_rate) == 1:
                init_scores = np.repeat(init_rate, self.N)
            else:
                init_scores = init_rate
        except:
            init_scores = init_rate
        
        self._scores = pd.Series(init_scores, index=self.idx_players)
    
    
        init_data = np.repeat(0.0, self.N*self.N).reshape(self.N, self.N)
        
        self._payoff_matrix = pd.DataFrame(init_data, index=self.idx_players, columns=self.idx_players)
    
    
        elo.setup(k_factor=elo_params[0], beta=elo_params[1])
                                
        self._calculate(repetitions, elo_params[0])
    
        
    def _calculate(self, repetitions, elo_k):
    
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

                # Player/Opponent current Elo ratings
                rate_Player, rate_Opponent = self._scores.loc[index_Player], self._scores.loc[index_Opponent]


                # Player wins
                if state_Player > state_Opponent:
                    self._payoff_matrix.at[index_Player, index_Opponent] += elo_k*elo.adjust_1vs1(rate_Player, rate_Opponent)
                    self._scores[index_Player], self._scores[index_Opponent] = elo.rate_1vs1(rate_Player, rate_Opponent)
                # Opponent wins
                elif state_Opponent > state_Player:
                    self._payoff_matrix.at[index_Opponent, index_Player] += elo_k*elo.adjust_1vs1(rate_Opponent, rate_Player)
                    self._scores[index_Opponent], self._scores[index_Player] = elo.rate_1vs1(rate_Opponent, rate_Player)
                # Drawn match
                else:
                    self._payoff_matrix.at[index_Player, index_Opponent] += elo_k*elo.adjust_1vs1(rate_Player, rate_Opponent, drawn=True)
                    self._scores[index_Player], self._scores[index_Opponent] = elo.rate_1vs1(rate_Player, rate_Opponent, drawn=True)
                
    @property
    def scores(self):
        return self._scores.to_numpy()
    
    @property
    def ranked_names(self):
        return self._scores.sort_values(ascending=False).index.to_numpy()
            
    @property
    def payoff_matrix(self):
        return self._payoff_matrix.div(self._reps).to_numpy()

