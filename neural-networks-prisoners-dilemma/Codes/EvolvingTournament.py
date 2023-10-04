#!/usr/bin/env python
# coding: utf-8

# In[2]:


from EloScoring import EloScoring
from StageMatrix import StageMatrix
import Evolutions
import axelrod as axl
import numpy as np
import pandas as pd
import itertools as it
import networkx as nx

class EvolvingTournament():

    def __init__(self, base_strategies, multiples=1, evolution_stages=0, graph="BarabasiAlbert", external_graph=False, init_elo=1200.0, seed=0):
        
        self.num_Stages = evolution_stages
        
        
        self.Strategies = base_strategies
        self.num_Strategies = len(self.Strategies)
        self.idx_Strategies = [str(s) for s in self.Strategies]
        
        
        init_strategies = self._generate_Strategies(base_strategies, multiples)
        self.num_Players = len(init_strategies)
        self.idx_Players = [str(p) for p in np.arange(self.num_Players)]
        
        
        self.data_Strategies = StageMatrix(self.num_Players, self.num_Stages+1, self.idx_Players, init='A')
        self.data_Strategies.set_column('0', init_strategies)
        
        
        self.data_Cooperation = StageMatrix(self.num_Players, self.num_Stages+1, self.idx_Players)
        
        
        self.rate_Axel = StageMatrix(self.num_Players, self.num_Stages+1, self.idx_Players, init=0.0)
        self.rate_Elo = StageMatrix(self.num_Players, self.num_Stages+1, self.idx_Players, init=0.0)
        self.rate_Elo.set_column('0', init_elo)
        
        
        self.data_Population = StageMatrix(self.num_Strategies, self.num_Stages+1, self.idx_Strategies)
        self.data_Population.set_column('0', np.repeat(multiples, self.num_Strategies))
        
        
        self.info_Axel = None
        self.info_Elo = None

        
        ### Other
        
        # Generate graph/network
        self.avg_k = None
        self.graph = self._generate_Graph(seed=seed, graph=graph, external_graph=external_graph)
        
        
        
        # Colors
        #self.colors_Colorized = pd.Series([ "rgb" + str(i) for i in list([(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40), (255, 250, 200), (128, 0, 0), (128, 128, 0), (0, 0, 128)])], index=self.idx_Strategies)
        
        self.colors_GreyScale = [np.repeat(8*i/(10*self.num_Strategies),3) for i in np.arange(self.num_Strategies)]
        
        if len(self.idx_Strategies) == 15:
            self.colors_Colorized = pd.Series([ "rgb" + str(c) for c in list([(0.1015625, 0.90234375, 0.70703125),(0.765625, 0.296875, 0.70703125),(0.00390625, 0.12109375, 0.90234375),(0.73828125, 0.61328125, 0.15625),(0.04296875, 0.4921875, 0.80859375),(0.43359375, 0.8828125, 0.296875),(0.7265625, 0.0625, 0.0625),(0.0625, 0.8046875, 0.1015625),(0.265625, 0.0390625, 0.953125),(0.0234375, 0.2578125, 0.2578125),(1.0, 0.5, 0.5),(0.1015625, 0.2578125, 0.00390625),(0.3984375, 0.61328125, 0.859375),(0.00390625, 0.0234375, 0.21875),(0.5, 1.0, 1.0)])], index=self.idx_Strategies)
            
        if len(self.idx_Strategies) == 14:
            self.colors_Colorized = pd.Series([ "rgb" + str(c) for c in list([(0.1015625, 0.90234375, 0.70703125),(0.765625, 0.296875, 0.70703125),(0.00390625, 0.12109375, 0.90234375),(0.73828125, 0.61328125, 0.15625),(0.04296875, 0.4921875, 0.80859375),(0.43359375, 0.8828125, 0.296875),(0.0625, 0.8046875, 0.1015625),(0.265625, 0.0390625, 0.953125),(0.0234375, 0.2578125, 0.2578125),(1.0, 0.5, 0.5),(0.1015625, 0.2578125, 0.00390625),(0.3984375, 0.61328125, 0.859375),(0.00390625, 0.0234375, 0.21875),(0.5, 1.0, 1.0)])], index=self.idx_Strategies)
            
        if len(self.idx_Strategies) == 2:
            self.colors_Colorized = pd.Series([ "rgb" + str(c) for c in list([(0.765625, 0.296875, 0.70703125),(0.00390625, 0.12109375, 0.90234375)])], index=self.idx_Strategies)

        
    def _generate_Strategies(self, base_strategies, multiples):
        
        strategies = []
        strategies = base_strategies * multiples
        
        return strategies
        
    
    def _generate_Graph(self, seed, graph, external_graph):
        
        if external_graph == True:
            H = graph
        elif graph == "BarabasiAlbert":
            # Generate base graph/network connections (edges)
            H = nx.barabasi_albert_graph(self.num_Players, 2, seed=seed)
            self.avg_k = np.mean(np.array([d[1] for d in nx.degree(H)]))
        
        # Generate geometric distribution of nodes
        G = nx.random_geometric_graph(self.num_Players, 0.1, seed=seed+2)
        
        # Place the network into the geometry
        G.update(edges=H.edges())
        
        return G

    
    def _update_Current(self, evolution_stage, repetitions):
        
        stage = str(evolution_stage)
        
        # Cooperation rating
        new_coop = np.array(self.info_Axel.cooperating_rating)
        self.data_Cooperation.set_column(stage, new_coop)
    
        
        # Axel rating
        new_rate_axel = [np.mean(s) for s in np.array(self.info_Axel.scores)]
        self.rate_Axel.set_column(stage, new_rate_axel)
        #self.rate_Axel.add_column(stage, new_rate_axel)
        
        
        # Elo rating
        new_rate_elo = self.info_Elo.scores
        self.rate_Elo.set_column(stage, new_rate_elo)
        
    
    def _update_Next(self, evolution_stage, rate_type, evolution_type):
        
        stage = str(evolution_stage)
        stageN = str(evolution_stage+1)
        
        # Elo rating
        next_rate_elo = self.info_Elo.scores
        self.rate_Elo.set_column(stageN, next_rate_elo)
        
        # Population
        next_population = self.data_Population.get_column(stage)
        self.data_Population.set_column(stageN, next_population)
        
        # Strategies
        next_strats = self.data_Strategies.get_column(stage)
        self.data_Strategies.set_column(stageN, next_strats)
        
        
        for player in np.arange(self.num_Players):
            
            s_player = str(player)
            
            neighbors = [str(n) for n in self.graph.neighbors(player)]
            
            if rate_type == "Axel":
                
                rate_neighbors = [self.rate_Axel.get_at(n,stage) for n in neighbors]
                                
                neighbor_max = np.argmax(rate_neighbors)
                neighbor_max = str(neighbor_max)
                
                rate_Player = self.rate_Axel.get_at(s_player,stage)
                rate_Opponent = self.rate_Axel.get_at(neighbor_max,stage)
                
            elif rate_type == "Elo":
                
                rate_neighbors = [self.rate_Elo.get_at(n,stage) for n in neighbors]
                                                
                neighbor_max = np.argmax(rate_neighbors)
                neighbor_max = str(neighbor_max)
                
                rate_Player = self.rate_Elo.get_at(s_player,stage)
                rate_Opponent = self.rate_Elo.get_at(neighbor_max,stage)
                
                
            player_strat = self.data_Strategies.get_at(s_player,stage)
            s_player_strat = str(player_strat)
            
            neighbor_strat = self.data_Strategies.get_at(neighbor_max,stage)
            s_neighbor_strat = str(neighbor_strat)
            
            
            if evolution_type == "Det":
                
                if Evolutions.swap_det(rate_Player, rate_Opponent) == True:
                    
                    self.data_Strategies.set_at(s_player,stageN,neighbor_strat)
                    
                    new_pop_p = self.data_Population.get_at(s_player_strat,stageN) - 1                    
                    self.data_Population.set_at(s_player_strat,stageN, new_pop_p)
                    
                    new_pop_o = self.data_Population.get_at(s_neighbor_strat,stageN) + 1
                    self.data_Population.set_at(s_neighbor_strat,stageN, new_pop_o)
                
            elif evolution_type == "Prob":
                
                if Evolutions.swap_prob(rate_Player, rate_Opponent) == True:
                    
                    player_strat = self.data_Strategies.get_at(s_player,stage)
                    s_player_strat = str(player_strat)
                    
                    neighbor_strat = self.data_Strategies.get_at(neighbor_max,stage)
                    s_neighbor_strat = str(neighbor_strat)
                    
                    self.data_Strategies.set_at(s_player,stageN,neighbor_strat)
                    
                    new_pop_p = self.data_Population.get_at(s_player_strat,stageN) - 1
                    self.data_Population.set_at(s_player_strat,stageN, new_pop_p)
                    
                    new_pop_o = self.data_Population.get_at(s_neighbor_strat,stageN) + 1
                    self.data_Population.set_at(s_neighbor_strat,stageN, new_pop_o)
                    
            elif evolution_type == "Stoch":
                
                if Evolutions.swap_prob_50() == True:
                    
                    player_strat = self.data_Strategies.get_at(s_player,stage)
                    s_player_strat = str(player_strat)
                    
                    neighbor_strat = self.data_Strategies.get_at(neighbor_max,stage)
                    s_neighbor_strat = str(neighbor_strat)
                    
                    self.data_Strategies.set_at(s_player,stageN,neighbor_strat)
                    
                    new_pop_p = self.data_Population.get_at(s_player_strat,stageN) - 1
                    self.data_Population.set_at(s_player_strat,stageN, new_pop_p)
                    
                    new_pop_o = self.data_Population.get_at(s_neighbor_strat,stageN) + 1
                    self.data_Population.set_at(s_neighbor_strat,stageN, new_pop_o)
                
    
    ### Tournament
            
    def play_Tournament(self, rate_type, evolution_type, game=axl.game.Game(), turns=10, repetitions=1, axel_seed=0, elo_params=[10,200]):
        
        # Random seed
        axl.seed(axel_seed)
        
        # Play the tournament for a set number of stages
        for stage in np.arange(self.num_Stages+1):
            
            # Run Axelrod tournament
            
            tour = axl.Tournament(self.data_Strategies.get_column(str(stage)), edges=self.graph.edges(), game=game, turns=turns, repetitions=repetitions)
            self.info_Axel = tour.play(processes=1, filename="tournament_data_stage_" + str(stage) + ".csv")
            
            old_rate_elo = self.rate_Elo.get_column(str(stage))
            self.info_Elo = EloScoring(self.idx_Players, "tournament_data_stage_" + str(stage) + ".csv", repetitions=repetitions, init_rate=old_rate_elo, elo_params=elo_params)
    
            # Current stage
            self._update_Current(stage, repetitions)
            
            # Next stage
            if stage != self.num_Stages:
                self._update_Next(stage, rate_type, evolution_type)

