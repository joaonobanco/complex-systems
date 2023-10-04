#!/usr/bin/env python
# coding: utf-8

# # Import

# In[ ]:


from EloScoring import EloScoring
from WinScoring import WinScoring
from Plotting import Heatmap, ChordDiagram, GGraphPlot, GGraphPlot2, MeanPlot, Parallel, Spaghetti, StackedPercent, LollipopPlot, RegPlot
from EvolvingTournament import EvolvingTournament
from Evolutions import *

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
from elo import rate_1vs1, setup, adjust_1vs1
import itertools as it

import networkx as nx
import axelrod as axl
from axelrod.strategy_transformers import *
from axelrod.graph import Graph
import csv
import pandas as pd
from pandas.plotting import parallel_coordinates
import seaborn as sns
import numpy as np
import colorlover as cl
import scipy as sci
from datetime import date as dtdate


# # Preliminary Tournament

# ## Setup

# In[ ]:


## Setup the game
strategies = (axl.Adaptive(), axl.Cooperator(), axl.Willing(), axl.Bully(), axl.FirmButFair(), axl.Grudger(), axl.Handshake(), axl.HardGoByMajority(), axl.Desperate(), axl.WinStayLoseShift(), axl.Random(), axl.GoByMajority(), axl.TitForTat(), axl.TitFor2Tats(), axl.TwoTitsForTat())
print("#Strategies =",len(strategies))

## Prisoners Dilemma Payoffs
# default payoffs
pdilemma = axl.game.Game()
print(pdilemma)


# ### Tournament

# In[ ]:


## Set seeding
axl.seed(0)

## Define tournament parameters
turns = 50
repetitions= 10

## Run the tournament
prelimTour = axl.Tournament(strategies, game=pdilemma, turns=turns, repetitions=repetitions)
prelimResults = prelimTour.play(processes=1, filename="preliminary_tournament.csv")

## Calculate Elo based on win results. Initial seeding rate of 1200 for all
prelimElo = EloScoring(strategies, "preliminary_tournament.csv", repetitions=repetitions, init_rate=1200.0, index_by_strat=True)
prelimWin = WinScoring(strategies, "preliminary_tournament.csv", repetitions=repetitions, index_by_strat=True)

## Axelrod library plots.
prelimPlots = axl.Plot(prelimResults)


# #### Plots and Tables

# In[ ]:


## Show boxplot with payoffs
_, ax = plt.subplots()

ax.set_ylabel('Payoff', fontsize=15)
ax.set_xlabel('Strategy', fontsize=15)

p = prelimPlots.boxplot(ax=ax)
#p.show()

plt.savefig("prelim_boxplot.pdf", format="PDF", bbox_inches='tight', dpi=300)


# In[ ]:


## Heatmap of the payoff matrix
Heatmap(strategies, prelimResults.payoff_matrix, dist_name="prelim_payoff_matrix")


# In[ ]:


## Heatmap of the payoff matrix given by Elo score
Heatmap(strategies, prelimElo.payoff_matrix, dist_name="prelim_payoff_elo")


# In[ ]:


## Heatmap of the normalised cooperation rate
Heatmap(strategies, prelimResults.normalised_cooperation, dist_name="prelim_cooperation")


# In[ ]:


## Table of the cooperation rating per strategy
prelimCoop = pd.Series(prelimResults.cooperating_rating, index=strategies)

print(prelimCoop.sort_values(ascending=False))
print("Mean:", prelimCoop.mean())
print("Median:", prelimCoop.median())


# In[ ]:


## Comparisson of Rankings
ranks = np.arange(1,len(strategies)+1,1)

# Table of ranking by PD score
rankedAxel = pd.Series(ranks, index=prelimResults.ranked_names, name="Axel Rank")
#print(rankedAxel)

# Table of ranking by Elo rate
rankedElo = pd.Series(ranks, index=prelimElo.ranked_names, name="Elo Rank")
#print(rankedElo)

rankedAll = pd.concat([rankedAxel, rankedElo], axis=1).sort_values(by="Axel Rank")
rankedAll['Delta'] = rankedAll.diff(axis=1).iloc[::,-1].mul(-1)
print(rankedAll)
print("Mean:", rankedAll['Delta'].abs().mean())
print("Median:", rankedAll['Delta'].abs().median())


# In[ ]:


## Linear Regression
xx = rankedAll['Axel Rank']
xx_name = "Axel Rank"

yy = rankedAll['Delta']
yy_name = "\u0394" #Delta

RegPlot(xx, yy, xx_name, yy_name, dist_name="Rank_vs_Delta")


# In[ ]:


# Plot
Parallel(rankedAxel, rankedElo, "Axel Rank", "Elo Rank", dist_name="Axel_vs_Elo")


# In[ ]:


## Heatmap of the Win matrix
Heatmap(strategies, prelimWin.win_matrix, dist_name="prelim_winmatrix")


# In[ ]:


## Chord diagram
strat = ('Adaptive', 'Cooperator', 'Willing', 'Bully', 'FirmButFair', 'Grudger', 'HS', 'HGBMajority', 'Desperate', 'WSLS', 'Random', 'GBMajority', 'TFT', 'TF2T', '2TFT')
X = np.array(prelimResults.payoff_matrix).T

ChordDiagram(X, strat, dist_name="prelim_axel")


# ### Moran

# In[ ]:


## Generate Multiple Moran Processes
moranCounter = pd.Series(np.repeat(0,len(strategies)), index=[str(s) for s in strategies])
for i in np.arange(200):
    moranSeed = np.random.randint(1000000)
    axl.seed(moranSeed)
    prelimMoran = axl.MoranProcess(strategies, game=pdilemma, turns=50)
    moranResults = prelimMoran.play()
    #print("-Step:", i, "-Seed:", moranSeed, "-Winner:", prelimMoran.winning_strategy_name)
    moranCounter.loc[prelimMoran.winning_strategy_name] += 1


# In[ ]:


## Moran rankin by wins
moranCounter.sort_values(ascending=False)


# # Evolving Tournament

# ## Setup

# In[ ]:


## Setup the game
strategies = (axl.Adaptive(), axl.Cooperator(), axl.Willing(), axl.Bully(), axl.FirmButFair(), axl.Grudger(), axl.Handshake(), axl.HardGoByMajority(), axl.Desperate(), axl.WinStayLoseShift(), axl.Random(), axl.GoByMajority(), axl.TitForTat(), axl.TitFor2Tats(), axl.TwoTitsForTat())
## Alternative
#strategies = (axl.Cooperator(), axl.Defector())
print("#Strategies =",len(strategies))

## Strategies multiples
multi = 35

## Prisoners Dilemma Payoffs
pdilemma = axl.game.Game()
print(pdilemma)

## Turns and Reps
Turn = 50
Reps = 10

## Evolution stages
stages = 20

## Number of networks
nets = 10

## Generate Seeds
# Either, for random seeds
#graphSeeds = np.random.randint(1000000, size=nets)
#axelSeeds = np.random.randint(1000000, size=nets)
# Or, for predetermined seeds
graphSeeds = np.array([311241, 582907, 392722,  34510, 994611, 479412, 159459, 537154, 161788, 742156])
axelSeeds = np.array([ 99816,  52652, 401970, 218664, 928531, 261145, 313441, 155399, 696413,  59651])

print("Graph Seeds:", graphSeeds)
print("Axel Seeds:", axelSeeds)

##
date = str(dtdate.today())
print(date)


# ### Tournament

# In[ ]:


## Tournament
tourAxelCounter = pd.DataFrame({'1': np.repeat(0,stages+1)}, index=[str(s) for s in np.arange(stages+1)])
tourAxelPopCounter = pd.DataFrame(np.repeat(0, len(strategies)*(stages+1)).reshape(len(strategies), stages+1), index=[str(s) for s in strategies], columns=[str(s) for s in np.arange(0,stages+1,1)])

# Network info
networksInfo = pd.DataFrame({'GraphSeed': np.repeat(0,nets), 'AxelSeed': np.repeat(0,nets), "<k>": np.repeat(0,nets)}, index=[str(n) for n in np.arange(1,nets+1,1)])

# Rates and Probability
# MANUALLY CHANGE VALUES
rate = "Elo" # Values: "Axel", or "Elo"
prob = "Det" # Values: "Prob", "Stoch", or "Det"

# Run
for i in np.arange(1,nets+1,1):
    print("Network: " + str(i))
    # Seeding
    graphSeed = int(graphSeeds[i-1])
    axelSeed = int(axelSeeds[i-1])
    
    # Setup the tournament
    tourAxel = EvolvingTournament(strategies, multiples=multi, evolution_stages=stages, seed=graphSeed)
    # Save the network info
    networksInfo.loc[str(i)] = np.array([graphSeed, axelSeed, tourAxel.avg_k])
    
    # Run
    tourAxel.play_Tournament(rate_type=rate, evolution_type=prob, game=pdilemma, turns=Turn, repetitions=Reps, elo_params=[10,200], axel_seed=axelSeed)
    # Save the tournament info
    tourAxelCounter[str(i)] = tourAxel.data_Cooperation.matrix.mean()
    tourAxelPopCounter = tourAxelPopCounter.add(tourAxel.data_Population.matrix)


# In[ ]:


#networksInfo


# #### Plots

# In[ ]:


## Cooperation Rates
Spaghetti(tourAxelCounter, dist_name=rate + "_" + prob + "_" + str(nets) + "nets_" + date)


# In[ ]:


#tourAxelPopCounter


# In[ ]:


## Save the final population for later comparison
tourAxelPopCounter.iloc[:,-1].divide(len(strategies)*multi*nets).to_csv("finalPopulation_" + rate + "_" + prob + "_" + str(nets) + "nets_" + date + ".csv", header=False)


# In[ ]:


## Fraction Population Plot
total_pop = len(strategies)*multi*nets

StackedPercent(tourAxelPopCounter, total_pop, dist_name=rate + "_" + prob + "_" + str(nets) + "nets_" + date)


# ### Network Chart

# In[ ]:


## Network chart
nwchart = EvolvingTournament(strategies, multiples=multi, evolution_stages=stages, seed=0)
GGraphPlot(nwchart.graph, nwchart.data_Strategies.get_column('0').to_numpy(), colors = nwchart.colors_Colorized)


# # Comparing

# In[ ]:


## LollipopPlot

# Preliminary Tournament
print(rankedAxel)

# Evolving Tournament
right_file = "finalPopulation_Axel_Det_10nets.csv"
right_name = "C"

mid_file = "finalPopulation_Axel_Stoch_10nets.csv"
mid_name = "B"

left_file = "finalPopulation_Axel_Prob_10nets.csv"
left_name = "A"

sns.set_style("darkgrid")
LollipopPlot(left_file, mid_file, right_file, left_name, mid_name, right_name, rankedAxel, dist_name="Axel_Det_vs_Axel_Prob_vs_Axel_Stoch")


# In[ ]:




