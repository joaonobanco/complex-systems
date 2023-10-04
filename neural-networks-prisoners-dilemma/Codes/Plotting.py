#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

import networkx as nx
import pandas as pd

from matplotlib_chord import *

from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
from pandas.plotting import parallel_coordinates

import plotly.graph_objects as go




## Colour coding
cccol15 = [(0.1015625, 0.90234375, 0.70703125),
      (0.765625, 0.296875, 0.70703125),
      (0.00390625, 0.12109375, 0.90234375),
      (0.73828125, 0.61328125, 0.15625),
      (0.04296875, 0.4921875, 0.80859375),
      (0.43359375, 0.8828125, 0.296875),
      (0.7265625, 0.0625, 0.0625),
      (0.0625, 0.8046875, 0.1015625),
      (0.265625, 0.0390625, 0.953125),
      (0.0234375, 0.2578125, 0.2578125),
      (1.0, 0.5, 0.5),
      (0.1015625, 0.2578125, 0.00390625),
      (0.3984375, 0.61328125, 0.859375),
      (0.00390625, 0.0234375, 0.21875),
      (0.5, 1.0, 1.0)]

cccol14 = [(0.1015625, 0.90234375, 0.70703125),
      (0.765625, 0.296875, 0.70703125),
      (0.00390625, 0.12109375, 0.90234375),
      (0.73828125, 0.61328125, 0.15625),
      (0.04296875, 0.4921875, 0.80859375),
      (0.43359375, 0.8828125, 0.296875),
      (0.0625, 0.8046875, 0.1015625),
      (0.265625, 0.0390625, 0.953125),
      (0.0234375, 0.2578125, 0.2578125),
      (1.0, 0.5, 0.5),
      (0.1015625, 0.2578125, 0.00390625),
      (0.3984375, 0.61328125, 0.859375),
      (0.00390625, 0.0234375, 0.21875),
      (0.5, 1.0, 1.0)]

cccol2 = [(0.765625, 0.296875, 0.70703125),
      (0.00390625, 0.12109375, 0.90234375)]

cocol15 = [(0.1015625, 0.90234375, 0.70703125),
      (0.73828125, 0.61328125, 0.15625),
      (0.765625, 0.296875, 0.70703125),
      (0.265625, 0.0390625, 0.953125),
      (0.04296875, 0.4921875, 0.80859375),
      (0.43359375, 0.8828125, 0.296875),
      (0.7265625, 0.0625, 0.0625),
      (0.0625, 0.8046875, 0.1015625),
      (1.0, 0.5, 0.5),
      (0.1015625, 0.2578125, 0.00390625),
      (0.00390625, 0.0234375, 0.21875),
      (0.3984375, 0.61328125, 0.859375),
      (0.5, 1.0, 1.0),
      (0.00390625, 0.12109375, 0.90234375),
      (0.0234375, 0.2578125, 0.2578125)]

cocol14 = [(0.1015625, 0.90234375, 0.70703125),
      (0.73828125, 0.61328125, 0.15625),
      (0.765625, 0.296875, 0.70703125),
      (0.265625, 0.0390625, 0.953125),
      (0.04296875, 0.4921875, 0.80859375),
      (0.43359375, 0.8828125, 0.296875),
      (0.0625, 0.8046875, 0.1015625),
      (1.0, 0.5, 0.5),
      (0.1015625, 0.2578125, 0.00390625),
      (0.00390625, 0.0234375, 0.21875),
      (0.3984375, 0.61328125, 0.859375),
      (0.5, 1.0, 1.0),
      (0.00390625, 0.12109375, 0.90234375),
      (0.0234375, 0.2578125, 0.2578125)]

cocol2 = [(0.73828125, 0.61328125, 0.15625),
      (0.765625, 0.296875, 0.70703125)]

cc = cccol15
co = cocol15


def Heatmap(players, matrix, xstep=1, dist_name=""):
    
    N = len(players)
    fs = 18
    
    fig, ax = plt.subplots()


    labels = [str(i) for i in players]
    ticks = np.arange(0,N,xstep)

    im = ax.matshow(matrix, cmap='viridis')
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', rotation=90, labelsize=fs)

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis='y', labelsize=fs)


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.05)

    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=fs)

    fig.set_size_inches(12,12)

    plt.savefig("heatmap_" + dist_name + ".pdf", format="PDF", bbox_inches='tight', dpi=300)
    plt.show()



def ChordDiagram(flow_matrix, player_index, dist_name=""): 
    
    fig, ax = plt.subplots()

    ax.axis('off')

    nodePos = chordDiagram(flow_matrix, ax=ax, colors=cc)
    prop = dict(fontsize=22*0.8, ha='center', va='center')
    nodes = player_index
    for i in np.arange(len(player_index)):
        ax.text(nodePos[i][0], nodePos[i][1], nodes[i], rotation=nodePos[i][2], **prop)
        
    fig.set_size_inches(12,12)
    
    plt.savefig("chord_" + dist_name + ".pdf", format="PDF", bbox_inches='tight', dpi=300)



def MeanPlot(means, dist_name=""):
    
    fig, ax = plt.subplots()
        
    ax.plot(means)
    
    ax.tick_params(labelsize=15)
    
    ax.set_xticks(np.arange(len(means)))
    ax.set_xlabel("Evolution Stage", fontsize=15)

    ax.set_yticks(np.arange(0,1.1,0.1))
    ax.set_ylim(0,1)
    ax.set_ylabel("Cooperation Rate", fontsize=15)
    
    fig.set_size_inches(8,6)
    
    plt.savefig("mean_" + dist_name + ".pdf", format="PDF", bbox_inches='tight', dpi=300)
    plt.show()

        
def GGraphPlot(graph, players, colors, dist_name=""):
    G = graph
    players_list = players
    cc = colors
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Rainbow',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=1,
                title='Strategies',
                xanchor='left',
                titleside='right'
            ),
            line_width=2)
    )

    
    node_adjacencies = []
    node_text = []
    #for node, adjacencies in enumerate(G.adjacency()):
    #    node_adjacencies.append()
    #    node_text.append('# of connections: '+str(len(adjacencies[1])))
    
    for node in G.nodes():
        node_adjacencies.append(cc[str(players_list[node])])
        node_text.append("node:" + str(node) + " strat:" + str(players_list[node]))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title="",
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    #fig.write_image("network_chart_round_" + str(tour_round) + dist_name + ".pdf")
    fig.show()


def GGraphPlot2(G):
    #graphSeed = 0
    #G = nx.random_regular_graph(4, len(players), seed=graphSeed)
    #G = nx.fast_gnp_random_graph(len(players), 0.3, directed=False, seed=graphSeed)

    #pos = nx.spring_layout(G)
    #pos = nx.spectral_layout(G)
    #pos = nx.kamada_kawai_layout(G)
    pos = nx.circular_layout(G)
    #pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size = 100)
    nx.draw_networkx_edges(G, pos, width=1, alpha = 0.5)
    nx.draw

    plt.savefig("randomGraph.pdf", format="PDF", bbox_inches='tight')
    plt.show()
    
    #pos = nx.circular_layout(G)
    pos = nx.shell_layout(G)
    #nx.draw_networkx_nodes(G, pos, node_size = 100)
    #nx.draw_networkx_edges(G, pos, width=1, alpha = 0.5)
    nx.draw(G, node_color=cc, pos=pos, node_size=500, width=1, with_labels=True)
    plt.savefig("randomGraph.pdf", format="PDF", bbox_inches='tight')
    plt.show()


def RegPlot(X, Y, X_name, Y_name, dist_name=""):
    
    plt.rcParams["figure.figsize"] = [10,6]
    
    pp = sns.regplot(x=X, y=Y)
    
    pp.tick_params(labelsize=15)
    
    pp.set_xticks([i for i in np.arange(1,len(X)+1,1)])
    pp.set_xticklabels([str(i) for i in np.arange(1,len(X)+1,1)])
    
    pp.set_xlabel(X_name, fontsize=18)
    pp.set_ylabel(Y_name, fontsize=18)
    
    #plt.legend(loc='center left', shadow=True, bbox_to_anchor=(1.1, 0.5), ncol=1, fontsize=15, title="", title_fontsize=18)
    
    plt.savefig("regplot_" + dist_name + ".pdf", format="PDF", bbox_inches='tight', dpi=300)
    plt.show()

    
def Parallel(rank_left, rank_right, name_left, name_right, dist_name=""):
    
    df = pd.concat([rank_left, rank_right], axis=1).reset_index()
    
    fig, ax = plt.subplots()
    ylabels = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th", "13th", "14th", "15th"]
    yticks = [i for i in range(1,16,1)]

    ax.set_yticks(yticks)
    ax.set_ylim(1, 15)
    ax.set_yticklabels(ylabels, fontsize=15)
    ax.invert_yaxis()
    ax.set_xticklabels([name_left, name_right], fontsize=18)

    ax2=ax.twinx()
    ax2.set_yticks(yticks)
    ax2.set_ylim(1, 15)
    ax2.set_yticklabels(ylabels, fontsize=15)
    ax2.invert_yaxis()

    ax2.grid(visible=False)

    parallel_coordinates(df, 'index', cols=[name_left, name_right], colors=co, ax=ax2, linewidth=2)
    
    leg = ax2.legend(loc='center left', shadow=True, bbox_to_anchor=(1.2, 0.5), ncol=1, fontsize=15, title="Strategies", title_fontsize=18)
    fig.set_size_inches(8, 6)
    
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    fig.tight_layout()
    plt.savefig("parallel_" + dist_name + ".pdf", format="PDF", bbox_inches='tight', dpi=300)
    plt.show()
    
    
def Spaghetti(data, dist_name=""):
    num_stages = len(data.index)
    
    fig, ax = plt.subplots()

    xticks = [i for i in range(0,num_stages+1,1)]
    xlabels = [str(i) for i in range(0,num_stages+1,1)]

    ax.set_xticks(xticks)
    
    ax.set_xlim(0, num_stages-1)
    ax.set_ylim(0, 1)
    
    ax.set_xlabel("Evolution Stage", fontsize=18)
    ax.set_ylabel("Mean Cooperation Rate", fontsize=18, rotation=90)

    for col in data.columns.to_numpy():
        ax.plot(data.index.to_numpy(), data[col], color='grey', linewidth=1, alpha=0.9)
    
    ax.plot(data.index.to_numpy(), data.mean(axis=1), color='red', linewidth=2, alpha=0.9)
    
    #ax.legend(loc='center left', shadow=True, bbox_to_anchor=(1.2, 0.5), ncol=1, fontsize=12)
    fig.set_size_inches(10, 6)

    fig.tight_layout()
    plt.savefig("spaghetti_" + dist_name + ".pdf", format="PDF", bbox_inches='tight')
    plt.show()
    
    
def StackedPercent(data, total, ymax=0.1, ystep=0.2, xmax=1, xstep=1, dist_name=""):
            
    ax = data.T.divide(total).plot.area(colors=cc, figsize=(10,6))

    ax.tick_params(labelsize=15)
    ax.set_xticks([i for i in np.arange(0,len(data.columns)+xmax,xstep)])
    ax.set_xticklabels([str(i) for i in np.arange(0,len(data.columns)+xmax,xstep)])
    ax.set_xlim(0,len(data.columns)-1)
    ax.set_xlabel("Evolution Stage", fontsize=18)
    
    ax.set_yticks([i for i in np.arange(0,1+ymax,ystep)])
    ax.set_ylim(0,1)
    ax.set_ylabel("Fraction of Population", fontsize=18)

    ax.legend(loc='center left', shadow=True, bbox_to_anchor=(1.1, 0.5), ncol=1, fontsize=15, title="Strategies", title_fontsize=18)
    
    plt.savefig("stackedpercent_" + dist_name + ".pdf", format="PDF", bbox_inches='tight', dpi=300)
    plt.show()
    
    
    
def LollipopPlot(file_left, file_mid, file_right, name_left, name_mid, name_right, rank_by, dist_name=""):

    # Create a dataframe
    values1 = pd.read_csv(file_left, index_col=0, header=None, names=[name_left])
    values2 = pd.read_csv(file_mid, index_col=0, header=None, names=[name_mid])
    values3 = pd.read_csv(file_right, index_col=0, header=None, names=[name_right])
    values4 = rank_by
        

    un_df = pd.concat([values1, values2, values3, values4], axis=1).reset_index()
    #print(un_df)

    # Reorder it following the values of the first value:
    df = un_df.sort_values(by="Axel Rank")
    #print(df)
    my_range=np.arange(1,len(un_df.index)+1,1)

    # The vertical plot is made using the hline function
    plt.rcParams["figure.figsize"] = [10,6]
    # I load the seaborn library only to benefit the nice looking feature
    plt.hlines(y=my_range, xmin=df[name_left], xmax=df[name_right], color='grey', alpha=0.4)
    plt.hlines(y=my_range, xmin=df[name_left], xmax=df[name_mid], color='grey', alpha=0.4)
    plt.hlines(y=my_range, xmin=df[name_mid], xmax=df[name_right], color='grey', alpha=0.4)
    plt.scatter(df[name_left], my_range, color='steelblue', alpha=1, label=name_left, marker="o")
    plt.scatter(df[name_mid], my_range, color='mediumorchid', alpha=1 , label=name_mid, marker="D")
    plt.scatter(df[name_right], my_range, color='limegreen', alpha=1 , label=name_right, marker="s")
    plt.legend(loc='center left', shadow=True, bbox_to_anchor=(1., 0.5), ncol=1, fontsize=15, title="Probability", title_fontsize=18)


    # Add title and axis names
    plt.yticks(my_range, df['index'].to_numpy(), fontsize=15)
    plt.xlabel('Fraction of Population', fontsize=18)
    plt.ylabel('Strategy', fontsize=18)
    plt.ylim(my_range[-1]+0.5,my_range[0]-0.5)
        
    
    #plt.figure(figsize=(10,6))
        
    plt.savefig("lollipop_" + dist_name + ".pdf", format="PDF", bbox_inches='tight', dpi=300)
    plt.show()