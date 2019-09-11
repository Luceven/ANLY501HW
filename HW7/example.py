# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:48:34 2018

@author: Yunjia
"""

##igraph in Python
##Gates 2018
## http://igraph.org/python/
## http://igraph.org/python/doc/tutorial/tutorial.html
##On the command line in Windows
##anaconda search -t conda python-igraph
##Then, from these choices - choose one
##for Win64 and ideally also for Win32
##and for py35_...
##Then...
##install - c vtraag python-igraph=0.7.1.post7

## network data to play with 
## http://www-personal.umich.edu/~mejn/netdata/

### VIEW THIS VIDEO FOR networkx ##########################
## https://www.youtube.com/watch?v=1ErL1z_lKd8&t=204s

import matplotlib.pyplot as plt
from igraph import *
import igraph
import networkx as nx

######## igraph ################
print("hello")
print (igraph.__version__)
g = igraph.Graph(directed=True)
print(g)
g.add_vertices(3)
g.add_edges([(0,1), (1,2)])
print(g)
print(g.degree())
print(g.degree(mode="in"))
print(g.degree(mode="out"))
print(g.edge_betweenness())


################ networkx #################
## https://networkx.github.io/documentation/stable/tutorial.html
NetxG = nx.Graph()
NetxG.add_nodes_from([1, 2, 3, 4, 5])
NetxG.add_edge(2, 3)
NetxG.add_edges_from([(1, 2), (4, 5)])
print(list(NetxG.nodes))
print(list(NetxG.edges))
nx.draw(NetxG)
## weighted
## SomeG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
## read from files options....
#nx.read_adjlist()
#nx.read_edgelist()
#nx.read_gml()
#nx.read_weighted_edgelist()