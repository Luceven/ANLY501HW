# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 22:25:37 2018

@author: Yunjia
"""

from igraph import *
import cairo

### create a simple graph ###
def create_simple_graph():
	g = Graph(directed=True)
	g.add_vertices(6)
	g.add_edges([(0,1), (1,2), (2,0), (2,3), (3,4), (4,5), (5,3)])
	plot(g)


### read in data to plot ###
def plot_graph():
	g = Graph.Read_Ncol('network.txt')
	plot(g)

def main():
	create_simple_graph()
	plot_graph()

main()