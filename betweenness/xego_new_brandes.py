import networkx as net
from operator import itemgetter, attrgetter
import math
from itertools import combinations, permutations
import Bio.Cluster
import matplotlib.pyplot as plt
import scipy.stats
import random
import numpy as np
import re
import codecs
import csv
import networkx.algorithms as algo
import math
import sys
import time
from scipy import stats
import matplotlib

matplotlib.use('pgf')

elapsed_time1 = 0.0
elapsed_time2 = 0.0
	
###########################################################################################################
def my_ego_betweenness_centrality(G, center, fe, node2Index, index2Node):
	nodes = G.nodes()
	nodes.remove(center)
	
	lenOfFirstNeighbors = len(fe)
	table = dict.fromkeys(index2Node.keys(), 0.0)
	for i in index2Node.keys():
		delta=dict.fromkeys(index2Node.keys(), 0.0)
		table[i] = delta
		
	betweenness = 0.0
	
	for i in range(1, lenOfFirstNeighbors+1):
		N_i = G[index2Node[i]].keys()
		for j in range(i+1, lenOfFirstNeighbors+1):
			N_j = G[index2Node[j]].keys()
			table[i][j] = dependency1(index2Node[j], N_i, N_j)
			betweenness += table[i][j]

	betweenness = my_rescale(betweenness, len(G))
	
	return betweenness

###########################################################################################################
def my_xEgo_betweenness_centrality(G, center, firstNeighbors, secondNeighbors, node2Index, index2Node):
	nodes = G.nodes()
	nodes.remove(center)
	
	lenOfFirstAndSecondNeigbors = len(nodes)
	lenOfFirstNeighbors = len(firstNeighbors)
	
	table = dict.fromkeys(index2Node.keys(), 0.0)
	for i in index2Node.keys():
		delta=dict.fromkeys(index2Node.keys(), 0.0)
		table[i] = delta
					
	betweenness = 0.0
	
	for i in range(1, lenOfFirstNeighbors+1):
		N_i = G[index2Node[i]].keys()
		for j in range(i+1, lenOfFirstNeighbors+1):
			N_j = G[index2Node[j]].keys()
			table[i][j] = dependency1(index2Node[j], N_i, N_j)
			betweenness += table[i][j]
		for j in range(lenOfFirstNeighbors+1, lenOfFirstAndSecondNeigbors+1):
			N_j = G[index2Node[j]].keys()
			table[i][j] = dependency2(i, N_j, node2Index, index2Node, table)
			betweenness += table[i][j]
				
	for i in range(lenOfFirstNeighbors+1, lenOfFirstAndSecondNeigbors+1):
		for j in range(i+1, lenOfFirstAndSecondNeigbors+1):
			N_j = G[index2Node[j]].keys()
			table[i][j] = dependency2(i, N_j, node2Index, index2Node, table)
			betweenness += table[i][j]

	betweenness = my_rescale(betweenness, len(G))
	
	return betweenness	

def dependency1(j_node, N_i, N_j):
	if j_node in N_i:
		return 0.0
	else:
		return 1.0 / len(set(N_i) & set(N_j))

def dependency2(i, N_j, node2Index, index2Node, table):
	values = []	
	for neighbor in N_j:
		f = 0.0
		if (i < node2Index[neighbor]):
			f = table[i][node2Index[neighbor]]
		else:
			f = table[node2Index[neighbor]][i]	
		if (f == 0.0):
			return 0.0
		else:
			values.append(f)
	return stats.hmean(values)	
			
def my_rescale(betweenness,n):
	if n <= 2:
		scale=None  # no normalization b=0 for all nodes
	else:
		scale=2.0/((n-1)*(n-2))
  
	if scale is not None:
		betweenness *= scale
	return betweenness
	
###############################################################################	
def brandes_betweenness_centrality(G):
	betweenness=dict.fromkeys(G,0.0) # b[v]=0 for v in G
	nodes = G
	for s in nodes:
		S, P, sigma = _single_source_shortest_path_basic(G,s)
		betweenness=_accumulate_basic(betweenness,S,P,sigma,s)
	betweenness=_rescale(betweenness, len(G))
	return betweenness
	
def _single_source_shortest_path_basic(G,s):
	S=[]
	P={}
	for v in G:
		P[v]=[]
	sigma=dict.fromkeys(G,0.0)    # sigma[v]=0 for v in G
	D={}
	sigma[s]=1.0
	D[s]=0
	Q=[s]
	while Q:   # use BFS to find shortest paths
		v=Q.pop(0)
		S.append(v)
		Dv=D[v]
		sigmav=sigma[v]
		for w in G[v]:
			if w not in D:
				Q.append(w)
				D[w]=Dv+1
			if D[w]==Dv+1:   # this is a shortest path, count paths
				sigma[w] += sigmav
				P[w].append(v) # predecessors 
	return S,P,sigma
	
def _accumulate_basic(betweenness,S,P,sigma,s):
	delta=dict.fromkeys(S, 0)
	while S:
		w=S.pop()
		coeff=(1.0+delta[w])/sigma[w]
		for v in P[w]:
			delta[v] += sigma[v]*coeff
		if w != s:
			betweenness[w]+=delta[w]
	return betweenness

def _rescale(betweenness,n):
	if n <= 2:
		scale=None  # no normalization b=0 for all nodes
	else:
		scale=1.0/((n-1)*(n-2))
  
	if scale is not None:
		for v in betweenness:
			betweenness[v] *= scale
	return betweenness
	

def isNumber(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

def _rank(m):
    (ivec, svec) = zip(*sorted(list(enumerate(m)), key=itemgetter(1)))
    sumranks = 0
    dupcount = 0
    newlist = [0] * len(m)
    for i in range(len(m)):
        sumranks += i
        dupcount += 1
        if i == len(m) - 1 or svec[i] != svec[i + 1]:
            averank = sumranks / float(dupcount) + 1
            for j in range(i - dupcount + 1, i + 1):
                newlist[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newlist
    
def spearman_rho(m, n):
    """ 
    return Spearman's rho; based off stats.py 

    >>> x = [2, 8, 5, 4, 2, 6, 1, 4, 5, 7, 4]
    >>> y = [3, 9, 4, 3, 1, 7, 2, 5, 6, 8, 3]
    >>> spearman_rho(x, y)
    0.9363636363636364
    """
    if len(m) != len(n):
        raise ValueError, 'Iterables (m, n) must be the same length'
    dsq = sum([(mi - ni) ** 2 for (mi, ni) in zip(_rank(m), _rank(n))])
    return 1. - 6. * dsq / float(len(m) * (len(n) ** 2 - 1.))

def frange5(start, end = None, inc = None):
  """A range function, that does accept float increments..."""
  import math

  if end == None:
      end = start + 0.0
      start = 0.0
  else: start += 0.0 # force it to be a float

  if inc == None:
      inc = 1.0
  count = int(math.ceil((end - start) / inc))

  L = [None,] * count

  L[0] = start
  for i in xrange(1,count):
      L[i] = L[i-1] + inc
  return L

def createRefGraph():
	g = net.Graph()
	g.add_edges_from([(0,3), (3,1), (2,4), (4,3), (4,5), (3,5), (6,4), (4,7), (5,7), (7,8), (7,9), (8,9), (10,8), (10,9), (9,11), (10,11), (11,12)])
	return g

def getNeighborGraph(g, center):
	g2 = net.Graph()
	neighbors = g.neighbors(center)
	for neighbor in neighbors:
		g2.add_edge(center, neighbor)
	return g2

def getEgoGraph(g, center):
	g2 = net.Graph()
	node2Index = {}
	index2Node = {}
	g2.add_node(center)
	firstNeighbors = g.neighbors(center)

	node2Index[center] = 0
	index2Node[0] = center

	i = 1;
	for neighbor in firstNeighbors:
		node2Index[neighbor] = i
		index2Node[i] = neighbor
		i += 1;

	for neighbor in firstNeighbors:
		g2.add_edge(neighbor, center)
		NeighborsOfNeighbors = g.neighbors(neighbor)
		for nneighbor in NeighborsOfNeighbors:
			if (nneighbor == center):
				continue
			if nneighbor in firstNeighbors:
				g2.add_edge(neighbor, nneighbor)
	return g2, firstNeighbors, node2Index, index2Node

def getExpandedEgoGraph(g, center):
	g2 = net.Graph()
	g2.add_node(center)
	firstNeighbors = g.neighbors(center)
	
	secondNeighbors = []
	for neighbor in firstNeighbors:
		g2.add_edge(neighbor, center)
		secondNeighborsOfNode = g.neighbors(neighbor)
		for nneighbor in secondNeighborsOfNode:
			if (nneighbor == center):
				continue
			if nneighbor in firstNeighbors:
				g2.add_edge(neighbor, nneighbor)
			else:
				g2.add_edge(neighbor, nneighbor)
				secondNeighbors.append(nneighbor)
	secondNeighbors = list(set(secondNeighbors))
	return g2, firstNeighbors, secondNeighbors

def getExpandedEgoGraph2(g, center):
	g2 = net.Graph()
	node2Index = {}
	index2Node = {}
	g2.add_node(center)
	firstNeighbors = g.neighbors(center)

	node2Index[center] = 0
	index2Node[0] = center
			
	i = 1;
	for neighbor in firstNeighbors:
		node2Index[neighbor] = i
		index2Node[i] = neighbor
		i += 1;

	secondNeighbors = []
	for neighbor in firstNeighbors:
		g2.add_edge(neighbor, center)
		secondNeighborsOfNode = g.neighbors(neighbor)
		for nneighbor in secondNeighborsOfNode:
			if (nneighbor == center):
				continue
			if nneighbor in firstNeighbors:
				g2.add_edge(neighbor, nneighbor)
			else:
				g2.add_edge(neighbor, nneighbor)
				if not (nneighbor in secondNeighbors):
					secondNeighbors.append(nneighbor)
					node2Index[nneighbor] = i
					index2Node[i] = nneighbor					
					i += 1
	return g2, firstNeighbors, secondNeighbors, node2Index, index2Node	

def get_CentralityList(g):
	t_bet_centrality = net.betweenness_centrality(g)
	return t_bet_centrality

##################################################################################################################
def get_Ego_CentralityList(g):
	global elapsed_time1
	elapsed_time1 = 0.0
	numNodes = g.number_of_nodes()
	centrality_map = {}	
	for node in g.nodes():
		egoNet = net.ego_graph(g, node)
		#######
		start_time = time.time()		
		centrality_map[node] = brandes_betweenness_centrality(egoNet).get(node)
		end_time = time.time()
		#######
		elapsed_time1 = elapsed_time1 + (end_time - start_time)		
	return centrality_map
	
def get_Ego_CentralityList_Proposed(g):
	global elapsed_time2
	elapsed_time2 = 0.0
	numNodes = g.number_of_nodes()
	centrality_map = {}	
	for node in g.nodes():
		egoNet, firstNeighbors, node2Index, index2Node = getEgoGraph(g, node)	
		#######
		start_time = time.time()
		centrality_map[node] = my_ego_betweenness_centrality(egoNet, node, firstNeighbors, node2Index, index2Node)
		end_time = time.time()
		#######
		elapsed_time2 = elapsed_time2 + (end_time - start_time)				
	return centrality_map
##################################################################################################################

##################################################################################################################			
def get_XEgo_CentralityList(g):
	global elapsed_time1
	elapsed_time1 = 0.0
	numNodes = g.number_of_nodes()
	centrality_map = {}
	for node in g.nodes():
		xEgoNet, firstNeighbors, secondNeighbors = getExpandedEgoGraph(g, node)
		#######
		start_time = time.time()
		centrality_map[node] = brandes_betweenness_centrality(xEgoNet).get(node)
		end_time = time.time()
		#######
		elapsed_time1 = elapsed_time1 + (end_time - start_time)
	return centrality_map

def get_XEgo_CentralityList_Proposed(g):
	global elapsed_time2
	elapsed_time2 = 0.0
	numNodes = g.number_of_nodes()
	centrality_map = {}	
	for node in g.nodes():
		xEgoNet, firstNeighbors, secondNeighbors, node2Index, index2Node = getExpandedEgoGraph2(g, node)	
		#######
		start_time = time.time()
		centrality_map[node] = my_xEgo_betweenness_centrality(xEgoNet, node, firstNeighbors, secondNeighbors, node2Index, index2Node)
		end_time = time.time()
		#######
		elapsed_time2 = elapsed_time2 + (end_time - start_time)				
	return centrality_map
##################################################################################################################
	
def show(g):
	net.draw(g)
	plt.show()
	plt.close()

def mean(values):
	if len(values) == 0:
		return None
	return sum(values) / len(values)

def standardDeviation(values):
	value_array = np.array(values)
	return value_array.std()
	

def get_contact_weight_map(n, filename):
	f = open(filename, 'r')
	reader = csv.reader(f, delimiter='\t')
	reader.next()
	reader.next()
	
	contact_time_map = {}
	x = 1
	for line in reader:
		if len(line) == 0:
			continue
		id = line[0]
		type = line[1]
		num_incidence_nodes = line[2]
		num_total_contacts = line[6]
		
		if type == '2':
			break
		
		line = reader.next()
		contact_time_list = []
		for y in range(n):
			contact_time_list.append(int(line[y+7]))
		contact_time_map[x] = contact_time_list
		x = x + 1
	f.close()	
	return contact_time_map
	
def get_contact_weight_map2(n, filename):
	f = open(filename, 'r')
	reader = csv.reader(f, delimiter=' ')
	contact_time_map = {}
	for x in range(1, n+1):
		contact_time_map[x] = {}
		for y in range(1, n+1):
			contact_time_map[x][y]=0

	for line in reader:
		if len(line) == 0:
			break
		source = int(line[0])
		target = int(line[1])
		contact_time = int(line[2])
		if ((source >= 1) and (source <= n)) and ((target >= 1) and (target <= n)):
			contact_time_map[source][target] = contact_time_map[source][target] + contact_time
		
	for x in range(1, n+1):
		contact_list_node = []
		for y in range(1, n+1):
			contact_list_node.append(contact_time_map[x][y])
		contact_time_map[x] = contact_list_node
	
	f.close()
	return contact_time_map

def get_adjacency_matrix(n, map, threshold):
	adjacency_map = {}
	
	mean_contact_duration = 0;
	for key, value in map.items():
		mean_contact_duration =  mean_contact_duration + mean(value)
	mean_contact_duration = mean_contact_duration / n
	
	#print mean_contact_duration
	#print mean_contact_duration * threshold
	
	id = 1
	for key, value in map.items():
		adjacency_list = []
		for x in value:
			if x >= mean_contact_duration * threshold:
				adjacency_list.append(id)
			id = id + 1	
		adjacency_map[key] = adjacency_list
		id = 1
		
	return adjacency_map
	
if __name__ == "__main__":
	if len(sys.argv) == 1:
		print "An input value is required"
		sys.exit(1)
	elif not isNumber(sys.argv[1]):
		print "Your input values is not number"
		sys.exit(2)
  
	data_type = sys.argv[1]
	data = None
	num_nodes = 0
	weight_map = None
	
	if data_type == '1':
		data = "infocom05_new.dat"
		num_nodes = 41
		weight_map = get_contact_weight_map(num_nodes, data)
	elif data_type == '2':
		data = "infocom06_new.dat"
		num_nodes = 99
		weight_map = get_contact_weight_map(num_nodes, data)
	elif data_type == '3':
		data = "data_intel.txt"
		num_nodes = 20
		weight_map = get_contact_weight_map2(num_nodes, data)	
	elif data_type == '4':
		data = "data_cambridge.txt"
		num_nodes = 54
		weight_map = get_contact_weight_map2(num_nodes, data)		
	else:
		data = "infocom05_new.dat"
		num_nodes = 41
		weight_map = get_contact_weight_map(num_nodes, data)

	print "Start to analyze the %s trace" % data
	print "The number of nodes: %d" % num_nodes
	print "The number of edges in the nodes' complete graph: %d" % (num_nodes * (num_nodes - 1) / 2)
	
	brandes_ego_elapse_time = []
	proposed_ego_elapse_time = []	

	brandes_xEgo_elapse_time = []
	proposed_xEgo_elapse_time = []	
	
	for x in [8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625]:
		print "Threshold: %5.4f" % x
		adj_map = get_adjacency_matrix(num_nodes, weight_map, x)
		#print adj_map
		g = net.Graph(adj_map)
		density = net.density(g)
		clustering = algo.clustering(g)
		scc_g = algo.connected_component_subgraphs(g)[0]
		diameter = algo.diameter(scc_g)
		
		clustering_coefficient = mean(clustering.values())  
		
		bet = get_CentralityList(g).values()
		myfile = open(data_type + "_" + str(x) + '_global_bet.csv', 'wb')
		wr = csv.writer(myfile)
		wr.writerow(bet)
		myfile.close()
		
		ego_bet_map = get_Ego_CentralityList(g)
		ego_bet = ego_bet_map.values()
		cor_1 = 1 - Bio.Cluster.distancematrix((bet, ego_bet), dist="s")[1][0]
		print "Brandes, ego-global correlation: %5.4f - elapsed_time: %5.4f" % (cor_1, elapsed_time1)
		myfile = open(data_type + "_" + str(x) + '_ego_bet.csv', 'wb')
		wr = csv.writer(myfile)
		wr.writerow(ego_bet)
		myfile.close()
		
		ego_bet_map = get_Ego_CentralityList_Proposed(g)
		ego_bet = ego_bet_map.values()
		cor_1 = 1 - Bio.Cluster.distancematrix((bet, ego_bet), dist="s")[1][0]
		print "Proposed, ego-global correlation: %5.4f - elapsed_time: %5.4f" % (cor_1, elapsed_time2)

		brandes_ego_elapse_time.append(elapsed_time1)
		proposed_ego_elapse_time.append(elapsed_time2)
		
		xego_bet_map = get_XEgo_CentralityList(g)
		xego_bet = xego_bet_map.values()
		cor_2 = 1 - Bio.Cluster.distancematrix((bet, xego_bet), dist="s")[1][0]		
		print "Brandes, xEgo-global correlation: %5.4f - elapsed_time: %5.4f" % (cor_2, elapsed_time1)
		myfile = open(data_type + "_" + str(x) + '_xEgo_bet.csv', 'wb')
		wr = csv.writer(myfile)
		wr.writerow(xego_bet)
		myfile.close()
		
		xego_bet_map = get_XEgo_CentralityList_Proposed(g)
		xego_bet = xego_bet_map.values()		
		cor_2 = 1 - Bio.Cluster.distancematrix((bet, xego_bet), dist="s")[1][0]
		print "Proposed, xEgo-global correlation: %5.4f - elapsed_time: %5.4f" % (cor_2, elapsed_time2)

		brandes_xEgo_elapse_time.append(elapsed_time1)
		proposed_xEgo_elapse_time.append(elapsed_time2)
		print "#####"
	
	print "[brandes_ego_elapse_time]"
	print brandes_ego_elapse_time
	
	print "[proposed_ego_elapse_time]"
	print proposed_ego_elapse_time
	
	print "[brandes_xEgo_elapse_time]"
	print brandes_xEgo_elapse_time
	
	print "[proposed_xEgo_elapse_time]"
	print proposed_xEgo_elapse_time
		
	print "Brandes, ego, cummulated elapsed time: ", sum(brandes_ego_elapse_time)		
	print "Proposed, ego, cummulated elapsed time: ", sum(proposed_ego_elapse_time)
	
	print "Brandes, xEgo, cummulated elapsed time: ", sum(brandes_xEgo_elapse_time)		
	print "Proposed, xEgo, cummulated elapsed time: ", sum(proposed_xEgo_elapse_time)			