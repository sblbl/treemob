"""
 #### #####  ###### ###### #    #  ####  #####  
  #   #    # #      #      ##  ## #    # #    # 
  #   #    # #####  #####  # ## # #    # #####  
  #   #####  #      #      #    # #    # #    # 
  #   #   #  #      #      #    # #    # #    # 
  #   #    # ###### ###### #    #  ####  #####  
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import networkx as nx 
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.readwrite import json_graph
from networkx.generators.trees import prefix_tree
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from networkx.linalg.graphmatrix import adjacency_matrix

import pydot
import graphviz

from edist.ted import standard_ted as ted
from edist.uted import uted

from rdp import rdp

from copy import deepcopy
import pickle
import itertools

def save_tree(obj, filename):
	""" Save tree
		Saves a PrefixTree or SpanningTree object
		in a .pkl file 

		Parameters:
		------------
		obj: PrefixTree or SpanningTree
			The tree to be saved
		filename: string
			The name of the file that will be saved (no extension is needed)
	"""
	with open(filename + '.pkl', 'wb') as outp:  # Overwrites any existing file.
		pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_tree(filename):
	""" Load tree
		Loads a .pkl file containing a PrefixTree or SpanningTree

		Parameters:
		------------
		filename: string
			The name of the file to be opened (no extension is needed) 
	"""
	with open(filename + '.pkl', 'rb') as inp:
		obj = pickle.load(inp)
	return obj

def generate_trajs(df):
	""" Generate trajectories
		From a Pandas dataframe containing the
		locations traversed by a single user,
		returns a list of trajectories

		Parameters:
		------------
		df: pandas.DataFrame
			The table with df.columns containing ['traj', 'cell'], 
			where 'traj' is the ID of the trajectory and 'cell' 
			is the ID of the traversed location

		Returns:
		---------
		trajs: list 
			A list of lists of location IDs, corresponding to the trajectories
			of a user

	"""
	trajs = []
	for i in range(max(df.traj)):
		traj = df.loc[df['traj'] == i]['cell'].values.tolist()
		if len(traj) > 1:
			trajs.append(traj)
	return trajs 

def _create_couples(traj):
	return [*itertools.zip_longest(traj, traj[1:], fillvalue=traj[0])]

def get_link_weights(trajs):
	all_couples = []
	weights = {}
	for traj in trajs:
		all_couples += _create_couples(traj)
		
	unique = list(set(all_couples))
	for u in unique:
		weights[u] = all_couples.count(u)

	return weights

def get_loc_weights(trajs):
	all_trajs = []
	weights = {}
		
	for traj in trajs:
		all_trajs += traj
	
	unique = list(set(all_trajs))

	for u in unique:
		weights[u] = all_trajs.count(u)
		
	return weights

def most_frequent_loc(locs):
	""" Most frequent location

		Parameters:
		-------------
		locs: dict 
			The node weights from which to extract the locations
		
		Returns:
		--------
		loc: [int, string]
			The location which occours more frequently in
			The dataset of trajectories

	"""
	
	idx = list(locs.values()).index(max(list(locs.values())))
	loc = list(locs.keys())[idx]

	return loc

class PrefixTree():
	""" Prefix Tree
		Prefix Tree data structure adapted to personal mobility data

		Parameters:
		------------
		trajectories: list
			The list of trajectories belonging to a user. 
			Each trajectory is a list of location IDs.
		root: string 
			The location ID of the designated root
		nodes_weights: dict 
			(optional) default=None
			A dictionary in form {locationID : weight}
		edges_weights: dict 
			(optional) default=None
			A dictionary in form {(from, to) : weight}. 
			Warning: the edge weight is directed and (from, to) 
			is not equal to (to, from)

	"""
	def __init__(self, trajectories, root, nodes_weights=None, edges_weights=None):
		self.trajs = None
		self.prep_trajs = []
		self.in_trajs = []
		self.out_trajs = []
		
		self.base_tree = None
		self.base_labels = None
		
		self.in_tree = None
		self.in_labels = None
		
		self.out_tree = None
		self.out_labels = None
		
		self.root = root
		self.root_ID = 1
		self.virtual_root_ID = 0
		self.brackets = {}
		self.nodes_weights = nodes_weights
		self.edges_weights = edges_weights
		
		self._generate(trajectories)
			
		if self.nodes_weights != None:
			self.add_nodes_weigths(tree='base_tree', weight_name='weight', default_val=1)
			self.add_nodes_weigths(tree='in_tree', weight_name='weight', default_val=1)
			self.add_nodes_weigths(tree='out_tree', weight_name='weight', default_val=1)
		if self.edges_weights != None:
			self.add_edges_weights(tree='base_tree', default_val=1)
			self.add_edges_weights(tree='in_tree', default_val=1)
			self.add_edges_weights(tree='out_tree', default_val=1)
		
	def _generate(self, trajectories):
		"""
		Generates the base prefix tree
			Parameters:
			-------------
			trajs: list 
				A list of lists of numerical or str
			root: int/str
				The location that will become the root
		"""
		#self.trajs = deepcopy(trajectories)
		self.trajs = []
		
		for traj in deepcopy(trajectories):
			if self.root in traj:
				traj.insert(0, '^')
				traj.append('$')
				self.trajs.append(traj)
				
				idx = traj.index(self.root)
				
				if idx > 1:
					t = traj[0 : idx+1]
					t.reverse()
					self.prep_trajs.append(t)
					self.in_trajs.append(t)

					t = traj[idx:]
					self.prep_trajs.append(t)
					self.out_trajs.append(t)
		
		# trees creation
		self.base_tree = nx.prefix_tree(self.prep_trajs)
		self.base_tree.remove_node(-1)
		self.base_labels = dict(zip(self.base_tree.nodes(), [self.base_tree.nodes[node]['source'] for node in self.base_tree.nodes()]))    
		self.base_labels[0] = 'root'
		nx.set_node_attributes(self.base_tree, self.base_labels, 'label')
		
		self.in_tree = nx.prefix_tree(self.in_trajs)
		self.in_tree.remove_node(-1)
		self.in_labels = dict(zip(self.in_tree.nodes(), [self.in_tree.nodes[node]['source'] for node in self.in_tree.nodes()]))    
		self.in_labels[0] = 'root'
		nx.set_node_attributes(self.in_tree, self.in_labels, 'label')
		
		self.out_tree = nx.prefix_tree(self.out_trajs)
		self.out_tree.remove_node(-1)
		self.out_labels = dict(zip(self.out_tree.nodes(), [self.out_tree.nodes[node]['source'] for node in self.out_tree.nodes()]))    
		self.out_labels[0] = 'root'
		nx.set_node_attributes(self.out_tree, self.out_labels, 'label')
	
	def show(self, with_labels=True, tree='base_tree'):
		""" Show
			Plots the tree

			Parameters:
			------------
			with_labels: bool
				Wherether to show the locationID in the nodes
			tree: ['base_tree', 'in_tree, out_tree'] default='base_tree'
				The internal tree to show: base_tree contains incoming and outcoming paths

		"""
		if tree == 'base_tree':
			pos = graphviz_layout(self.base_tree, prog='dot', root=0)
			if with_labels == True:
				nx.draw(self.base_tree, pos, with_labels=True, labels=self.base_labels)
			else:
				nx.draw(self.base_tree, pos, with_labels=True)
		elif tree == 'in_tree':
			pos = graphviz_layout(self.in_tree, prog='dot', root=self.root_ID)
			if with_labels == True:
				nx.draw(self.in_tree, pos, with_labels=True, labels=self.in_labels)
			else:
				nx.draw(self.in_tree, pos, with_labels=True)
		elif tree == 'out_tree':
			pos = graphviz_layout(self.out_tree, prog='dot', root=self.root_ID)
			if with_labels == True:
				nx.draw(self.out_tree, pos, with_labels=True, labels=self.out_labels)
			else:
				nx.draw(self.out_tree, pos, with_labels=True)
		else:
			print('please enter a valid tree')
			
		plt.show()
		
	def to_json(self, tree='base_tree'):
		""" To JSON
			Saves the tree in json format

			Parameters:
			------------
			tree: ['base_tree', 'in_tree, out_tree'] default='base_tree'
				The internal tree to show: base_tree contains incoming and outcoming paths

			Returns:
				tree: dict

		"""
		if tree == 'base_tree':
			return nx.node_link_data(self.base_tree)
		elif tree == 'in_tree':
			return nx.node_link_data(self.in_tree)
		elif tree == 'out_tree':
			return nx.node_link_data(self.out_tree)
		else:
			print('please enter a valid tree')
	
	def to_brackets(self, tree='base_tree'):
		""" To brackets
			Saves the tree in brackets format

			Parameters:
			------------
			tree: ['base_tree', 'in_tree, out_tree'] default='base_tree'
				The internal tree to show: base_tree contains incoming and outcoming paths

			Returns:
				tree: dict

		"""
		if tree == 'base_tree':
			root = [node for node in list(self.base_tree.nodes()) if self.base_tree.in_degree(node)==0][0]
			return json_graph.tree_data(self.base_tree, root=root)
		elif tree == 'in_tree':
			root = [node for node in list(self.in_tree.nodes()) if self.in_tree.in_degree(node)==0][0] 
			return json_graph.tree_data(self.in_tree, root=root)
		elif tree == 'out_tree':
			root = [node for node in list(self.out_tree.nodes()) if self.out_tree.in_degree(node)==0][0] 
			return json_graph.tree_data(self.out_tree, root=root)
		else:
			print('please enter a valid tree')
		
	def lookup_matrix(self, tree='base_tree'):
		""" Lookup Matrix
			Saves the tree in the format of a lookup matrix

			Parameters:
			------------
			tree: ['base_tree', 'in_tree, out_tree'] default='base_tree'
				The internal tree to show: base_tree contains incoming and outcoming paths

			Returns:
				tree: numpy.array

		"""
		if tree == 'base_tree':
			return np.array(nx.adjacency_matrix(self.base_tree, weight='weight').todense())
		elif tree == 'in_tree':
			return np.array(nx.adjacency_matrix(self.in_tree, weight='weight').todense())
		elif tree == 'out_tree':
			return np.array(nx.adjacency_matrix(self.out_tree, weight='weight').todense())
		else:
			print('please enter a valid tree')
				
	def add_nodes_weigths(self, tree, weight_name='weight', default_val=0):
		""" Assigns to each node a weight.
			
			Parameters:
			-------------
			tree: str ['base_tree', 'in_tree', 'out_tree']
				The tree to modify
			weights: dict
				{node_name : weight}
				The function assigns the same weight to each occurrence 
				of the same node (designated by the label). 
			weight_name: (optional) default='weight'
				The name of the attribute
			default_val: (optional) default=None
				The value of the weight if not defined in weights
		"""
		if tree == 'base_tree':
			weigths_with_ID = {}
			for node in self.base_tree.nodes():
				try:
					weigths_with_ID[node] = self.nodes_weights[self.base_labels(node)]
				except:
					weigths_with_ID[node] = default_val
			nx.set_node_attributes(self.base_tree, weigths_with_ID, name=weight_name)
		elif tree == 'in_tree':
			weigths_with_ID = {}
			for node in self.in_tree.nodes():
				try:
					weigths_with_ID[node] = self.nodes_weights[self.in_labels(node)]
				except:
					weigths_with_ID[node] = default_val
			nx.set_node_attributes(self.in_tree, weigths_with_ID, name=weight_name)
		elif tree == 'out_tree':
			weigths_with_ID = {}
			for node in self.out_tree.nodes():
				try:
					weigths_with_ID[node] = self.nodes_weights[self.out_labels(node)]
				except:
					weigths_with_ID[node] = default_val
			nx.set_node_attributes(self.out_tree, weigths_with_ID, name=weight_name)
		else:
			print('please enter a valid tree')
		
	def add_edges_weights(self, tree, default_val=0, update_prefix=True):
		""" Assigns each edge from one node labeled A to one node labeled B
			a weight. Edges between couples of nodes with the same couple of 
			labels have the same weight.
			Parameters:
			-------------
			tree: str ['base_tree', 'in_tree', 'out_tree']
				The tree to modify
			weights: dict
				{(from, to) : weight}
			default_val: (optional)
				The value of the weight if not defined in weights
		"""
		if tree == 'base_tree':
			for edge in self.base_tree.edges():
				edge_label_from = self.base_labels[edge[0]]
				edge_label_to = self.base_labels[edge[1]]
				try:
					self.base_tree[edge[0]][edge[1]]['weight'] = self.edges_weights[(edge_label_from, edge_label_to)]
				except:
					self.base_tree[edge[0]][edge[1]]['weight'] = default_val
		elif tree == 'in_tree':
			for edge in self.in_tree.edges():
				edge_label_from = self.in_labels[edge[0]]
				edge_label_to = self.in_labels[edge[1]]
				try:
					self.in_tree[edge[0]][edge[1]]['weight'] = self.edges_weights[(edge_label_from, edge_label_to)]
				except:
					self.in_tree[edge[0]][edge[1]]['weight'] = default_val
		elif tree == 'out_tree':
			for edge in self.out_tree.edges():
				edge_label_from = self.out_labels[edge[0]]
				edge_label_to = self.out_labels[edge[1]]
				try:
					self.out_tree[edge[0]][edge[1]]['weight'] = self.edges_weights[(edge_label_from, edge_label_to)]
				except:
					self.out_tree[edge[0]][edge[1]]['weight'] = default_val
		else:
			print('please enter a valid tree')

def _lid_2_coords(loc_ids, loc_longs, loc_lats):
	#return dict(zip(locations.id.values, zip(locations.long.values, locations.lat.values)))
	return dict(zip(loc_ids, zip(loc_longs, loc_lats)))

def _coords_2_lid(loc_ids, loc_longs, loc_lats): 
	#return dict(zip(zip(locations.long.values, locations.lat.values), locations.id.values))
	return dict(zip(zip(loc_longs, loc_lats), loc_ids))

def _id_trajs_to_coord_trajs(trajs, lid_2_coords):
	new_trajs = []
	for traj in trajs:
		new_traj = []
		for idp in traj:
			new_traj.append([lid_2_coords[idp][0], lid_2_coords[idp][1]])
		new_trajs.append(new_traj)
	return new_trajs

def prepocess_trajs(trajs, loc_ids, loc_longs, loc_lats, epsilon):
	""" Preprocess trajectories
		Cleans a user's trajectories by using RDP, then 
		finds the root as the most frequent location and finally 
		deletes all trajectories not containint it.
		
		Parameters:
		------------
		trajs: list 
			The list of trajectories, being each a list of location IDs
		loc_ids: list 
			The list of the location's IDs
		loc_longs: list 
			The list of the location's longitudes (loc_ids[0] has longitude loc_longs[0])
		loc_lats: list 
			The list of the location's longitudes (loc_ids[0] has longitude loc_lats[0])
		epsilon: float
			The RDP epsilon
	"""
	lid_2_coords = _lid_2_coords(loc_ids, loc_longs, loc_lats)
	coords_2_lid = _coords_2_lid(loc_ids, loc_longs, loc_lats)
	trajs_c = _id_trajs_to_coord_trajs(trajs, lid_2_coords)
	pruned = []
	for traj in trajs_c:
		t_prun = rdp(np.array(traj), epsilon=epsilon)
		pruned.append([coords_2_lid[(coords[0], coords[1])] for coords in t_prun])
	trajs = pruned
	
	e_weights = get_link_weights(trajs)
	n_weights = get_loc_weights(trajs)
	root = most_frequent_loc(n_weights)
	
	pruned = []
	for traj in trajs:
		if root in traj:
			pruned.append(traj)
	
	trajs = pruned
	
	return {'trajs' : trajs, 'root' : root, 'e_weights' : e_weights, 'n_weights' : n_weights}

def _median_tree_depth(trajs):
	""" Median tree depth
	Returns the median tree depth of a mobility Prefix Tree
	Parameters:
	-----------
	trajs: list
		the user's trajectories
	Returns:
	--------
	mean_depth : float
	"""
	if len(trajs) > 0:
		return round(np.median([len(t) for t in trajs]), 4)
	else: 
		return 0

def _max_tree_depth(trajs):
	""" Maximum tree depth
	Returns the maximum tree depth of a mobility Prefix Tree
	Parameters:
	-----------
	trajs: list
		the user's trajectories
	Returns:
	--------
	max_depth : float
	"""
	if len(trajs) > 0:
		return np.max([len(t) for t in trajs])
	else: 
		return 0

def _locations_nodes_ratio(trajs):
	""" Locations/Nodes ratio
	The proportion between visited places and the total nodes of the tree
	Parameters:
	-----------
	trajs: list
		the user's trajectories
	Returns:
	--------
	ratio : float
	"""
	all_trajs = []
	for t in trajs:
		all_trajs += t
	return round(len(set(all_trajs)) / len(all_trajs), 4)

def _median_out_degree(tree):
	""" Median out degree
	Measures the median out degree of a tree
	Parameters:
	-----------
	tree: nx.diGraph
		The prefix tree to be analysed. Note that nx.is_tree(tree) must be true
	Returns:
	--------
	degree: int
	"""
	if len(trajs) > 0:
		return round(np.median(np.array([val for (node, val) in tree.degree()])), 4)
	else: 
		return 0
	
def _max_out_degree(tree):
	""" Maximum out degree
	Measures the maximum out degree of a tree
	Parameters:
	-----------
	tree: nx.diGraph
		The prefix tree to be analysed. Note that nx.is_tree(tree) must be true
	Returns:
	--------
	degree: int
	"""
	if len(trajs) > 0:
		return np.max(np.array([val for (node, val) in tree.degree()]))
	else: 
		return 0    

def tree_to_vec(tree):
	""" Tree to vector
	Represents a tree through a vector with the following attributes:
	0. median in_tree depth
	1. median out_tree depth
	2. max in_tree depth
	3. max out_tree depth
	4. location / nodes ratio (whole tree)
	5. median in_tree out_degree
	6. median out_tree out_degree
	7. max in_tree out_degree
	8. max out_tree out_degree
	
	Parameters: 
	-----------
	tree: treemob.PrefixTree
		the tree to convert
	Returns:
	--------
	vec: list
		the vectorial representation of the tree
	"""
	
	base_tree = tree.base_tree
	in_tree = tree.in_tree
	out_tree = tree.out_tree
	
	trajs = [t[1 : -1] for t in tree.trajs]
	in_trajs = [t[:-1] for t in tree.in_trajs if len(t) > 1]
	out_trajs = [t[:-1] for t in tree.out_trajs if len(t) > 1]
	
	tree_vec = []
	
	tree_vec.append(_median_tree_depth(in_trajs))
	tree_vec.append(_median_tree_depth(out_trajs))
	
	tree_vec.append(_max_tree_depth(in_trajs))
	tree_vec.append(_max_tree_depth(out_trajs))
	
	tree_vec.append(_locations_nodes_ratio(trajs))
	
	tree_vec.append(_median_out_degree(tree.in_tree))
	tree_vec.append(_median_out_degree(tree.out_tree))
	
	tree_vec.append(_max_out_degree(tree.in_tree))
	tree_vec.append(_max_out_degree(tree.out_tree))
	
	return tree_vec

def _tf(trajs, vocab):
	trajs_tf = []
	
	for traj in trajs:
		temp = []
		num_t = len(traj)
		for t in vocab:
			temp.append(traj.count(t)/num_t)
		trajs_tf.append(np.array(temp))
	return np.array(trajs_tf)

def _relative_tf(trajs, vocab):
tf = []
for i, user in enumerate(trajs):
	root = treemob.most_frequent_loc(treemob.get_loc_weights(trajs[i]))
	weights = treemob.weight_locations(trajs[i], root, vocab=vocab)
	tf.append(list(weights.values()))
return np.array(tf)

def _idf(trajs, vocab):
	vocab_idf = list(np.zeros(len(vocab), dtype=int))
	
	for i, t in enumerate(vocab):
		for traj in trajs:
			if t in traj:
				vocab_idf[i] += 1
				
	vocab_idf = np.log(len(trajs) / (np.array(vocab_idf)))+1
	
	return vocab_idf

def tree_to_tfidf(trajs):
	""" TF-IDF for mobility data
	Vectorises the users' trajectories dataset assigning to each location
	a weight corresponding to its importance in the user's personal mobility.
	IDF is smoothed.
	
	Parameters:
	------------
	trajs: list of lists of int
		the dataset containing the trajectories of all the users
	"""
	
	N = len(trajs)
	vocab = treemob.create_vocab(T)

	trajs = [list(itertools.chain(*trajs[i])) for i in range(N)]
	
	tf = _tf(trajs, vocab)
	idf = _idf(trajs, vocab)
	
	tf_idf = tf * idf

	return tf_idf


def tree_to_relative_tfidf(trajs):
    """ TF-IDF for mobility data
    Vectorises the users' trajectories dataset assigning to each location
    a weight corresponding to its importance in the user's personal mobility.
    IDF is smoothed.
    
    Parameters:
    ------------
    trajs: list of lists of int
        the dataset containing the trajectories of all the users
    """
    
    N = len(trajs)
    vocab = treemob.create_vocab(T)

    original_trajs = trajs
    trajs = [list(itertools.chain(*trajs[i])) for i in range(N)]
    
    tf = _relative_tf(original_trajs, vocab)
    idf = _idf(trajs, vocab)
    
    tf_idf = tf * idf

    return tf_idf
	