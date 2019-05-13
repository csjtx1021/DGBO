#!/usr/bin/python

from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Draw
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from neuralFingerprintUtils import *
#import cPickle as pickle
import pickle as pickle
import time, sys

smilesList = ['CC']
degrees = [0, 1, 2, 3, 4, 5]


class MolGraph(object):
    def __init__(self):
        self.nodes = {} # dict of lists of nodes, keyed by node type

    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))

    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i : [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)

        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]

class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']
    def __init__(self, ntype, features, rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix

    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]

class memoize(object):
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            result = self.func(*args)
            self.cache[args] = result
            return result

    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)

def graph_from_smiles_tuple(smiles_tuple):
    graph_list = [graph_from_smiles(s) for s in smiles_tuple]
    big_graph = MolGraph()
    for subgraph in graph_list:
        big_graph.add_subgraph(subgraph)

    # This sorting allows an efficient (but brittle!) indexing later on.
    big_graph.sort_nodes_by_degree('atom')
    return big_graph

def graph_from_smiles(smiles):
    graph = MolGraph()
    mol = MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():
        new_atom_node = graph.new_node('atom', features=atom_features(atom), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node
    
    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))

    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph

def array_rep_from_smiles(molgraph):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    #molgraph = graph_from_smiles_tuple(tuple(smiles))
    degrees = [0,1,2,3,4,5]
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'),
                'rdkit_ix'      : molgraph.rdkit_ix_array(),
                'atom_neighbors_list': molgraph.neighbor_list('atom', 'atom'),
                'bond_neighbors_list': molgraph.neighbor_list('bond', 'atom'),
                }
    
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)


    return arrayrep


def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))

def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))


def gen_descriptor_data(smilesList):

    #first get the keys from the graph representation
    #molgraph = graph_from_smiles(smilesList[0])
    #molgraph.sort_nodes_by_degree('atom')
    #arrayrep = array_rep_from_smiles(molgraph)
    #print(arrayrep,list(arrayrep.keys()))
    #arrayRepKeys = sorted(arrayrep.keys())
    smiles_to_fingerprint_array = {}

    for i,smiles in enumerate(smilesList):
        
        if i % 500 == 0:
            print(i)
        mol = MolFromSmiles(smiles)


        try:
            molgraph = graph_from_smiles(smiles)
            molgraph.sort_nodes_by_degree('atom')
            arrayrep = array_rep_from_smiles(molgraph)

            smiles_to_fingerprint_array[smiles] = arrayrep
        

        except:
            print(smiles)
            time.sleep(3)
    #print(smilesList[-1],smiles_to_fingerprint_array[smilesList[-99]]['atom_neighbors_list'])
    return smiles_to_fingerprint_array


def reprocess_smiles_to_graph(inputFilename,outputFilename,smile_idx,label_idx,attr_idx_list):
    smiles_attr_dict = {}
    smiles_label_dict = {}
    INPUT = open(inputFilename, 'r')
    count=0
    for line in INPUT:
        count+=1
        if count==1:
            continue
        line = line.rstrip()
        line_list = line.split(',')
        print(line_list)
        smiles = line_list[smile_idx]
        smiles_label_dict[smiles] = line_list[label_idx]
        
        print([np.float(line_list[idx]) for idx in attr_idx_list])
        smiles_attr_dict[smiles] = [np.float(line_list[idx]) for idx in attr_idx_list]
    
        #print(line_list[1],line_list[0])
    INPUT.close()

    smilesList=list(smiles_label_dict.keys())
    smiles_to_fingerprint_array = gen_descriptor_data(smilesList)

    pickle.dump(smiles_to_fingerprint_array,open("%s-graph.pkl"%outputFilename,"wb"),protocol=0)
    pickle.dump(smiles_attr_dict,open("%s-attr.pkl"%outputFilename,"wb"),protocol=0)
    pickle.dump(smiles_label_dict,open("%s-label.pkl"%outputFilename,"wb"),protocol=0)
    
    print("Num Smiles",len(list(smiles_label_dict.keys())))
    #read
    edge_lists=[]
    feature_lists=[]
    label_lists=[]
    attr_lists=[]
    graphs=pickle.load(open("%s-graph.pkl"%outputFilename, 'rb'))
    attrs=pickle.load(open("%s-attr.pkl"%outputFilename, 'rb'))
    labels=pickle.load(open("%s-label.pkl"%outputFilename, 'rb'))
    smilesList=list(graphs.keys())
    stat_num_node=0
    stat_num_edge=0
    stat_num_nodefeature=0
    stat_num_edgefeature=0
    for idx in range(len(smilesList)):
        attr_lists.append(attrs[smilesList[idx]])
        label_lists.append(labels[smilesList[idx]])
        adj_list=graphs[smilesList[idx]]['atom_neighbors_list']
        lists=[]
        for neig_list_idx in range(len(adj_list)):
            lists=lists+[[neig_list_idx,neig] for neig in adj_list[neig_list_idx]]
        #print(idx+1,"/",len(smilesList),smilesList[idx])
        stat_num_node+=len(graphs[smilesList[idx]]['atom_list'][0])
        stat_num_edge+=len(lists)
        edge_lists.append(lists)
        feature_lists.append(np.array(smiles_to_fingerprint_array[smilesList[idx]]['atom_features'],dtype=np.integer))
        stat_num_nodefeature=len(np.array(smiles_to_fingerprint_array[smilesList[idx]]['atom_features'],dtype=np.integer)[0])
        stat_num_edgefeature=len(np.array(smiles_to_fingerprint_array[smilesList[idx]]['bond_features'],dtype=np.integer)[0])

    print(np.array(label_lists,np.float))
    print('len(adj)=',len(edge_lists),'len(attr)=',len(attr_lists),'len(label)=',len(label_lists))
    print('stat_num_node=',stat_num_node,', stat_num_edge=',stat_num_edge,', stat_num_nodefeature=',stat_num_nodefeature,', stat_num_edgefeature=',stat_num_edgefeature)
    return edge_lists,feature_lists,np.array(label_lists,np.float),np.array(attr_lists,np.float)

if __name__ == "__main__":
    
    if sys.argv[1]=="delaney":        
        inputFilename = '../datasets/delaney-processed.csv'
        outputFilename = '../datasets/delaney-processed'
        edge_lists,feature_lists,label_lists,attr_lists=reprocess_smiles_to_graph(inputFilename,outputFilename,8,7,[1,2,3,4,5,6])
    elif sys.argv[1]=="malaria":
        inputFilename = '../datasets/malaria-processed.csv'
        outputFilename = '../datasets/malaria-processed'
        edge_lists,feature_lists,label_lists,attr_lists=reprocess_smiles_to_graph(inputFilename,outputFilename,0,1,[])
    elif sys.argv[1]=="cep":
        inputFilename = '../datasets/cep-processed.csv'
        outputFilename = '../datasets/cep-processed'
        edge_lists,feature_lists,label_lists,attr_lists=reprocess_smiles_to_graph(inputFilename,outputFilename,0,1,[])
    elif sys.argv[1]=="zinc":
        inputFilename = '../datasets/20k_rndm_zinc_drugs_clean_3.csv'
        outputFilename = '../datasets/20k_rndm_zinc_drugs_clean_3'
        edge_lists,feature_lists,label_lists,attr_lists=reprocess_smiles_to_graph(inputFilename,outputFilename,0,4,[])
    else:
        print("Do not know dataset name [%s]. You should choose from delaney, malaria or cep."%sys.argv[1])


