#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:18:43 2017

@author: cuijiaxu
"""

import numpy as np
import os
import cPickle as pickle
import time

def load_data(inputFilename,topN=None):
    
    #read
    edge_lists=[]
    feature_lists=[]
    label_lists=[]
    attr_lists=[]
    graphs=pickle.load(open("%s-graph.pkl"%inputFilename, 'rb'))
    attrs=pickle.load(open("%s-attr.pkl"%inputFilename, 'rb'))
    labels=pickle.load(open("%s-label.pkl"%inputFilename, 'rb'))
    smilesList=list(graphs.keys())
    for idx in range(len(smilesList)):
        attr_lists.append(attrs[smilesList[idx]])
        label_lists.append(labels[smilesList[idx]])
        #adj_list=graphs[smilesList[idx]]['atom_neighbors_list']
#        lists=[]
#        for neig_list_idx in range(len(adj_list)):
#            lists=lists+[[neig_list_idx,int(neig)] for neig in adj_list[neig_list_idx]]
#        #print(idx+1,"/",len(smilesList),smilesList[idx])
#        edge_lists.append(lists)
        
        #all edges
        bnl=np.array(graphs[smilesList[idx]]['bond_neighbors_list'],np.integer)
        #put all edges into different relation parts
        bf=np.array(graphs[smilesList[idx]]['bond_features'],np.integer)
        lists=[]
        for rel_idx in range(bf.shape[1]):
            lists.append(bnl[bf[:,rel_idx]>0])
        edge_lists.append(lists)
 
        feature_lists.append(np.array(graphs[smilesList[idx]]['atom_features'],dtype=np.integer))
    print "Load %s ok!"%inputFilename
    
    if topN==None:
        return smilesList,edge_lists,feature_lists,np.array(label_lists,np.float),np.array(attr_lists,np.float)
    else:
        return smilesList[0:topN],edge_lists[0:topN],feature_lists[0:topN],np.array(label_lists[0:topN],np.float),np.array(attr_lists[0:topN],np.float)
def load_data_y(inputFilename,topN=None):
    #read
    label_lists=[]
    graphs=pickle.load(open("%s-graph.pkl"%inputFilename, 'rb'))
    labels=pickle.load(open("%s-label.pkl"%inputFilename, 'rb'))
    smilesList=list(graphs.keys())
    for idx in range(len(smilesList)):
       
        label_lists.append(labels[smilesList[idx]])
    if topN==None:
        return np.array(label_lists,np.float)
    else:
        return np.array(label_lists[0:topN],np.float)
