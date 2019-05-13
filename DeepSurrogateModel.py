#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:03:09 2018

@author: cuijiaxu
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import time, os
import tensorflow as tf
import pickle

from gcn.utils import *
from gcn.layers import *
from gcn.metrics import *

import scipy.sparse as sp
from scipy.linalg import block_diag


class CombDGCNWithDNN():

    def __init__(self, model='gcn',basis_num=2 ,conv_layers=3, dens_layers=3, conv_hidden=16, dens_hidden=50, pool_hidden=50,conv_act_type=1, pool_act_type=2, dens_act_type=2,  learning_rate=0.01, epochs=200, batch_size=10,  dropout=0.5, weight_decay=5e-4, early_stopping=10, max_degree=3, dnn_act_type=2, dnn_layers=3, dnn_hidden=50, comb_act_type=2, last_hidden=50, inputvec_dim=None, ALPHA=0.5):
        """
        ('learning_rate', 0.01, 'Initial learning rate.')
        ('epochs', 200, 'Number of epochs to train.')
        ('hidden1', 16, 'Number of units in hidden layer 1.')
        ('dropout', 0.5, 'Dropout rate (1 - keep probability).')
        ('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
        ('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
        ('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
        """
        self.basis_num=basis_num
        
        self.model=model
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.batch_size=batch_size
        
        self.dropout=dropout
        self.weight_decay=weight_decay
        self.early_stopping=early_stopping
        self.max_degree=max_degree
        
        self.placeholders=None
        self.network=None
        self.sess=None
        
        self.canfeatures=None
        self.cansupport=None
        self.first_all=True
        self.All_cand_node_num=0
        self.dataset=None
        
        self.pre_features_train=None
        self.pre_support_train=None
        self.pre_first_train=True
        self.last_num_train=None
        self.save_summary=False
        self.info=None
        
        self.conv_layers=conv_layers
        self.dens_layers=dens_layers#not use
        self.conv_hidden=conv_hidden
        self.dens_hidden=dens_hidden#not use
        self.pool_hidden=pool_hidden
        self.conv_act_type=conv_act_type
        self.pool_act_type=pool_act_type
        self.dens_act_type=dens_act_type#not use
    
        ##for dnn
        self.dnn_act_type=dnn_act_type
        self.dnn_hidden=dnn_hidden
        self.dnn_layers=dnn_layers
        self.inputvec_dim=inputvec_dim
    
        #last comb layer
        self.comb_act_type=comb_act_type#not use
        self.last_hidden=last_hidden#not use
    
        self.ALPHA=ALPHA
        

    def train(self, X, y, flag):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X: Graph (N,)
            Input data points.
        y: np.ndarray (N,)
            The corresponding target values.

        """
        #load data
        #adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
        adj=X[:,1]        
        features=X[:,2]
        graphlevel_features_=X[:,3]
        if len(graphlevel_features_)>1:
                graphlevel_features=graphlevel_features_[0]
                for ff in graphlevel_features_[1:]:
                    #print(ff.shape)
                    #features_comb=block_diag(features_comb, ff)
                    graphlevel_features=np.vstack((graphlevel_features, ff))
        else:
            graphlevel_features=graphlevel_features_[0]
        #print("graphlevel_features",graphlevel_features,graphlevel_features.shape)
        #node_num_list=[]
        transition_matrix=np.zeros(len(features)*features[0].shape[0]).reshape(len(features),features[0].shape[0])
        transition_matrix[0,:]=1
        transition_matrix=sp.csr_matrix(transition_matrix)
    
        #y_train=[]
        for i in range(1,len(features)):
            #y_train=y_train+[y[i] for j in range(adj[i].shape[0])]
            #node_num_list.append(adj[i].shape[0])
            transition_matrix_i=np.zeros(len(features)*features[i].shape[0]).reshape(len(features),features[i].shape[0])
            transition_matrix_i[i,:]=1
            #transition_matrix=np.hstack((transition_matrix,transition_matrix_i))
            transition_matrix=sp.hstack([transition_matrix,sp.csr_matrix(transition_matrix_i)])
        #node_num_list=np.array(node_num_list).reshape(len(adj),1)
        print("transition_matrix:",transition_matrix.shape)
        
        transition_matrix_for_train=transition_matrix.todense()

        y_train=y
        y_train=np.array(y_train).reshape(len(y_train),1)
        
        #print(adj,features,y_train,np.array(adj).shape,adj[0].shape)

        #exit(1)
        #adj, features = X
        if self.pre_first_train==True:
        #if True:
            # Some preprocessing      
            if len(features)>1:
                features_comb=features[0]
                for ff in features[1:]:
                    #print(ff.shape)
                    #features_comb=block_diag(features_comb, ff)
                    features_comb=np.vstack((features_comb, ff))
            else:
                features_comb=features[0]
            """
            #add 0 until reaching self.All_cand_node_num:
            if features_comb.shape[0]<self.All_cand_node_num:
                extend_mat=np.zeros(features_comb.shape[0]*(self.All_cand_node_num-features_comb.shape[0])).reshape(features_comb.shape[0],self.All_cand_node_num-features_comb.shape[0])
                features_comb=np.hstack((features_comb,extend_mat))
            """
            #print(features_comb)
            print("train: features_comb shape is :",features_comb.shape)
            #exit(1)
            features=preprocess_features(sp.csr_matrix(features_comb).tolil())
            
            if self.model == 'gcn':
                if len(adj)>1:
                    adj_comb=adj[0].todense()
                    for adj_i in adj[1:]:
                        adj_comb=block_diag(adj_comb, adj_i.todense())
                else:
                    adj_comb=adj[0]
                print("train: adj_comb shape is :",adj_comb.shape)
                support = [preprocess_adj(sp.coo_matrix(adj_comb))]
                num_supports = 1
                model_func = CombGCNwithDNN
            elif self.model == 'gcn_cheby':
                if len(adj)>1:
                    adj_comb=adj[0].todense()
                    for adj_i in adj[1:]:
                        adj_comb=block_diag(adj_comb, adj_i.todense())
                else:
                    adj_comb=adj[0]
                print("train: adj_comb shape is :",adj_comb.shape)
                support = chebyshev_polynomials(adj_comb, self.max_degree)
                num_supports = 1 + self.max_degree
                model_func = CombGCNwithDNN
            elif self.model == 'rgcn_no_reg' or self.model == 'rgcn_with_reg':
                ##here is the code about support
                print("train: constructing adj...")
                num_supports=len(adj[0])
                adj_combs=[]
                if len(adj)>1:
                    for rela_num in range(num_supports):  
                        adj_comb=adj[0][rela_num]#.todense()
                        adj_combs.append(adj_comb)
                    for adj_idx in range(1,len(adj)):
                        for rela_num in range(num_supports):  
                            adj_i=adj[adj_idx][rela_num]#.todense()
                            #print(adj_combs[rela_num],adj_i)
                            adj_combs[rela_num]=sp.block_diag((adj_combs[rela_num], adj_i))
                else:
                    for rela_num in range(num_supports):  
                        adj_comb=adj[0][rela_num]
                        adj_combs.append(adj_comb)
                #support = [sp.coo_matrix(adj_comb_) for adj_comb_ in adj_combs]
                support = [preprocess_adj(sp.coo_matrix(adj_comb_)) for adj_comb_ in adj_combs]
                #support = [preprocess_adj(sp.coo_matrix(adj_comb))]
                #num_supports=len(adj[0])
                model_func = CombGCNwithDNN
            else:
                raise ValueError('Invalid argument for model: ' + str(self.model))
            self.pre_features_train=features
            self.pre_support_train=support
            self.pre_first_train=False
        else:
            ###前面处理过的+补上新处理的
            new_num=len(adj)-self.last_num_train
        
            adj=adj[-new_num:]
            features=features[-new_num:]
            # Some preprocessing      
            if len(features)>1:
                features_comb=features[0]
                for ff in features[1:]:
                    #print(ff.shape)
                    #features_comb=block_diag(features_comb, ff)
                    features_comb=np.vstack((features_comb, ff))
            else:
                features_comb=features[0]
            """
            #add 0 until reaching self.All_cand_node_num:
            if features_comb.shape[0]<self.All_cand_node_num:
                features_comb=np.hstack((features_comb,np.zeros(features_comb.shape[0]*(self.All_cand_node_num-features_comb.shape[0])).reshape(features_comb.shape[0],self.All_cand_node_num-features_comb.shape[0])))
            """
            #print(features_comb)
            #print("train: features_comb shape is :",features_comb.shape)
            #exit(1)
            features=preprocess_features(sp.csr_matrix(features_comb).tolil())
            features=sp.csr_matrix(np.vstack((self.pre_features_train.todense(),features.todense()))).tolil()
            print("train: features shape is :",features.shape)
            if self.model == 'gcn':
                if len(adj)>1:
                    adj_comb=adj[0].todense()
                    for adj_i in adj[1:]:
                        adj_comb=block_diag(adj_comb, adj_i.todense())
                else:
                    adj_comb=adj[0]
                #print("train: adj_comb shape is :",adj_comb.shape)
                support = [sp.coo_matrix(block_diag(self.pre_support_train[0].todense(),preprocess_adj(sp.coo_matrix(adj_comb)).todense())).tolil()]
                print("train: support shape is :",support[0].shape)
                num_supports = 1
                model_func = CombGCNwithDNN
            elif self.model == 'gcn_cheby':
                if len(adj)>1:
                    adj_comb=adj[0].todense()
                    for adj_i in adj[1:]:
                        adj_comb=block_diag(adj_comb, adj_i.todense())
                else:
                    adj_comb=adj[0]
                #print("train: adj_comb shape is :",adj_comb.shape)
                support = chebyshev_polynomials(adj_comb, self.max_degree)
                support_temp=[]
                for i in range(len(support)):
                    support_temp.append(sp.coo_matrix(block_diag(self.pre_support_train[i].todense(),support[i].todense())).tolil())
                support=support_temp
                print("train: support shape is :",support[0].shape)
                num_supports = 1 + self.max_degree
                model_func = CombGCNwithDNN
                
            elif self.model == 'rgcn_no_reg' or self.model == 'rgcn_with_reg':
                ##here is the code about support
                print("train: constructing adj...")
                num_supports=len(adj[0])
                #print(num_supports)
                adj_combs=[]
                if len(adj)>1:
                    for rela_num in range(num_supports):  
                        adj_comb=adj[0][rela_num]#.todense()
                        adj_combs.append(adj_comb)
                    for adj_idx in range(1,len(adj)):
                        for rela_num in range(num_supports):  
                            adj_i=adj[adj_idx][rela_num]#.todense()
                            adj_combs[rela_num]=sp.block_diag((adj_combs[rela_num], adj_i))
                else:
                    for rela_num in range(num_supports):  
                        adj_comb=adj[0][rela_num]
                        adj_combs.append(adj_comb)
               
                support = [sp.coo_matrix(sp.block_diag((self.pre_support_train[adj_comb_idx],preprocess_adj(sp.coo_matrix(adj_combs[adj_comb_idx]))))).tolil() for adj_comb_idx in range(len(adj_combs))]
                #support = [preprocess_adj(sp.coo_matrix(adj_comb))]
                #num_supports=len(adj[0])
                model_func = CombGCNwithDNN
            
            else:
                raise ValueError('Invalid argument for model: ' + str(self.model))
            #
            self.pre_features_train=features
            self.pre_support_train=support
            
        features=sparse_to_tuple(features)
        support_ = []
        for support_i in support:
            support_.append(sparse_to_tuple(support_i))
        support=support_
        if flag==True:
            #print(features[2],features[2][1],len(features))
            # Define placeholders
            placeholders = { 
                    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
                    #'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
                    'features': tf.sparse_placeholder(tf.float32),
                    #'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
                    'labels': tf.placeholder(tf.float32, shape=(None, 1)),
                    #'labels_mask': tf.placeholder(tf.int32),#do not use
                    'dropout': tf.placeholder_with_default(0., shape=()),
                    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
                    'transition_matrix': tf.placeholder(tf.float32, shape=(None,None)),
                    'graphlevel_features' : tf.placeholder(tf.float32)
            }
            self.placeholders=placeholders
            # Create model
            self.network = model_func(self,placeholders, input_dim=features[2][1], logging=True)
            # Initialize session
            self.sess = tf.Session()
           
            # Init variables
            self.sess.run(tf.global_variables_initializer())
            
            
            if self.save_summary==True:
                #save summary information
                merged_summary_op = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter('results/temp/train_summary_logs', self.sess.graph)
            

            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, support, y_train.reshape(len(y_train),1), placeholders, transition_matrix_for_train, graphlevel_features)
            feed_dict.update({placeholders['dropout']: self.dropout})
            print("Start to train network.")
            start_time = time.time()
            # Train model
            for epoch in range(self.epochs):
    
                t = time.time()
                
    
                # Training step
                outs = self.sess.run([self.network.opt_op, self.network.loss], feed_dict=feed_dict)
                
                if epoch % 100 == 0:
                    # Print results
                    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                          "time=", "{:.5f}s".format(time.time() - t), "total_time=", "{:.5f}s".format(time.time() - start_time))
        
                if self.save_summary==True:
                    if (epoch+1) % 100 == 0:
                        summary_str = self.sess.run(merged_summary_op,feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, epoch+1)
                
                #if epoch > self.early_stopping and cost_val[-1] > np.mean(cost_val[-(self.early_stopping+1):-1]):
                    #print("Early stopping...")
                    #break
    
            print("Train net Finished!")
            
        feed_dict_val = construct_feed_dict(features, support, [], self.placeholders, transition_matrix_for_train, graphlevel_features)
        outs_val = self.sess.run([self.network.basis], feed_dict=feed_dict_val)
        
        self.last_num_train=len(X[:,1])
        
        return outs_val[-1]
    
    # Get features from the net
    def get_basis(self,X_test,y_test):
        
        #node_num_list=np.array(node_num_list).reshape(len(adj),1)
        #print(transition_matrix,transition_matrix.shape)
        #print(self.train_nodesnum,len(X_test[:,1]),len(self.train_nodesnum),sum(self.train_nodesnum))
        #exit(1)
        if len(y_test)!=0:  
            y_test=np.array(y_test).reshape(len(y_test),1)
        
        if self.first_all==True:
            start_prepro=time.time()
            rseed=self.info.rseed
            if os.path.exists("%s/temp"%(self.dataset)) and os.path.exists("%s/temp/canfeatures.pk"%(self.dataset)) and os.path.exists("%s/temp/cansupport.pk"%(self.dataset)) and os.path.exists("%s/temp/cantransition_matrix.pk"%(self.dataset)):
                with open("%s/temp/canfeatures.pk"%(self.dataset), 'r') as f1:
                    self.canfeatures=pickle.load(f1)
                with open("%s/temp/cansupport.pk"%(self.dataset), 'r') as f2:
                    self.cansupport=pickle.load(f2)
                with open("%s/temp/cantransition_matrix.pk"%(self.dataset), 'r') as f3:
                    self.cantransition_matrix=sp.csr_matrix(pickle.load(f3)).todense()
            else:
                print("start to construct adj ... ")
                adj=X_test[:,1]
                features=X_test[:,2]
                               
                if self.model == 'gcn':
                    if len(adj)>1:
                        adj_comb=adj[0]#.todense()
                        #print("adj_comb",adj_comb.toarray())
                        for adj_i in adj[1:]:
                            #print("adj_i",sp.coo_matrix(adj_i.toarray()))
                            adj_comb=sp.block_diag((adj_comb, adj_i))
                            #print("adj_comb",adj_comb)
                    else:
                        adj_comb=adj[0]
                    print("get_basis: adj_comb shape is :",adj_comb.shape)
                    support = [sparse_to_tuple(preprocess_adj(sp.coo_matrix(adj_comb)))]
                    #exit(1)
                elif self.model == 'gcn_cheby':
                    if len(adj)>1:
                        adj_comb=adj[0].todense()
                        for adj_i in adj[1:]:
                            adj_comb=block_diag(adj_comb, adj_i.todense())
                    else:
                        adj_comb=adj[0]
                    print("get_basis: adj_comb shape is :",adj_comb.shape)
                    support = chebyshev_polynomials(adj_comb, self.max_degree)
                    support_ = []
                    for support_i in support:
                        support_.append(sparse_to_tuple(support_i))
                    support=support_
                elif self.model == 'rgcn_no_reg' or self.model == 'rgcn_with_reg':
                    ##here is the code about support
                    num_supports=len(adj[0])
                    adj_combs=[]
                    print(num_supports,"making a diag block matrix ... ")
                    if len(adj)>1:
                        for rela_num in range(num_supports):  
                            adj_comb=adj[0][rela_num]#.todense()
                            adj_combs.append(adj_comb)
                        for adj_idx in range(1,len(adj)):
                            if adj_idx%200==1:
                                print(adj_idx+1,'/',len(adj))
                            for rela_num in range(num_supports):  
                                adj_i=adj[adj_idx][rela_num]#.todense()
                                adj_combs[rela_num]=sp.block_diag((adj_combs[rela_num], adj_i))
                    else:
                        for rela_num in range(num_supports):  
                            adj_comb=adj[0][rela_num]
                            adj_combs.append(adj_comb)
                    print("transfering diag block matrix into sparse tuple ... ")
                    support = [sparse_to_tuple(preprocess_adj(sp.coo_matrix(adj_comb_))) for adj_comb_ in adj_combs]
                    #support = [preprocess_adj(sp.coo_matrix(adj_comb))]
                    #num_supports=len(adj[0])
                    #model_func = CombGCNwithDNN
                else:
                    raise ValueError('Invalid argument for model: ' + str(self.model))
                print("start to construct transition_matrix ... ")
                adj=X_test[:,1]
                transition_matrix=np.zeros(len(features)*features[0].shape[0]).reshape(len(features),features[0].shape[0])
                transition_matrix[0,:]=1
                transition_matrix=sp.csr_matrix(transition_matrix)
                #y_train=[]
                for i in range(1,len(features)):
                    #y_train=y_train+[y[i] for j in range(adj[i].shape[0])]
                    #node_num_list.append(adj[i].shape[0])
                    transition_matrix_i=np.zeros(len(features)*features[i].shape[0]).reshape(len(features),features[i].shape[0])
                    transition_matrix_i[i,:]=1
                    #transition_matrix=np.hstack((transition_matrix,transition_matrix_i))
                    transition_matrix=sp.hstack([transition_matrix,sp.csr_matrix(transition_matrix_i)])
                print("ok!")
                print("start to construct features ... ")
                # Some preprocessing
                if len(features)>1:
                    features_comb=features[0]
                    for ff in features[1:]:
                        #print(ff.shape)
                        #features_comb=block_diag(features_comb, ff)
                        features_comb=np.vstack((features_comb, ff))
                else:
                    features_comb=features[0]
                """
                #add 0 until reaching self.All_cand_node_num:
                if features_comb.shape[0]<self.All_cand_node_num:
                    features_comb=np.hstack((features_comb,np.zeros(features_comb.shape[0]*(self.All_cand_node_num-features_comb.shape[0])).reshape(features_comb.shape[0],self.All_cand_node_num-features_comb.shape[0])))
                """
                print("get_basis: features_comb shape is :",features_comb.shape)
                        
                features=sparse_to_tuple(preprocess_features(sp.csr_matrix(features_comb).tolil()))
                
                """
                features=sparse_to_tuple(features)
                support_ = []
                for support_i in support:
                    support_.append(sparse_to_tuple(support_i))
                support=support_
                """
                self.canfeatures=features
                self.cansupport=support
                self.cantransition_matrix=transition_matrix.todense()
    
                if not os.path.exists("%s/temp"%(self.dataset)):
                    os.makedirs("%s/temp"%(self.dataset))
                    
                with open("%s/temp/canfeatures.pk"%(self.dataset), 'w') as f1:
                    pickle.dump(self.canfeatures, f1)
                with open("%s/temp/cansupport.pk"%(self.dataset), 'w') as f2:
                    pickle.dump(self.cansupport, f2)
                with open("%s/temp/cantransition_matrix.pk"%(self.dataset), 'w') as f3:
                    pickle.dump(sp.csr_matrix(transition_matrix), f3)
                
            self.first_all=False
            print("Preprocess time is : ",time.time()-start_prepro,"s")
            
        graphlevel_features_=X_test[:,3]
        if len(graphlevel_features_)>1:
            graphlevel_features=graphlevel_features_[0]
            for ff in graphlevel_features_[1:]:
                graphlevel_features=np.vstack((graphlevel_features, ff))
        else:
            graphlevel_features=graphlevel_features_[0]
        
        feed_dict_val = construct_feed_dict(self.canfeatures, self.cansupport, y_test, self.placeholders, self.cantransition_matrix, graphlevel_features)
        if len(y_test)!=0:
            outs_val = self.sess.run([self.network.accuracy,self.network.basis], feed_dict=feed_dict_val)
            print("candidates_acc = %s"%outs_val[0])
            #print(outs_val[-2],outs_val[-2].shape,len(outs_val[-1]))
            #print(y_train,outs_val[-1][-1])
        else:
            outs_val = self.sess.run([self.network.basis], feed_dict=feed_dict_val)
        return outs_val[-1]

        # Get features from the net
        ###############
    def get_basis_one(self,X_test):
        adj=X_test[1]
        features=X_test[2]
        if self.model == 'gcn':
            support = [sparse_to_tuple(preprocess_adj(sp.coo_matrix(adj)))]
        elif self.model == 'gcn_cheby':
            support = chebyshev_polynomials(adj, self.max_degree)
            support_ = []
            for support_i in support:
                support_.append(sparse_to_tuple(support_i))
            support=support_
        elif self.model == 'rgcn_no_reg' or self.model == 'rgcn_with_reg':
            ##here is the code about support
            #support=sparse_to_tuple(adj)
            
            ###In this function, we do nothing here, cause we have done preprocess in load_candidates() of main file.
            num_supports=len(adj)
            adj_combs=[]
            for rela_num in range(num_supports):
                adj_comb=adj[rela_num]
                adj_combs.append(adj_comb)
            support = [sparse_to_tuple((sp.coo_matrix(adj_comb_))) for adj_comb_ in adj_combs]
            #support = [sparse_to_tuple(preprocess_adj(sp.coo_matrix(adj_comb_))) for adj_comb_ in adj_combs]
                #support = [preprocess_adj(sp.coo_matrix(adj_comb))]
                #num_supports=len(adj[0])
                #model_func = CombGCNwithDNN
            
        else:
            raise ValueError('Invalid argument for model: ' + str(self.model))
                    
        transition_matrix=np.ones(1*features.shape[0]).reshape(1,features.shape[0])
        #transition_matrix=sp.csr_matrix(transition_matrix)
        
        features=sparse_to_tuple(preprocess_features(sp.csr_matrix(features).tolil()))
        
        graphlevel_features=X_test[3].reshape(1,len(X_test[3]))
        #print("graphlevel_features : %s"%(graphlevel_features))

        feed_dict_val = construct_feed_dict(features, support, [], self.placeholders, transition_matrix, graphlevel_features)
            
        outs_val = self.sess.run([self.network.basis], feed_dict=feed_dict_val)
        return outs_val[-1]

    def _build_net(self, input_var, features):
        raise NotImplementedError

def construct_feed_dict(features, support, labels, placeholders, transition_matrix, graphlevel_features):
    """Construct feed dictionary."""
    feed_dict = dict()
    if len(labels)!=0:    
        feed_dict.update({placeholders['labels']: labels})
    #feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    feed_dict.update({placeholders['transition_matrix']: transition_matrix})
    feed_dict.update({placeholders['graphlevel_features']: graphlevel_features})
    
    return feed_dict
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features
def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized
def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return t_k

    
class CombModel(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []
        self.activations_dnn = []
        self.basis = None
        
        self.ALPHA=None
        
        self.inputs = None
        self.inputs_vec=None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.layernumofgcn=None
    

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        print("Build DGCN_Comb2...")
        # Build sequential layer model
        count=0
        self.activations.append(self.inputs)
        for layer in self.layers[0:self.layernumofgcn]:
            count+=1
            print("comb gcn hahaha",count)
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
                
        print("ALPHA=%s"%self.ALPHA)
        self.activations_dnn.append(tf.concat([self.ALPHA*self.activations[-1],(1.0-self.ALPHA)*self.inputs_vec],1))
        for layer in self.layers[self.layernumofgcn:]:
            count+=1
            print("comb dnn hahaha",count)
            hidden = layer(self.activations_dnn[-1])
            self.activations_dnn.append(hidden)
        
        self.basis=self.activations_dnn[-2]
        #self.basis = tf.concat([ALPHA*self.activations[-1],(1.0-ALPHA)*self.activations_dnn[-1]],1)
        
        #self.basis = alpha*self.activations[-1]+(1.0-alpha)*self.activations_dnn[-1]
        #self.basis=self.ALPHA*self.layers[-3](self.activations[-1])+(1-self.ALPHA)*self.layers[-2](self.activations_dnn[-1])
        
        #count+=1
        #print("comb out hahaha",count)
        self.outputs = self.activations_dnn[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)    
        
class CombGCNwithDNN(CombModel):
    def __init__(self, dgcn, placeholders, input_dim, **kwargs):
        super(CombGCNwithDNN, self).__init__(**kwargs)
        
        self.layernumofgcn=dgcn.conv_layers+1
        
        self.dgcn=dgcn
        self.ALPHA=dgcn.ALPHA
        
        self.inputs_vec = placeholders['graphlevel_features']
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.dgcn.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for idx in range(len(self.layers)):
            for var in self.layers[idx].vars.values():
                self.loss += self.dgcn.weight_decay * tf.nn.l2_loss(var)
            
        #MSE,mean squared error 
        #self.loss += tf.nn.l2_loss(self.outputs - self.placeholders['labels'])
        self.loss+=tf.reduce_mean(tf.square(self.outputs - self.placeholders['labels']))

    def _accuracy(self):
        #self.accuracy=tf.nn.l2_loss(self.outputs - self.placeholders['labels'])
        self.accuracy=tf.reduce_mean(tf.square(self.outputs - self.placeholders['labels']))
        #self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
         #                               self.placeholders['labels_mask'])

    def _build(self):
        
        ##GCN
        input_dim=self.input_dim
        output_dim=self.dgcn.conv_hidden
        for idx in range(self.dgcn.conv_layers):
            if idx==0:
                sparse_inputs_flag=True
            else:
                sparse_inputs_flag=False
            if self.dgcn.conv_act_type==1:
                act_type=tf.nn.relu
            elif self.dgcn.conv_act_type==2:
                act_type=tf.nn.tanh
            elif self.dgcn.conv_act_type==0:
                act_type=lambda x: x
            else:
                print("conv_act_type is wrong!")
                exit(1)
            if self.dgcn.model=='gcn' or self.dgcn.model=='gcn_cheby':                
                self.layers.append(GraphConvolution(input_dim=input_dim,
                                                    output_dim=output_dim,
                                                    placeholders=self.placeholders,
                                                    act=act_type,
                                                    dropout=True,
                                                    sparse_inputs=sparse_inputs_flag,
                                                    logging=self.logging))
            elif self.dgcn.model=='rgcn_no_reg':
                self.layers.append(RelationGraphConvolution_noBasisRegularization(input_dim=input_dim,
                                                    output_dim=output_dim,
                                                    placeholders=self.placeholders,
                                                    act=act_type,
                                                    dropout=True,
                                                    sparse_inputs=sparse_inputs_flag,
                                                    logging=self.logging))
            elif self.dgcn.model=='rgcn_with_reg':
                self.layers.append(RelationGraphConvolution_withBasisRegularization(basis_num=self.dgcn.basis_num,input_dim=input_dim,
                                                    output_dim=output_dim,
                                                    placeholders=self.placeholders,
                                                    act=act_type,
                                                    dropout=True,
                                                    sparse_inputs=sparse_inputs_flag,
                                                    logging=self.logging))
            else:
                print("model is wrong!")
                exit(1)
            input_dim=output_dim
            output_dim=self.dgcn.conv_hidden
            
        if self.dgcn.pool_act_type==1:
            act_type=tf.nn.relu
        elif self.dgcn.pool_act_type==2:
            act_type=tf.nn.tanh
        elif self.dgcn.pool_act_type==0:
            act_type=lambda x: x
        else:
            print("pool_act_type is wrong!")
            exit(1)
        self.layers.append(Pooling_sum_normal_params(input_dim=self.dgcn.conv_hidden,
                                   output_dim=self.dgcn.pool_hidden,
                                   placeholders=self.placeholders,
                                   act=act_type,
                                   dropout=True,
                                   logging=self.logging))
        
        ##DNN
        #print("self.dgcn.inputvec_dim",self.dgcn.inputvec_dim)
        input_dim=self.dgcn.inputvec_dim+self.dgcn.pool_hidden
        output_dim=self.dgcn.dnn_hidden
        for  idx in range(self.dgcn.dnn_layers):
            if self.dgcn.dnn_act_type==1:
                act_type=tf.nn.relu
            elif self.dgcn.dnn_act_type==2:
                act_type=tf.nn.tanh
            elif self.dgcn.dnn_act_type==0:
                act_type=lambda x: x
            else:
                print("dnn_act_type is wrong!")
                exit(1)
            self.layers.append(Dense(input_dim=input_dim,
                                     output_dim=output_dim,
                                     placeholders=self.placeholders,
                                     act=act_type,
                                     dropout=True,
                                     bias=True,
                                     logging=self.logging))
            input_dim=output_dim
            output_dim=self.dgcn.dnn_hidden
        
        ##output
        self.layers.append(Dense(input_dim=self.dgcn.dnn_hidden,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 bias=True,
                                 logging=self.logging))

        

    def predict(self):
        return self.outputs


    
class Pooling_sum_paramsfree(Layer):
    """Pooling layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.tanh, bias=False,
                 featureless=False, **kwargs):
        super(Pooling_sum_paramsfree, self).__init__(**kwargs)
        
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        #self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        
        self.transition_matrix=placeholders['transition_matrix']
        """
        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
        """

        #if self.logging:
            #self._log_vars()

    def _call(self, inputs):
        x = inputs
        """
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
        """
        # transform 
        #output=tf.reduce_sum(x, 0, keep_dims=True)
        """
        output=[]
        print(self.dgcn.train_nodesnum,x.shape)
        start_idx=0
        for idx in range(len(self.dgcn.train_nodesnum)):
            g_pooling=tf.reduce_sum(x[start_idx:start_idx+self.dgcn.train_nodesnum[idx],:], 0)
            output=output+[g_pooling for i in range(self.dgcn.train_nodesnum[idx])]
            start_idx=self.dgcn.train_nodesnum[idx]
        #print(tf.stack(output))
        output = tf.stack(output)
        return self.act(output)
        """
        """
        output=[]
        print(self.node_num_list,x.shape,self.node_num_list.shape[0])
        start_idx=0
        for idx in range(self.node_num_list.shape[0]):
            g_pooling=tf.reduce_sum(x[start_idx:start_idx+self.node_num_list[idx],:], 0)
            output=output+[g_pooling]
            start_idx=self.node_num_list[idx]
        print(tf.stack(output))
        #exit(1)
        output = tf.stack(output)
        """
        output=tf.matmul(self.transition_matrix, x)
        
        return self.act(output)
    
class Pooling_sum_normal_params(Layer):
    """Pooling layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.tanh, bias=False,
                 featureless=False, **kwargs):
        super(Pooling_sum_normal_params, self).__init__(**kwargs)
        
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.transition_matrix=placeholders['transition_matrix']
        
       # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
       # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']
        #output = dot(self.transition_matrix, tf.nn.softmax(output), sparse=True)
        output=tf.matmul(self.transition_matrix, tf.nn.softmax(output))
        
        return self.act(output)
class Pooling_sum_normal_params2(Layer):
    """Pooling layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.tanh, bias=False,
                 featureless=False, **kwargs):
        super(Pooling_sum_normal_params2, self).__init__(**kwargs)
        
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.transition_matrix=placeholders['transition_matrix']
        
       # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
       # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']
        
        output=tf.matmul(self.transition_matrix, tf.nn.softmax(self.act(output)))
        
        return self.act(output)


###RelationGraphConvolution Layer####
class RelationGraphConvolution_noBasisRegularization(Layer):
    """Relation convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(RelationGraphConvolution_noBasisRegularization, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


###RelationGraphConvolution Layer####
class RelationGraphConvolution_withBasisRegularization(Layer):
    """Relation convolution layer."""
    def __init__(self, basis_num ,input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(RelationGraphConvolution_withBasisRegularization, self).__init__(**kwargs)
        
        self.basis_num=basis_num
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(self.basis_num):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            
            self.vars['weights_comp'] = glorot([len(self.support), self.basis_num],
                                                        name='weights_comp')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            weight_i = list()
            for j in range(self.basis_num):
                weight_i.append(self.vars['weights_comp'][i,j]*self.vars['weights_' + str(j)])
            weight_i=tf.add_n(weight_i)
                
            if not self.featureless:
                pre_sup = dot(x, weight_i,
                              sparse=self.sparse_inputs)
            else:
                pre_sup = weight_i
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
