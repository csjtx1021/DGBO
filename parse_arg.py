#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 8 10:18:43 2018

@author: cuijiaxu
"""

import argparse

def parse_arg():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--run', default=False,
                        help='flag for running this model (default: False), if you want to run, you should add [--run=True] option.')
    parser.add_argument('--rseed', default=None,
                        help='random seed (default: None)')
    parser.add_argument('--conv_layers', default=5,
                        help='the number of graph convolution layers (default: 5)')
    parser.add_argument('--dnn_layers', default=5,
                        help='the number of fully connected layers (default: 5)')
    parser.add_argument('--conv_hidden', default=48,
                        help='the number of hidden units on convolution layers (default: 48)')
    parser.add_argument('--pool_hidden', default=50,
                        help='the number of hidden units on pooling layer (default: 50)')
    parser.add_argument('--dnn_hidden', default=45,
                        help='the number of hidden units on fully connected layers (default: 45)')
    parser.add_argument('--conv_act_type', default=2,
                        help='the type of activation function on convolution layers 0:Identity, 1:ReLU, 2:tanH (default: 2)')
    parser.add_argument('--pool_act_type', default=0,
                        help='the type of activation function on pooling layer 0:Identity, 1:ReLU, 2:tanH (default: 0)')
    parser.add_argument('--dnn_act_type', default=2,
                        help='the type of activation function on fully connected layers 0:Identity, 1:ReLU, 2:tanH (default: 2)')
    parser.add_argument('--model_name', default="rgcn_with_reg",
                        help='choose model {"rgcn_with_reg","rgcn_no_reg"} (default: "rgcn_with_reg")')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='set learning_rate (default: 1e-4)')
    parser.add_argument('--dropout', default=0.03,
                        help='set dropout (default: 0.03). when you use a large dataset (e.g., zinc), you can set dropout to 0.0 to increase the capacity of model')
    parser.add_argument('--weight_decay', default=0.07,
                        help='set weight_decay (default: 0.07). when you use a large dataset (e.g., zinc), you can set weight_decay to 1e-5 to increase the capacity of model')
    parser.add_argument('--epochs', default=2000,
                        help='set epochs (default: 2000)')
    parser.add_argument('--ALPHA', default=0.8,
                        help='set ALPHA (default: 0.8)')
    parser.add_argument('--basis_num', default=4,
                        help='the number of basis used in basis regularization (default: 4)')
    parser.add_argument('--inputvec_dim', default=6,
                        help='the number of dimension of preknown global attributes (default: 6)')
    parser.add_argument('--dataset', default="delaney-processed",
                        help='the name of data set {"delaney-processed" or "20k_rndm_zinc_drugs_clean_3"} (default: "delaney-processed")')
    parser.add_argument('--topN', default=None,
                        help='the number of graphs used for doing this experiment in data set, None for all (default: None)')
    parser.add_argument('--initn', default=20,
                        help='initial evaluation times (default: 20)')
    parser.add_argument('--maxiter', default=200,
                        help='the maximal evaluation times (default: 200)')
    parser.add_argument('--resample_period', default=20,
                        help='the period of resampling (default: 20)')
    parser.add_argument('--relearn_NN_period', default=20,
                        help='the period of relearning network (default: 20)')

    args = parser.parse_args()
    
    return args




