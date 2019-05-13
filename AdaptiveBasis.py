#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:52:41 2018

@author: cuijiaxu
"""

import numpy as np
import pylab as pl


def simpleBasis(x):
    if len(x)==1:
        return np.array([1.0,x[0],x[0]*x[0]])
    else:  
        xx=[]
        for i in range(len(x)):
            xx.append(np.array([1.0,x[i,0],x[i,0]*x[i,0]]))
        return np.array(xx)
def AdaptiveBasis(data,info,x,retrain=False,OneTest=False):
    """test 345 start"""
    """
    if info.gcn==True:
        if len(data.candidates)==len(x):
            return rseed345_basis(info)
        else:
            return rseed345_basis(info)[info.observedx]
    """
    """test 345 end"""
    if retrain==True:
        if info.gcn==False: 
            info.dnn.rng=info.rng
            info.dnn.train(data.candidates[info.observedx],np.array(info.observedy).reshape(len(info.observedy),))
            #info.set_w_m0(dnn.get_weight().reshape(dnn.get_weight().shape[1],1))
            basis=info.dnn.get_basis(x)
        else:
            info.dgcn.info=info
            info.dgcn.dataset=info.dataset
            info.dgcn.All_cand_node_num=info.All_cand_node_num
            #basis=dgcn.train_minibatch(data.candidates[info.observedx],np.array(info.observedy).reshape(len(info.observedy),),True)
            basis=info.dgcn.train(data.candidates[info.observedx],np.array(info.observedy).reshape(len(info.observedy),),True)
            #print basis,basis.shape
    else:
        if info.gcn==False:
            basis=info.dnn.get_basis(x)
        else:
            info.dgcn.info=info
            info.dgcn.dataset=info.dataset
            if OneTest==False:
                if len(data.candidates)==len(x):
                    #get all cand
                    basis=info.dgcn.get_basis(x,data.y)
                else:
                    #get part
                    #basis=dgcn.train_minibatch(data.candidates[info.observedx],np.array(info.observedy).reshape(len(info.observedy),),False)
                    basis=info.dgcn.train(data.candidates[info.observedx],np.array(info.observedy).reshape(len(info.observedy),),False)
            else:
                basis=info.dgcn.get_basis_one(x)

            #print basis,basis.shape
    """
    if info.gcn==True:
        #print basis
        np.savetxt("results/basis-RGBODGCN-r%s.txt"%info.rseed, basis, fmt='%s')
        exit(1)
    """
    return basis
    """
    return simpleBasis(x)
    """
def rseed345_basis(info):
    basis=np.loadtxt("results/basis-RGBODGCN-r%s.txt"%info.rseed)
    pl.figure(4)
    pl.matshow(basis.dot(basis.T))
    pl.colorbar()
    pl.title("Similarity matrix.")
    pl.savefig("results/matshow-RGBODGCN-r%s.pdf"%info.rseed)
    #exit(1)
    return basis
  
