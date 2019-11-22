#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:27:24 2018

@author: km
"""
import networkx as nx
import numpy as np
from scipy.io import loadmat








import networkx as nx
import numpy as np
import math
import scipy
from collections import defaultdict

def read_file(filename):
    G=nx.read_weighted_edgelist(filename,encoding="utf-8")
    return G

def matrix_conversion(Graph):
    return nx.to_numpy_matrix(Graph)

def lookup_table_creation(Adjmat):
    num_of_nodes,_=Adjmat.shape
    weight_sum=0
    weights_dict={}
    for i in xrange(num_of_nodes):
        weight_sum=np.sum(Adjmat[i])
        weights_dict[i]=weight_sum
    return weights_dict
def node_list_lookup(table,Adjmat,Graph):
    row,col=Adjmat.shape
    
    for i in xrange(0,row):
        node_list1=[]
        #nodelist=[]
        node_list1+=list(Graph.neighbors((i)))
        nodelist=[int(x) for x in node_list1]
        table[i].extend(nodelist)
        
def node_list_lookupwt(table,Adjmat,G):
    row,col=Adjmat.shape
    
    for j in [1]: #range(row):
            print j
            node1=[]

            node1+=G.neighbors(j)
            print node1
            if not node1:
                table[j].extend([])
                pass
            else:
                size=len(node1)
                scores=[0]*size
                for i in xrange(len(node1)):
                    num=len(sorted(nx.common_neighbors(G,j,node1[i])))
                    scores[i]=float(num+1)/float(len(G.neighbors(node1[i]))+1)
                X=[x for (y,x) in sorted(zip(scores,node1))]
                cutoff=int(cs*len(X))
                del X[0:cutoff]
                
                X11=set(X)
                size=len(X)
            # print 'inside'+str(i)
                scores=[0]*size
                X=list(X)
                for ii in xrange(len(X)):
                    t=set(G.neighbors(X[ii]))
                    num=X11 & t
                #num=len(sorted(nx.common_neighbors(g,node,new[i])))
                #scores[i]=float(num)/float(len(g.neighbors(new[i])))
                    scores[ii]=float(len(num)+1)/float(len(X)+1)
                # print 'score'+str((scores))
                X1=[x for (y,x) in sorted(zip(scores,X))]
                cutoff=int(cs*len(X1))
                del X1[0:cutoff]
                print "1st order neighbor with start node:"+str(j)+"------"+"\n"
                print  X1
                print "\n"
                
                table[j].extend(X1)
        
        
        
        
def addition(original,node,row):

    temp=[]
    #marked=[0]*3890
    #new=[]
    tt1=original
    original=set(original)
    original=list(original)

    for i in xrange(len(original)):
       new=[]
       if (marked[int(original[i])]==0):
            marked[int(original[i])]=1
            #print i

            # print 'original'+str(original[i])
            new+=node_lookup_table[int(original[i])]
            # print 'inside'+str(i)
            X=new
            cutoff=int(float(cs)*len(X))
            del X[0:cutoff]
            temp+=X
       else:
            #print "Already marked 1"
            continue
    tt1+=temp
    return (tt1)
        

blog_cat = loadmat('/home/rcciit/Documents/sourin/youtube_sampled_unweighted2.mat')
trainp1 = blog_cat['trainp1'][0]
trainp2 = blog_cat['trainp2'][0]
testp1 = blog_cat['testp1'][0]
testp2 = blog_cat['testp2'][0]
max_list = []
max_list.append(max(trainp1))
max_list.append(max(trainp2))
max_list.append(max(testp1))
max_list.append(max(testp2))
total_nodes = max(max_list) + 1
ad_mat = np.zeros([total_nodes, total_nodes], dtype = 'int')
ad_mat[trainp1, trainp2] = 1
ad_mat[trainp2, trainp1] = 1
G=nx.from_numpy_matrix(ad_mat)
cs=0.6
#G=nx.from_numpy_matrix(ra_mat)
row,col=ad_mat.shape
node_lookup_table=defaultdict(list)
node_list_lookupwt(node_lookup_table,ad_mat,G)
#print node_lookup_table[0]

global marked
list3,original,original1,original2,original3,original4=[],[],[],[],[],[]
marked=[0]*row
#print Adj_mat[104,200]
sourcelistf=[]
list3,marked,original,original1,original2,original3,original4=[],[0]*row,[],[],[],[],[]
#print Adj_mat[104,200]
for j in range(row):
    print j
    node1=[]
    node1+=node_lookup_table[j]
    #print "Nodelist with networkx functionality -------"+str(list(G.neighbors(str(j))))+"\n"
    #print "Nodelist for--"+str(j)+"---------"+str(node1)+"\n"
    if not node1:
        list3.append(node1)
        pass
    else:
        marked=[0]*row
        marked[j]=1
        original=[]
        original.append(j)
#        print weights
           
        X=node1
        
        cutoff=int(cs*len(node1))
        del X[0:cutoff]
        #X=[x.encode('utf-8') for x in X]
        
        original+=X
#        print "first"+ str(len(original) )
        original1=addition(original,j,row)
#        print "After walk 1"
#        print list(set(original1))
#        print len(list(set(original1)))
#        print "\n"
        original2=addition(original1,j,row)
#        print "after walk 2"
#        print list(set(original2))
#        print len(list(set(original2)))
#        print "\n"
        original3=addition(original2,j,row)
#        print "after walk 3"
#        print list(set(original3))
#        print len(list(set(original3)))
#        print "\n"
        original4=addition(original3,j,row)
#        print "after walk 4"
#        print list(set(original4))
#        print len(list(set(original4)))
#        print "\n"
#        original5=addition(original4,j,row)
#        original6=addition(original5,j,row)
#        original7=addition(original6,j,row)
#        original8=addition(original7,j,row)
#            #original9=addition(original8,j,row)
#            #original9=list(set(original9))
        list3.append(original4)
        inputs = []
for i in range(len(list3)):
    for k in range(len(list3[i])):
        inputs.append(list3[i][k])
