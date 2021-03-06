#*********************************
#        ens_eof_kmeans          *
#*********************************

# Standard packages
import numpy as np
import sys
import os
from sklearn.cluster import KMeans
import datetime
import math
import pandas as pd
import collections
from itertools import combinations
from numpy import linalg as LA


def clus_eval_indexes(elements, centroids, labels):
    """
    Computes clustering evaluation indexes, as the Davies-Bouldin Index, the Dunn Index, the optimal variance ratio and the Silhouette value. Also computes cluster sigmas and distances.
    """
    PCs = elements
    ### Computing clustering evaluation Indexes
    numclus = len(centroids)
    inertia_i = np.empty(numclus)
    for i in range(numclus):
        lab_clus = labels == i
        inertia_i[i] = np.sum([np.sum((pcok-centroids[i])**2) for pcok in PCs[lab_clus]])

    clus_eval = dict()
    clus_eval['Indexes'] = dict()

    # Optimal ratio

    n_clus = np.empty(numclus)
    for i in range(numclus):
        n_clus[i] = np.sum(labels == i)

    mean_intra_clus_variance = np.sum(inertia_i)/len(labels)

    dist_couples = dict()
    coppie = list(combinations(range(numclus), 2))
    for (i,j) in coppie:
        dist_couples[(i,j)] = LA.norm(centroids[i]-centroids[j])

    mean_inter_clus_variance = np.sum(np.array(dist_couples.values())**2)/len(coppie)

    clus_eval['Indexes']['Inter-Intra Variance ratio'] = mean_inter_clus_variance/mean_intra_clus_variance

    sigma_clusters = np.sqrt(inertia_i/n_clus)
    clus_eval['Indexes']['Inter-Intra Distance ratio'] = np.mean(dist_couples.values())/np.mean(sigma_clusters)

    # Davies-Bouldin Index
    R_couples = dict()
    for (i,j) in coppie:
        R_couples[(i,j)] = (sigma_clusters[i]+sigma_clusters[j])/dist_couples[(i,j)]

    DBI = 0.
    for i in range(numclus):
        coppie_i = [coup for coup in coppie if i in coup]
        Di = np.max([R_couples[cop] for cop in coppie_i])
        DBI += Di

    DBI /= numclus
    clus_eval['Indexes']['Davies-Bouldin'] = DBI

    # Dunn Index

    Delta_clus = np.empty(numclus)
    for i in range(numclus):
        lab_clus = labels == i
        distances = [LA.norm(pcok-centroids[i]) for pcok in PCs[lab_clus]]
        Delta_clus[i] = np.sum(distances)/n_clus[i]

    clus_eval['Indexes']['Dunn'] = np.min(dist_couples.values())/np.max(Delta_clus)

    clus_eval['Indexes']['Dunn 2'] = np.min(dist_couples.values())/np.max(sigma_clusters)

    # Silhouette
    sils = []
    for ind, el, lab in zip(range(len(PCs)), PCs, labels):
        lab_clus = labels == lab
        lab_clus[ind] = False
        ok_Pcs = PCs[lab_clus]
        a = np.sum([LA.norm(okpc - el) for okpc in ok_Pcs])/n_clus[lab]

        bs = []
        others = range(numclus)
        others.remove(lab)
        for lab_b in others:
            lab_clus = labels == lab_b
            ok_Pcs = PCs[lab_clus]
            b = np.sum([LA.norm(okpc - el) for okpc in ok_Pcs])/n_clus[lab_b]
            bs.append(b)

        b = np.min(bs)
        sils.append((b-a)/max([a,b]))

    sils = np.array(sils)
    sil_clus = []
    for i in range(numclus):
        lab_clus = labels == i
        popo = np.sum(sils[lab_clus])/n_clus[i]
        sil_clus.append(popo)

    siltot = np.sum(sil_clus)/numclus

    clus_eval['Indexes']['Silhouette'] = siltot
    clus_eval['clus_silhouettes'] = sil_clus

    clus_eval['Indexes']['Dunn2/DB'] = clus_eval['Indexes']['Dunn 2']/clus_eval['Indexes']['Davies-Bouldin']

    clus_eval['R couples'] = R_couples
    clus_eval['Inter cluster distances'] = dist_couples
    clus_eval['Sigma clusters'] = sigma_clusters

    return clus_eval


def ens_eof_kmeans(inputs):
    '''
    Find the most representative ensemble member for each cluster.
    METHODS:
    - Empirical Orthogonal Function (EOF) analysis of the input file
    - K-means cluster analysis applied to the retained Principal Components (PCs)

    TODO:
    - Order clusters per frequency
    - Give the anomalies in input (not from file)

    '''

    # User-defined libraries
    from read_netcdf import read_N_2Dfields
    from eof_tool import eof_computation

    OUTPUTdir = inputs['OUTPUTdir']
    numens = inputs['numens']
    name_outputs = inputs['name_outputs']
    filenames = inputs['filenames']
    numpcs = inputs['numpcs']
    perc = inputs['perc']
    numclus = inputs['numclus']

    # Either perc (cluster analysis is applied on a number of PCs such as they explain
    # 'perc' of total variance) or numpcs (number of PCs to retain) is set:
    if numpcs is not None:
        print('Number of principal components: {0}'.format(numpcs))

    if perc is not None:
        print('Percentage of explained variance: {0}%'.format(int(perc)))

    if (perc is None and numpcs is None) or (perc is not None and numpcs is not None):
        raise ValueError('You have to specify either "perc" or "numpcs".')

    print('Number of clusters: {0}'.format(numclus))

    #____________Reading the netCDF file of N 2Dfields of anomalies, saved by ens_anom.py
    ifile=os.path.join(OUTPUTdir,'ens_anomalies_{0}.nc'.format(name_outputs))
    var, varunits, lat, lon = read_N_2Dfields(ifile)
    print('var dim: (numens x lat x lon)={0}'.format(var.shape))


    #____________Compute EOFs (Empirical Orthogonal Functions)
    #____________and PCs (Principal Components) with respect to ensemble memeber
    print('____________________________________________________________________________________________________________________')
    print('EOF analysis')
    #----------------------------------------------------------------------------------------
    solver, pcs_scal1, eofs_scal2, pcs_unscal0, eofs_unscal0, varfrac = eof_computation(var,varunits,lat,lon)

    acc=np.cumsum(varfrac*100)
    if perc is not None:
        # Find how many PCs explain a certain percentage of variance
        # (find the mode relative to the percentage closest to perc, but bigger than perc)
        numpcs=min(enumerate(acc), key=lambda x: x[1]<=perc)[0]+1
        print('\nThe number of PCs that explain the percentage closest to {0}% of variance (but grater than {0}%) is {1}'.format(perc,numpcs))
        exctperc=min(enumerate(acc), key=lambda x: x[1]<=perc)[1]
    if numpcs is not None:
        exctperc=acc[numpcs-1]
    if np.isnan(exctperc):
        print(acc)
        raise ValueError('NaN in evaluation of variance explained by first pcs')
    print('(the first {0} PCs explain exactly the {1}% of variance)'.format(numpcs,"%.2f" %exctperc))


    #____________Compute k-means analysis using a subset of PCs
    print('__________________________________________________\n')
    print('k-means analysis using a subset of PCs')
    print('_____________________________________________\n')
    #----------------------------------------------------------------------------------------
    PCs=pcs_unscal0[:,:numpcs]

    clus=KMeans(n_clusters=numclus, n_init=600, max_iter=1000)

    start = datetime.datetime.now()
    clus.fit(PCs)
    end = datetime.datetime.now()
    print('k-means algorithm took me %s seconds' %(end-start))

    centroids=clus.cluster_centers_          # shape---> (numclus,numpcs)
    labels=clus.labels_                      # shape---> (numens,)
    inertia = clus.inertia_

    ## Ordering clusters for number of members
    centroids = np.array(centroids)
    labels = np.array(labels)

    num_mem = []
    for i in range(numclus):
        num_mem.append(np.sum(labels == i))
    num_mem = np.array(num_mem)

    new_ord = num_mem.argsort()[::-1]
    centroids = centroids[new_ord]

    labels_new = np.array(labels)
    for nu, i in zip(range(numclus), new_ord):
        labels_new[labels == i] = nu
    labels = labels_new
    ###
    clus_eval = clus_eval_indexes(PCs, centroids, labels)
    for nam in clus_eval['Indexes'].keys():
        print(nam, clus_eval['Indexes'][nam])

    print('\nClusters are identified for {0} PCs (explained variance {1}%)'.format(numpcs, "%.2f" %exctperc))
    print('PCs dim: (number of ensemble members, number of PCs)={0}, EOF dim: (number of ensemble members, lat, lon)={1}'.format(pcs_unscal0[:,:numpcs].shape,eofs_unscal0[:numpcs].shape))
    print('Centroid coordinates dim: (number of clusters, number of PCs)={0}, labels dim: (number of ensemble members,)={1}\n'.format(centroids.shape,labels.shape))

    #____________Save labels
    namef=os.path.join(OUTPUTdir,'labels_{0}.txt'.format(name_outputs))
    #np.savetxt(namef,labels,fmt='%d')
    filo = open(namef, 'w')
    stringo = '{:6s} {:20s} {:8s}\n'.format('#', 'filename', 'cluster')
    filo.write(stringo)
    filo.write(' \n')
    for filnam, ii, lab in zip(inputs['filenames'], range(numens), labels):
        indr = filnam.rindex('/')
        filnam = filnam[indr+1:]
        stringo = '{:6d} {:20s} {:8d}\n'.format(ii, filnam, lab)
        filo.write(stringo)
    filo.close()

    #____________Compute cluster frequencies
    L=[]
    for nclus in range(numclus):
        cl=list(np.where(labels==nclus)[0])
        fr=len(cl)*100/len(labels)
        L.append([nclus,fr,cl])
    print('Cluster labels:')
    print([L[ncl][0] for ncl in range(numclus)])
    print('Cluster frequencies (%):')
    print([round(L[ncl][1],3) for ncl in range(numclus)])
    print('Cluster members:')
    print([L[ncl][2] for ncl in range(numclus)])

    #____________Find the most representative ensemble member for each cluster
    print('____________________________________________________________________________________________________________________')
    print('In order to find the most representative ensemble member for each cluster\n(which is the closest member to the cluster centroid)')
    print('the Euclidean distance between cluster centroids and each ensemble member is computed in the PC space')
    print('____________________________________________________________________________________________________________________')
    # 1)
    print('Check: cluster #1 centroid coordinates vector dim {0} should be the same as the member #1 PC vector dim {1}\n'.format(centroids[1,:].shape,PCs[1,:].shape))
    #print('\nIn the PC space, the distance between:')
    norm=np.empty([numclus,numens])
    finalOUTPUT=[]
    repres=[]

    ens_mindist = []
    ens_maxdist = []
    for nclus in range(numclus):
        for ens in range(numens):
            normens=centroids[nclus,:]-PCs[ens,:]
            norm[nclus,ens]=math.sqrt(sum(normens**2))
            #print('The distance between centroid of cluster {0} and member {1} is {2}'.format(nclus,ens,round(norm[nclus,ens],3)))
        print('The distances between centroid of cluster {0} and member #0 to #{1} are:\n{2}'.format(nclus,numens-1,np.round(norm[nclus],3)))

        ens_mindist.append((np.argmin(norm[nclus,:]), norm[nclus].min()))

        print('MINIMUM DISTANCE FOR CLUSTER {0} IS {1} --> member #{2}'.format(nclus, round(ens_mindist[-1][1],3), ens_mindist[-1][0]))

        repres.append(np.where(norm[nclus] == norm[nclus].min())[0][0])

        ens_maxdist.append((np.argmax(norm[nclus,:]), norm[nclus].max()))

        print('MAXIMUM DISTANCE FOR CLUSTER {0} IS {1} --> member #{2}'.format(nclus, round(ens_maxdist[-1][1],3), ens_maxdist[-1][0]))

        txt='Closest ensemble member/members to centroid of cluster {0} is/are {1}\n'.format(nclus,list(np.where(norm[nclus] == norm[nclus].min())[0]))
        finalOUTPUT.append(txt)
    with open(OUTPUTdir+'RepresentativeEnsembleMembers_{0}.txt'.format(name_outputs), "w") as text_file:
        text_file.write(''.join(str(e) for e in finalOUTPUT))

    #____________Save the most representative ensemble members
    namef=os.path.join(OUTPUTdir,'repr_ens_{0}.txt'.format(name_outputs))
    filo = open(namef, 'w')
    filo.write('List of cluster representatives\n')
    stringo = '{:10s} {:8s} -> {:20s}\n'.format('', '#', 'filename')
    filo.write(stringo)
    filo.write(' \n')
    for ii in range(numclus):
        okin = repres[ii]
        filnam = inputs['filenames'][okin]
        indr = filnam.rindex('/')
        filnam = filnam[indr+1:]
        stringo = 'Cluster {:2d}: {:8d} -> {:20s}\n'.format(ii, okin, filnam)
        filo.write(stringo)
    filo.close()
    #np.savetxt(namef,repres,fmt='%i')


    print('____________________________________________________________________________________________________________________')
    print('In order to study the spread of each cluster,')
    print('the standard deviation of the distances between each member in a cluster and the cluster centroid is computed in the PC space')
    print('____________________________________________________________________________________________________________________')
    print('\nIn the PC space:')
    statOUTPUT=[]
    for nclus in range(numclus):
        members=L[nclus][2]
        norm=np.empty([numclus,len(members)])
        for mem in range(len(members)):
            #print('mem=',mem)
            ens=members[mem]
            #print('ens',ens)
            normens=centroids[nclus,:]-PCs[ens,:]
            norm[nclus,mem]=math.sqrt(sum(normens**2))
            #print('norm=',norm[nclus],norm.dtype)
        print('the distances between centroid of cluster {0} and its belonging members {1} are:\n{2}'.format(nclus,members,np.round(norm[nclus],3)))
        print('MINIMUM DISTANCE WITHIN CLUSTER {0} IS {1} --> member #{2}'.format(nclus,round(norm[nclus].min(),3),members[np.where(norm[nclus] == norm[nclus].min())[0][0]]))
        print('MAXIMUM DISTANCE WITHIN CLUSTER {0} IS {1} --> member #{2}'.format(nclus,round(norm[nclus].max(),3),members[np.where(norm[nclus] == norm[nclus].max())[0][0]]))
        print('INTRA-CLUSTER STANDARD DEVIATION FOR CLUSTER {0} IS {1}\n'.format(nclus,norm[nclus].std()))

        d_stat=collections.OrderedDict()
        d_stat['cluster']=nclus
        d_stat['member']=members
        d_stat['d_to_centroid']=np.round(norm[nclus],3)
        d_stat['intra-clus_std']=norm[nclus].std()
        d_stat['d_min']=round(norm[nclus].min(),3)
        d_stat['d_max']=round(norm[nclus].max(),3)
        d_stat['freq(%)']=round(L[nclus][1],3)
        stat=pd.DataFrame(d_stat)
        statOUTPUT.append(stat)
    statOUTPUT = pd.concat(statOUTPUT, axis=0)
    #____________Save statistics of cluster analysis
    namef=os.path.join(OUTPUTdir,'statistics_clutering_{0}.txt'.format(name_outputs))
    with open(namef, 'w') as text_file:
        text_file.write(statOUTPUT.__repr__())

    return centroids, labels, ens_mindist, ens_maxdist, clus_eval


#========================================================

# if __name__ == '__main__':
#     print('This program is being run by itself')
#
#     print('**************************************************************')
#     print('Running {0}'.format(sys.argv[0]))
#     print('**************************************************************')
#     dir_OUTPUT    = sys.argv[1]  # OUTPUT DIRECTORY
#     name_outputs  = sys.argv[2]  # name of the outputs
#     numens        = int(sys.argv[3])  # number of ensemble members
#     numpcs        = sys.argv[4]  # number of retained PCs
#     perc          = sys.argv[5]  # percentage of explained variance by PCs
#     numclus       = int(sys.argv[6])  # number of clusters
#
#     ens_eof_kmeans(dir_OUTPUT,name_outputs,numens,numpcs,perc,numclus)
#
# else:
#     print('ens_eof_kmeans is being imported from another module')
