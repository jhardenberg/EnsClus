#!/usr/bin/env python3

'''
;;#############################################################################
;; Ensemble Clustering Diagnostics
;; Author: Irene Mavilia (ISAC-CNR, Italy)
;; Copernicus C3S 34a lot 2 (MAGIC)
;;#############################################################################
;; Description
;;    Cluster analysis tool based on the k-means algorithm
;;    for ensembles of climate model simulations
;;
;; Modules called: ens_anom.py and ens_eof_kmeans.py
;;
;; Modification history
;;    20170905-A_mavi_ir: stand-alone version of the ESMValTool diagnostic.
;;
;;#############################################################################
'''

# Standard packages
import os
import sys

# CLUStool directory
dir_CLUStool='/home/fabiano/Research/git/EnsClus/clus/'

# User-defined packages
sys.path.insert(0,dir_CLUStool)
from ens_anom import ens_anom
from ens_eof_kmeans import ens_eof_kmeans
from ens_plots import ens_plots

# Information required by the CLUStool:
#-------------------------------about paths------------------------------------------
# Input data directory:
INPUT_PATH='/home/fabiano/DATA/Medscope/seasonal_forecasts/input_par167_1ens/'
# Input file names included the common string:
string = 'spred_2011_nov_ens'

# OUTPUT directory
dir_OUTPUT='/home/fabiano/Research/lavori/MedscopeEnsClus/Winter_2012_4clus/'
if not os.path.exists(dir_OUTPUT):
    os.mkdir(dir_OUTPUT)

par = 167
climat_file = '/home/fabiano/DATA/Medscope/seasonal_forecasts/all_fields_climat_{}.p'.format(par)

#-------------------------------about data-------------------------------------------
# Write only letters or numbers, no punctuation marks!
# If you want to leave the field empty write 'no'
varname='2t'                #variable name in the file
model='Medscope'           #model name ECEARTH31 NCEPNCAR ERAInterim

timestep = 'month' # month, day

numens = 51                   #total number of ensemble members
season = 'DJF'                #seasonal average
area = 'Med'                   #regional average (examples:'EAT':Euro-Atlantic
                            #                           'PNA': Pacific North American
                            #                           'NH': Northern Hemisphere)
                            #                           'Eu': Europe)
kind='hist'                 #hist: historical, scen:scenario
extreme='mean'   #75th_percentile, mean, maximum, std, trend

#---------------------about cluster analysis------------------------------------------
numclus=4              #number of clusters
#Either set perc or numpcs:
perc=80                #cluster analysis is applied on a number of PCs such as they explain
                       #'perc' of total variance
numpcs='no'            #number of PCs

#---------------------about plots------------------------------------------------------
field_to_plot='anomalies'     #field to plot ('climatologies', 'anomalies', '75th_percentile', 'mean', 'maximum', 'std', 'trend')

#____________Building the name of output files
s = "_"
if season is None:
    sea = ''
else:
    sea = season
seq = (varname,model,str(numens)+'ens',sea,area,kind)
name_outputs=s.join(seq)
#print('The name of the output files will be <variable>_{0}.ext'.format(name_outputs))

# Creating the log file in the Log directory
if not os.path.exists(dir_OUTPUT+'Log'):
    os.mkdir(dir_OUTPUT+'Log')
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

f = open(dir_OUTPUT+'Log/Printed_messages.txt', 'w')
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)


#____________Building the array of file names
fn = [i for i in os.listdir(INPUT_PATH) \
    if os.path.isfile(os.path.join(INPUT_PATH,i)) and string in i]
filenames=[os.path.join(INPUT_PATH,i) for i in fn]

print('\n***********************************INPUT***********************************')
print('Input file names contain the string: {0}'.format(string))
print('_____________________________\nARRAY OF {0} INPUT FILES:'.format(len(filenames)))
for i in filenames:
    print(i)
print('_____________________________\n')

####################### PRECOMPUTATION #######################################
#____________run ens_anom as a module
climatology, ensemble_mean = ens_anom(filenames,dir_OUTPUT,name_outputs,varname,numens,season,area,extreme, timestep, climat_file = climat_file)

####################### EOF AND K-MEANS ANALYSES #############################
#____________run ens_eof_kmeans as a module
ens_mindist, ens_maxdist = ens_eof_kmeans(dir_OUTPUT,name_outputs,numens,numpcs,perc,numclus)

####################### PLOT AND SAVE FIGURES ################################
#____________run ens_plots as a module
ens_plots(dir_OUTPUT,name_outputs,numclus,field_to_plot, ens_mindist, climatology = climatology, ensemble_mean = ensemble_mean)

print('\n>>>>>>>>>>>> ENDED SUCCESSFULLY!! <<<<<<<<<<<<\n')

sys.stdout = original
f.close()
