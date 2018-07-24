#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
;;#####################################################
# Modified by Federico Fabiano (ISAC-CNR, Italy)
# 24 July 2018
########################
;; Ensemble Clustering Diagnostics
;; Author: Irene Mavilia (ISAC-CNR, Italy)
;;###########################################################
;; Description
;;    Cluster analysis tool based on the k-means algorithm
;;    for ensembles of climate model simulations
;;
;;#############################################################################
'''

# Standard packages
import os
import sys
import lib_WRtool as lwr

# User-defined packages
sys.path.insert(0, sys.argv[0]+'clus/')
from ens_anom import ens_anom
from ens_eof_kmeans import ens_eof_kmeans
from ens_plots import ens_plots


### Reading inputs from input file
if len(sys.argv) > 1:
    file_input = sys.argv[1] # Name of input file (relative path)
else:
    file_input = 'input_CLUStool.in'

keys = 'INPUT_PATH string_name dir_OUTPUT exp_name varname model timestep level numens season area extreme numclus perc numpcs field_to_plot n_color_levels n_levels draw_contour_lines overwrite_output clim_compare obs_compare climat_file obs_file'
keys = keys.split()
itype = [str, str, str, str, str, str, str, float, int, str, str, str, int, float, int, str, int, int, bool, bool, bool, bool, str, str]

if len(itype) != len(keys):
    raise RuntimeError('Ill defined input keys in {}'.format(__file__))
itype = dict(zip(keys, itype))

defaults = dict()
defaults['numclus'] = 4 # 4 clusters
defaults['numpcs'] = 4 # 4 pcs
defaults['n_color_levels'] = 21
defaults['n_levels'] = 5
defaults['draw_contour_lines'] = False
defaults['field_to_plot'] = 'anomalies'
defaults['overwrite_output'] = False
defaults['run_compare'] = False


inputs = lwr.read_inputs(file_input, keys, n_lines = None, itype = itype, defaults = defaults)

OUTPUTdir=inputs['dir_OUTPUT']+'OUT_'+inputs['model']+'_'+inputs['exp_name']+'/'

# Creating OUTPUT directory
def kill_proc():
    top.destroy()
    raise RuntimeError('exp_name already used. Change the exp_name in the file {}'.format(file_input))
    return

if not os.path.exists(OUTPUTdir):
    os.mkdir(OUTPUTdir)
    print('The output directory {0} is created'.format(OUTPUTdir))
else:
    print('The output directory {0} already exists'.format(OUTPUTdir))
    if inputs['overwrite_output']:
        print('Key [overwrite_output] active, writing files in the output directory {0}'.format(OUTPUTdir))
    else:
        top = Tkinter.Tk()
        top.title = 'Folder already exists'
        txt = Tkinter.Text()
        txt.insert(1.0, "Folder {} already exists. Do you want to write in the same folder? (this will overwrite file with same name) (No will kill the program and let you change exp_name)".format(OUTPUTdir))
        txt.pack()
        B1 = Tkinter.Button(top, text = 'Yes', command = top.destroy)
        B1.pack()
        B2 = Tkinter.Button(top, text = 'No', command = kill_proc)
        B2.pack()
        top.mainloop()

#____________Building the name of output files
s = "_"
if inputs['season'] is None:
    sea = ''
else:
    sea = inputs['season']
seq = (inputs['varname'],inputs['model'],str(inputs['numens'])+'ens',sea,inputs['area'])
name_outputs=s.join(seq)

inputs['name_outputs'] = name_outputs

#print('The name of the output files will be <variable>_{0}.ext'.format(name_outputs))

# Creating the log file
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

f = open(OUTPUTdir+'log_file.log', 'w')
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)

#____________Building the array of file names
fn = [i for i in os.listdir(inputs['INPUT_PATH']) \
    if os.path.isfile(os.path.join(inputs['INPUT_PATH'],i)) and inputs['string_name'] in i]
filenames=[os.path.join(inputs['INPUT_PATH'],i) for i in fn]

print('\n***************************INPUT*************')
print('Input file names contain the string: {0}'.format(inputs['string_name']))
print('_____________________________\nARRAY OF {0} INPUT FILES:'.format(len(filenames)))
for i in filenames:
    print(i)
print('_____________________________\n')


inputs['filenames'] = filenames
inputs['OUTPUTdir'] = OUTPUTdir

####################### PRECOMPUTATION #######################################
#____________run ens_anom as a module
varextreme_ens_np, vartimemean_ens, ensemble_mean = ens_anom(inputs)

climatology = None
if inputs['clim_compare']:
    if climat_file is None:
        raise ValueError('climat_file not specified')
    all_fields, climat_mean, climat_std = pickle.load(open(climat_file, 'rb'))
    climatology = np.mean(climat_mean['nov'][:3,:,:], axis = 0)

    climatology, _, _ = sel_area(lat,lon,climatology,area)

observation = None
if inputs['obs_compare']:
    if obs_file is None:
        raise ValueError('obs_file not specified')
    all_fields, climat_mean, climat_std = pickle.load(open(climat_file, 'rb'))
    observation = np.mean(climat_mean['nov'][:3,:,:], axis = 0)

    observation, _, _ = sel_area(lat,lon, observation,area)


####################### EOF AND K-MEANS ANALYSES #############################
#____________run ens_eof_kmeans as a module
ens_mindist, ens_maxdist = ens_eof_kmeans(inputs)




####################### PLOT AND SAVE FIGURES ################################
#____________run ens_plots as a module
ens_plots(dir_OUTPUT,name_outputs,numclus,field_to_plot, ens_mindist, climatology = climatology, ensemble_mean = ensemble_mean)

print('\n>>>>>>>>>>>> ENDED SUCCESSFULLY!! <<<<<<<<<<<<\n')

sys.stdout = original
f.close()
