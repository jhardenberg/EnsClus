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
from read_inputs import read_inputs
import numpy as np
import pandas as pd

# User-defined packages
prog_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,prog_folder+'/clus/')
from ens_anom import ens_anom
from ens_eof_kmeans import ens_eof_kmeans
from ens_plots import ens_plots
from read_netcdf import read3Dncfield
from sel_season_area import sel_season, sel_area


### Reading inputs from input file
if len(sys.argv) > 1:
    file_input = sys.argv[1] # Name of input file (relative path)
else:
    file_input = 'input_CLUStool.in'

keys = 'INPUT_PATH string_name dir_OUTPUT exp_name varname model timestep level season area extreme numclus perc numpcs field_to_plot n_color_levels n_levels draw_contour_lines overwrite_output clim_compare obs_compare climat_file climat_std obs_file cmap cmap_cluster clim_sigma_value cb_label medscope_run medscope_year_pred'
keys = keys.split()
itype = [str, str, str, str, str, str, str, float, str, str, str, int, float, int, str, int, int, bool, bool, bool, bool, str, str, str, str, str, float, str, bool, int]

if len(itype) != len(keys):
    raise RuntimeError('Ill defined input keys in {}'.format(__file__))
itype = dict(zip(keys, itype))

defaults = dict()
defaults['numclus'] = 4 # 4 clusters
defaults['n_color_levels'] = 21
defaults['n_levels'] = 5
defaults['draw_contour_lines'] = False
defaults['field_to_plot'] = 'anomalies'
defaults['overwrite_output'] = False
defaults['run_compare'] = False
defaults['cmap'] = 'RdBu_r'
defaults['cmap_cluster'] = 'nipy_spectral'


inputs = read_inputs(file_input, keys, n_lines = None, itype = itype, defaults = defaults, verbose = True)

if inputs['medscope_run']:
    print('Medscope run active. Setting standard names.\n')
    if inputs['medscope_year_pred'] is None:
        raise ValueError('[medscope_year_pred] not set!')
    season = inputs['season']
    if season in ['DJF', 'MAM', 'DJFM', 'NDJFM', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']:
        month_pred = 'nov'
        name_fold = 'win{}-{}'.format(inputs['medscope_year_pred'], inputs['medscope_year_pred']+1)
    elif season in ['JJA', 'SON', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']:
        month_pred = 'may'
        name_fold = 'sum{}'.format(inputs['medscope_year_pred'])
    inputs['string_name'] = 'spred_{}_{}_ens'.format(inputs['medscope_year_pred'],month_pred)
    print('Setting string_name: '+inputs['string_name']+'\n')

    inputs['exp_name'] = season+'_'+name_fold
    print('Setting exp_name: '+inputs['exp_name']+'\n')

    inputs['climat_file'] = inputs['INPUT_PATH']+'climatology_mean_{}_1993-2016.nc'.format(month_pred)
    print('Setting climat_file: '+inputs['climat_file']+'\n')

    inputs['climat_std'] = inputs['INPUT_PATH']+'climatology_std_{}_1993-2016.nc'.format(month_pred)
    print('Setting climat_std: '+inputs['climat_std']+'\n')


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
fn.sort()
filenames=[os.path.join(inputs['INPUT_PATH'],i) for i in fn]

print('\n***************************INPUT*************')
print('Input file names contain the string: {0}'.format(inputs['string_name']))
print('_____________________________\nARRAY OF {0} INPUT FILES:'.format(len(filenames)))
for i in filenames:
    print(i)
print('_____________________________\n')


inputs['filenames'] = filenames
inputs['numens'] = len(filenames)
inputs['OUTPUTdir'] = OUTPUTdir

#____________Building the name of output files
s = "_"
season = inputs['season']
if season is None:
    sea = ''
else:
    sea = season
seq = (inputs['varname'],inputs['model'],str(inputs['numens'])+'ens',sea,inputs['area'])
name_outputs=s.join(seq)

inputs['name_outputs'] = name_outputs

####################### PRECOMPUTATION #######################################
#____________run ens_anom as a module
varextreme_ens_np, vartimemean_ens, ensemble_mean, dates = ens_anom(inputs)

dates_pdh = pd.to_datetime(dates)

climatology = None
if inputs['clim_compare']:
    if inputs['climat_file'] is None:
        raise ValueError('climat_file not specified')
    var, lat, lon, dates_clim, time_units, var_units = read3Dncfield(inputs['climat_file'])

    if season is not None:
        var_season, dates_season = sel_season(var, dates_clim, season,
        inputs['timestep'])
    else:
        var_season = var
        dates_season = dates_clim

    climatology_tot, _, _ = sel_area(lat, lon, var_season, inputs['area'])
    climatology = np.mean(climatology_tot, axis = 0)

    if inputs['clim_sigma_value'] is None and inputs['climat_std'] is not None:
        var, lat, lon, dates_clim, time_units, var_units = read3Dncfield(inputs['climat_std'])

        if season is not None:
            var_season, dates_season = sel_season(var, dates_clim, season, inputs['timestep'])
        else:
            var_season = var
            dates_season = dates_clim

        climatology_std_tot, _, _ = sel_area(lat, lon, var_season, inputs['area'])

        climatology_std = np.mean(climatology_std_tot, axis = 0)

        csv = np.mean(climatology_std)
        inputs['clim_sigma_value'] = csv

        if csv == 0. or np.isnan(csv):
            raise ValueError('Problems in calculating model sigma')

observation = None
if inputs['obs_compare']:
    if inputs['obs_file'] is None:
        raise ValueError('obs_file not specified')
    var, lat, lon, dates_obs, time_units, var_units = read3Dncfield(inputs['obs_file'])
    ### I need to extract the right year
    dates_obs_pdh = pd.to_datetime(dates_obs)
    if inputs['timestep'] == 'month':
        delta = pd.Timedelta(weeks=1)
    elif inputs['timestep'] == 'day':
        delta = pd.Timedelta(hours=12)
    else:
        raise ValueError('timestep not understood')

    mask = (dates_obs_pdh > dates_pdh[0] - delta) & (dates_obs_pdh < dates_pdh[-1] + delta)
    var = var[mask,:,:]
    dates_obs = dates_obs[mask]

    if season is not None:
        var_season, dates_season = sel_season(var, dates_obs, season, inputs['timestep'])
    else:
        var_season = var
        dates_season = dates_obs
    observation, _, _ = sel_area(lat, lon, var_season, inputs['area'])
    observation = np.mean(observation, axis = 0)


####################### EOF AND K-MEANS ANALYSES #############################
#____________run ens_eof_kmeans as a module
centroids, labels, ens_mindist, ens_maxdist = ens_eof_kmeans(inputs)

####################### PLOT AND SAVE FIGURES ################################
#____________run ens_plots as a module
ens_plots(inputs, labels, ens_mindist, climatology = climatology, ensemble_mean = ensemble_mean, observation = observation)

print('\n>>>>>>>>>>>> ENDED SUCCESSFULLY!! <<<<<<<<<<<<\n')

sys.stdout = original
f.close()
