#*********************************
#           ens_plots            *
#*********************************

# Standard packages
import os
import sys
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
#from mpl_toolkits.basemap import Basemap
import math

def ens_plots(inputs, ens_mindist, climatology = None, ensemble_mean = None, observation = None):
    '''
    \nGOAL:
    Plot the chosen field for each ensemble
    NOTE:
    '''
    # User-defined libraries
    import matplotlib.path as mpath
    from read_netcdf import read_N_2Dfields, save_N_2Dfields

    cmappa = cm.get_cmap(inputs['cmap'])
    cmappa_clus = cm.get_cmap(inputs['cmap_cluster'])

    OUTPUTdir = inputs['OUTPUTdir']
    numens = inputs['numens']
    name_outputs = inputs['name_outputs']
    filenames = inputs['filenames']
    numpcs = inputs['numpcs']
    perc = inputs['perc']
    numclus = inputs['numclus']
    varname = inputs['varname']
    field_to_plot = inputs['field_to_plot']

    n_color_levels = inputs['n_color_levels']
    n_levels = inputs['n_levels']
    draw_contour_lines = inputs['draw_contour_lines']

    tit=field_to_plot
    print('Number of clusters: {0}'.format(numclus))

    #____________Reading the netCDF file of N 2Dfields of anomalies, saved by ens_anom.py
    ifile=os.path.join(OUTPUTdir,'ens_anomalies_{0}.nc'.format(name_outputs))
    vartoplot, varunits, lat, lon = read_N_2Dfields(ifile)
    print('vartoplot dim: (numens x lat x lon)={0}'.format(vartoplot.shape))

    print(vartoplot.shape)
    print(ensemble_mean.shape)

    if observation is not None:
        print(observation.shape)
        print('Plotting differences between the observation  and the model climatology\n')
        vartoplot3 = observation
    if climatology is not None:
        print(climatology.shape)
        print('Plotting differences with the model climatology instead that with the ensemble mean\n')
        vartoplot_new = []
        for var in vartoplot:
            vartoplot_new.append(var + ensemble_mean - climatology)
        vartoplot = np.array(vartoplot_new)

    if inputs['clim_sigma_value'] is not None:
        print('Plotting anomalies in term of the model sigma {}\n'.format(inputs['clim_sigma_value']))
        vartoplot2 = vartoplot/inputs['clim_sigma_value']

        if observation is not None:
            vartoplot4 = observation/inputs['clim_sigma_value']

    # print(vartoplot2.shape)
    # print(vartoplot.shape)
    ofile = OUTPUTdir + 'Clusters_closest_ensmember.nc'
    print('Saving clustern anomalies (vs model climatology)\n')
    okins = [cos[0] for cos in ens_mindist]
    save_N_2Dfields(lat,lon,vartoplot[okins],'clus_anom_closer',varunits,ofile)

    #____________Load labels
    namef=os.path.join(OUTPUTdir,'labels_{0}.txt'.format(name_outputs))
    labels=np.loadtxt(namef,dtype=int)
    print(labels)

    mi = np.percentile(vartoplot, 5)
    ma = np.percentile(vartoplot, 95)
    oko = max(abs(mi), abs(ma))
    clevels = np.linspace(-math.ceil(oko*100)/100, math.ceil(oko*100)/100, n_color_levels)

    print('levels', len(clevels), min(clevels), max(clevels))

    if inputs['clim_sigma_value'] is not None:
        mi = np.percentile(vartoplot4, 5)
        ma = np.percentile(vartoplot4, 95)
        oko = max(abs(mi), abs(ma))
        clevels_sigma = np.linspace(-math.ceil(oko*100)/100, math.ceil(oko*100)/100, n_color_levels)

    colors = []
    for cos in np.linspace(0.05,0.95,numclus):
        colors.append(cmappa_clus(cos))
    #colors = ['b','g','r','c','m','y','DarkOrange','grey']

    clat=lat.min()+abs(lat.max()-lat.min())/2
    clon=lon.min()+abs(lon.max()-lon.min())/2

    boundary = np.array([[lat.min(),lon.min()], [lat.max(),lon.min()], [lat.max(),lon.max()], [lat.min(),lon.max()]])
    bound = mpath.Path(boundary)


    proj = ccrs.PlateCarree()

    side1 = int(np.ceil(np.sqrt(numens)))
    side2 = int(np.ceil(numens/float(side1)))

    fig = plt.figure(figsize=(24,14))
    for nens in range(numens):
        #print('//////////ENSEMBLE MEMBER {0}'.format(nens))
        ax = plt.subplot(side1, side2, nens+1, projection=proj)
        ax.set_global()
        ax.coastlines()

        # use meshgrid to create 2D arrays
        xi,yi=np.meshgrid(lon,lat)

        # Plot Data
        if field_to_plot=='anomalies':
            map_plot = ax.contourf(xi,yi,vartoplot[nens],clevels,cmap=cmappa, transform = proj, extend = 'both')
        else:
            map_plot = ax.contourf(xi,yi,vartoplot[nens],clevels, transform = proj, extend = 'both')

        latlonlim = [lon.min(), lon.max(), lat.min(), lat.max()]
        ax.set_extent(latlonlim, crs = proj)

        # Add Title
        subtit = nens
        title_obj=plt.title(subtit, fontsize=32, fontweight='bold')
        for nclus in range(numclus):
            if nens in np.where(labels==nclus)[0]:
                bbox=dict(facecolor=colors[nclus], alpha = 0.5, edgecolor='black', boxstyle='round,pad=0.4')
                title_obj.set_bbox(bbox)
                #title_obj.set_backgroundcolor(colors[nclus])

    cax = plt.axes([0.1, 0.03, 0.8, 0.03]) #horizontal
    cb=plt.colorbar(map_plot,cax=cax, orientation='horizontal')#, labelsize=18)
    cb.ax.tick_params(labelsize=18)
    cb.set_label(inputs['cb_label'], fontsize=20)

    plt.suptitle(varname+' '+tit+' ('+varunits+')', fontsize=45, fontweight='bold')

    plt.subplots_adjust(top=0.85)
    top    = 0.89  # the top of the subplots of the figure
    bottom = 0.12    # the bottom of the subplots of the figure
    left   = 0.02    # the left side of the subplots of the figure
    right  = 0.98  # the right side of the subplots of the figure
    hspace = 0.36   # the amount of height reserved for white space between subplots
    wspace = 0.14    # the amount of width reserved for blank space between subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    # plot the selected fields
    namef=os.path.join(OUTPUTdir,'{0}_{1}.eps'.format(field_to_plot,name_outputs))
    fig.savefig(namef)#bbox_inches='tight')
    print('An eps figure for the selected fields is saved in {0}'.format(OUTPUTdir))
    print('____________________________________________________________________________________________________________________')

    plt.ion()
    fig2 = plt.figure(figsize=(16,12))
    print(numclus)
    side1 = int(np.ceil(np.sqrt(numclus)))
    side2 = int(np.ceil(numclus/float(side1)))
    print(side1,side2,numclus)

    for clu in range(numclus):
        ax = plt.subplot(side1, side2, clu+1, projection=proj)
        ok_ens = ens_mindist[clu][0]

        ax.set_global()
        ax.coastlines(linewidth = 2)
        xi,yi=np.meshgrid(lon,lat)

        map_plot = ax.contourf(xi,yi,vartoplot[ok_ens],clevels,cmap=cmappa, transform = proj, extend = 'both')
        if draw_contour_lines:
            map_plot_lines = ax.contour(xi,yi,vartoplot[ok_ens], n_levels, colors = 'k', transform = proj, linewidth = 0.5)

        latlonlim = [lon.min(), lon.max(), lat.min(), lat.max()]
        ax.set_extent(latlonlim, crs = proj)
        # proj_to_data = proj._as_mpl_transform(ax) - ax.transData
        # rect_in_target = proj_to_data.transform_path(bound)
        # ax.set_boundary(rect_in_target)

        title_obj=plt.title('Cluster {} - {:3.0f}% of cases'.format(clu, (100.0*sum(labels == clu))/numens), fontsize=20, fontweight='bold')
        bbox=dict(facecolor=colors[clu], alpha = 0.5, edgecolor='black', boxstyle='round,pad=0.4')
        title_obj.set_bbox(bbox)
        #title_obj.set_backgroundcolor(colors[clu])

    cax = plt.axes([0.1, 0.05, 0.8, 0.05]) #horizontal
    cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')#, labelsize=18)
    cb.ax.tick_params(labelsize=18)
    cb.set_label(inputs['cb_label'], fontsize=20)

    # plot the selected fields
    namef=os.path.join(OUTPUTdir,'Clusters_{0}_{1}.eps'.format(field_to_plot,name_outputs))
    fig2.savefig(namef)#bbox_inches='tight')

################## OBSERVATIONSSSSSSSSSSSSSS

    if inputs['clim_sigma_value'] is not None:
        fig3 = plt.figure(figsize=(16,12))
        side1 = int(np.ceil(np.sqrt(numclus)))
        side2 = int(np.ceil(numclus/float(side1)))

        for clu in range(numclus):
            ax = plt.subplot(side1, side2, clu+1, projection=proj)
            ok_ens = ens_mindist[clu][0]

            ax.set_global()
            ax.coastlines(linewidth = 2)
            xi,yi=np.meshgrid(lon,lat)

            map_plot = ax.contourf(xi,yi,vartoplot2[ok_ens],clevels_sigma,cmap=cmappa, transform = proj, extend = 'both')
            if draw_contour_lines:
                map_plot_lines = ax.contour(xi,yi,vartoplot2[ok_ens], n_levels, colors = 'k', transform = proj, linewidth = 0.5)

            latlonlim = [lon.min(), lon.max(), lat.min(), lat.max()]
            ax.set_extent(latlonlim, crs = proj)
            # proj_to_data = proj._as_mpl_transform(ax) - ax.transData
            # rect_in_target = proj_to_data.transform_path(bound)
            # ax.set_boundary(rect_in_target)

            title_obj=plt.title('Cluster {} - {:3.0f}% of cases'.format(clu, (100.0*sum(labels == clu))/numens), fontsize=20, fontweight='bold')
            bbox=dict(facecolor=colors[clu], alpha = 0.5, edgecolor='black', boxstyle='round,pad=0.4')
            title_obj.set_bbox(bbox)
            #title_obj.set_backgroundcolor(colors[clu])

        cax = plt.axes([0.1, 0.05, 0.8, 0.05]) #horizontal
        cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')#, labelsize=18)
        cb.ax.tick_params(labelsize=18)
        cb.set_label(r'# of model $\sigma$', fontsize=20)

        # plot the selected fields
        namef=os.path.join(OUTPUTdir,'Clusters_{0}_{1}_sigmascale.eps'.format(field_to_plot,name_outputs))
        fig3.savefig(namef)

        ################# Observations vs climatology
    if observation is not None:
        fig4 = plt.figure(figsize=(8,6))
        ax = plt.subplot(projection=proj)

        ax.set_global()
        ax.coastlines(linewidth = 2)
        xi,yi=np.meshgrid(lon,lat)

        map_plot = ax.contourf(xi,yi,vartoplot3,clevels,cmap=cmappa, transform = proj, extend = 'both')
        if draw_contour_lines:
            map_plot_lines = ax.contour(xi,yi,vartoplot3, n_levels, colors = 'k', transform = proj, linewidth = 0.5)

        latlonlim = [lon.min(), lon.max(), lat.min(), lat.max()]
        ax.set_extent(latlonlim, crs = proj)

        title_obj=plt.title('Observed anomaly', fontsize=20, fontweight='bold')

        cax = plt.axes([0.1, 0.05, 0.8, 0.05]) #horizontal
        cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')#, labelsize=18)
        cb.ax.tick_params(labelsize=18)
        cb.set_label(inputs['cb_label'], fontsize=20)

        # plot the selected fields
        namef=os.path.join(OUTPUTdir,'Observed_anomaly_{}.eps'.format(name_outputs))
        fig4.savefig(namef)#bbox_inches='tight')


        fig5 = plt.figure(figsize=(8,6))
        ax = plt.subplot(projection=proj)

        ax.set_global()
        ax.coastlines(linewidth = 2)
        xi,yi=np.meshgrid(lon,lat)

        map_plot = ax.contourf(xi,yi,vartoplot4,clevels_sigma,cmap=cmappa, transform = proj, extend = 'both')
        if draw_contour_lines:
            map_plot_lines = ax.contour(xi,yi,vartoplot4, n_levels, colors = 'k', transform = proj, linewidth = 0.5)

        latlonlim = [lon.min(), lon.max(), lat.min(), lat.max()]
        ax.set_extent(latlonlim, crs = proj)

        title_obj=plt.title('Observed anomaly', fontsize=20, fontweight='bold')

        cax = plt.axes([0.1, 0.05, 0.8, 0.05]) #horizontal
        cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')#, labelsize=18)
        cb.ax.tick_params(labelsize=18)
        cb.set_label(r'# of model $\sigma$', fontsize=20)

        # plot the selected fields
        namef=os.path.join(OUTPUTdir,'Observed_anomaly_{}_sigmascale.eps'.format(name_outputs))
        fig5.savefig(namef)#bbox_inches='tight')

    return


#========================================================

if __name__ == '__main__':
    print('This program is being run by itself')

    print('**************************************************************')
    print('Running {0}'.format(sys.argv[0]))
    print('**************************************************************')
    dir_OUTPUT    = sys.argv[1]          # OUTPUT DIRECTORY
    name_outputs  = sys.argv[2]          # name of the outputs
    numclus       = int(sys.argv[3])  # number of clusters
    field_to_plot = sys.argv[4]          #field to plot ('climatologies', 'anomalies', '75th_percentile', 'mean', 'maximum', 'std', 'trend')

    ens_plots(dir_OUTPUT,name_outputs,numclus,field_to_plot)

else:
    print('ens_plots is being imported from another module')
