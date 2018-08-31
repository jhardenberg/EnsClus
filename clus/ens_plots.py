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
from numpy import linalg as LA
import pickle


def Rcorr(x,y):
    """
    Returns correlation coefficient between two array of arbitrary shape.
    """
    return np.corrcoef(x.flatten(), y.flatten())[1,0]


def E_rms(x,y):
    """
    Returns root mean square deviation: sqrt(1/N sum (xn-yn)**2).
    """
    n = x.size
    #E = np.sqrt(1.0/n * np.sum((x.flatten()-y.flatten())**2))
    E = 1/np.sqrt(n) * LA.norm(x-y)

    return E


def E_rms_cp(x,y):
    """
    Returns centered-pattern root mean square, e.g. first subtracts the mean to the two series and then computes E_rms.
    """
    x1 = x - x.mean()
    y1 = y - y.mean()

    E = E_rms(x1, y1)

    return E


def cosine(x,y):
    """
    Calculates the cosine of the angle between x and y. If x and y are 2D, the scalar product is taken using the np.vdot() function.
    """

    if x.ndim != y.ndim:
        raise ValueError('x and y have different dimension')
    elif x.shape != y.shape:
        raise ValueError('x and y have different shapes')

    if x.ndim == 1:
        return np.dot(x,y)/(LA.norm(x)*LA.norm(y))
    elif x.ndim == 2:
        return np.vdot(x,y)/(LA.norm(x)*LA.norm(y))
    else:
        raise ValueError('Too many dimensions')


def cosine_cp(x,y):
    """
    Before calculating the cosine, subtracts the mean to both x and y. This is exactly the same as calculating the correlation coefficient R.
    """

    x1 = x - x.mean()
    y1 = y - y.mean()

    return cosine(x1,y1)


def ens_plots(inputs, labels, ens_mindist, climatology = None, ensemble_mean = None, observation = None):
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

    # cmappa.set_under('violet')
    # cmappa.set_over('brown')

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
    print('Number of clusters: {}'.format(numclus))

    #____________Reading the netCDF file of N 2Dfields of anomalies, saved by ens_anom.py
    ifile=os.path.join(OUTPUTdir,'ens_anomalies_{}.nc'.format(name_outputs))
    vartoplot, varunits, lat, lon = read_N_2Dfields(ifile)
    print('vartoplot dim: (numens x lat x lon)={}'.format(vartoplot.shape))

    print('Unitssssss',varunits)

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
    ofile = OUTPUTdir + 'Clusters_closest_ensmember_{}.nc'.format(name_outputs)
    print('Saving clustern anomalies (vs model climatology)\n')
    okins = [cos[0] for cos in ens_mindist]
    save_N_2Dfields(lat,lon,vartoplot[okins],'clus_anom_closer',varunits,ofile)

    #____________Load labels
    # namef=os.path.join(OUTPUTdir,'labels_{}.txt'.format(name_outputs))
    # labels=np.loadtxt(namef,dtype=int)
    # print(labels)
    if observation is not None and inputs['fig_ref_to_obs']:
        reference = vartoplot3
        reference_sigma = vartoplot4
    else:
        reference = vartoplot
        reference_sigma = vartoplot2

    mi = np.percentile(reference, 5)
    ma = np.percentile(reference, 95)
    oko = max(abs(mi), abs(ma))
    spi = 2*oko/(n_color_levels-1)
    spi_ok = np.ceil(spi*100)/100
    oko_ok = spi_ok*(n_color_levels-1)/2

    clevels = np.linspace(-oko_ok, oko_ok, n_color_levels)
    #clevels = np.linspace(-math.ceil(oko*100)/100, math.ceil(oko*100)/100, n_color_levels)

    print('levels', len(clevels), min(clevels), max(clevels))

    if inputs['clim_sigma_value'] is not None:
        mi = np.percentile(reference_sigma, 5)
        ma = np.percentile(reference_sigma, 95)
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

    num_figs = int(np.ceil(1.0*numens/inputs['max_ens_in_fig']))
    numens_ok = int(np.ceil(numens/num_figs))

    side1 = int(np.ceil(np.sqrt(numens_ok)))
    side2 = int(np.ceil(numens_ok/float(side1)))

    for i in range(num_figs):
        fig = plt.figure(figsize=(24,14))
        for nens in range(numens_ok*i, numens_ok*(i+1)):
            nens_rel = nens - numens_ok*i
            #print('//////////ENSEMBLE MEMBER {}'.format(nens))
            ax = plt.subplot(side1, side2, nens_rel+1, projection=proj)
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
            # title_obj=plt.title(subtit, fontsize=20, fontweight='bold', loc = 'left')
            title_obj = plt.text(-0.05, 1.05, subtit, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20, fontweight='bold', zorder = 20)
            for nclus in range(numclus):
                if nens in np.where(labels==nclus)[0]:
                    okclus = nclus
                    bbox=dict(facecolor=colors[nclus], alpha = 0.7, edgecolor='black', boxstyle='round,pad=0.2')
                    title_obj.set_bbox(bbox)
                    #title_obj.set_backgroundcolor(colors[nclus])

            if nens == ens_mindist[okclus][0]:
                #print('piruuuuuuuuuuuuuuuuuuuuuuuuuuuuu', okclus, nens)
                #rect = ax.patch
                rect = plt.Rectangle((-0.01,-0.01), 1.02, 1.02, fill = False, transform = ax.transAxes, clip_on = False, zorder = 10)#joinstyle='round')
                rect.set_edgecolor(colors[okclus])
                rect.set_linewidth(6.0)
                ax.add_artist(rect)
                #plt.draw()

        cax = plt.axes([0.1, 0.06, 0.8, 0.03]) #horizontal
        cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')#, labelsize=18)
        cb.ax.tick_params(labelsize=18)
        cb.set_label(inputs['cb_label'], fontsize=20)

        plt.suptitle(varname+' '+tit+' ('+varunits+')', fontsize=35, fontweight='bold')

        plt.subplots_adjust(top=0.85)
        top    = 0.90  # the top of the subplots of the figure
        bottom = 0.13    # the bottom of the subplots of the figure
        left   = 0.02    # the left side of the subplots of the figure
        right  = 0.98  # the right side of the subplots of the figure
        hspace = 0.20   # the amount of height reserved for white space between subplots
        wspace = 0.05    # the amount of width reserved for blank space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        # plot the selected fields

        namef = OUTPUTdir + '{}_{}_{}.{}'.format(field_to_plot, name_outputs, i, inputs['fig_format'])
        fig.savefig(namef)#bbox_inches='tight')

    print('____________________________________________________________________________________________________________________')

    #plt.ion()
    fig2 = plt.figure(figsize=(18,12))
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

        title_obj = plt.title('Cluster {} - {:3.0f}% of cases'.format(clu, (100.0*sum(labels == clu))/numens), fontsize=24, fontweight='bold')
        title_obj.set_position([.5, 1.05])
        bbox=dict(facecolor=colors[clu], alpha = 0.5, edgecolor='black', boxstyle='round,pad=0.3')
        title_obj.set_bbox(bbox)
        #title_obj.set_backgroundcolor(colors[clu])

    cax = plt.axes([0.1, 0.07, 0.8, 0.03]) #horizontal
    cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')#, labelsize=18)
    cb.ax.tick_params(labelsize=22)
    cb.set_label(inputs['cb_label'], fontsize=22)

    top    = 0.92  # the top of the subplots of the figure
    bottom = 0.13    # the bottom of the subplots of the figure
    left   = 0.02    # the left side of the subplots of the figure
    right  = 0.98  # the right side of the subplots of the figure
    hspace = 0.20   # the amount of height reserved for white space between subplots
    wspace = 0.05    # the amount of width reserved for blank space between subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    # plot the selected fields
    namef=os.path.join(OUTPUTdir,'Clusters_{}_{}.{}'.format(field_to_plot,name_outputs, inputs['fig_format']))
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

            title_obj = plt.title('Cluster {} - {:3.0f}% of cases'.format(clu, (100.0*sum(labels == clu))/numens), fontsize=24, fontweight='bold')
            title_obj.set_position([.5, 1.05])
            bbox=dict(facecolor=colors[clu], alpha = 0.5, edgecolor='black', boxstyle='round,pad=0.3')
            title_obj.set_bbox(bbox)
            #title_obj.set_backgroundcolor(colors[clu])

        cax = plt.axes([0.1, 0.07, 0.8, 0.03]) #horizontal
        cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')#, labelsize=18)
        cb.ax.tick_params(labelsize=20)
        cb.set_label(r'# of model $\sigma$', fontsize=22)

        top    = 0.92  # the top of the subplots of the figure
        bottom = 0.13    # the bottom of the subplots of the figure
        left   = 0.02    # the left side of the subplots of the figure
        right  = 0.98  # the right side of the subplots of the figure
        hspace = 0.20   # the amount of height reserved for white space between subplots
        wspace = 0.05    # the amount of width reserved for blank space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        # plot the selected fields
        namef=os.path.join(OUTPUTdir,'Clusters_{}_{}_sigmascale.{}'.format(field_to_plot,name_outputs, inputs['fig_format']))
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
        title_obj.set_position([.5, 1.05])

        cax = plt.axes([0.1, 0.11, 0.8, 0.05]) #horizontal
        cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')#, labelsize=18)
        cb.ax.tick_params(labelsize=14)
        cb.set_label(inputs['cb_label'], fontsize=16)

        top    = 0.88  # the top of the subplots of the figure
        bottom = 0.20    # the bottom of the subplots of the figure
        left   = 0.02    # the left side of the subplots of the figure
        right  = 0.98  # the right side of the subplots of the figure
        hspace = 0.20   # the amount of height reserved for white space between subplots
        wspace = 0.05    # the amount of width reserved for blank space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        # plot the selected fields
        namef=os.path.join(OUTPUTdir,'Observed_anomaly_{}.{}'.format(name_outputs, inputs['fig_format']))
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
        title_obj.set_position([.5, 1.05])

        cax = plt.axes([0.1, 0.11, 0.8, 0.05]) #horizontal
        cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')#, labelsize=18)
        cb.ax.tick_params(labelsize=14)
        cb.set_label(inputs['cb_label'], fontsize=16)
        cb.set_label(r'# of model $\sigma$', fontsize=16)

        top    = 0.88  # the top of the subplots of the figure
        bottom = 0.20    # the bottom of the subplots of the figure
        left   = 0.02    # the left side of the subplots of the figure
        right  = 0.98  # the right side of the subplots of the figure
        hspace = 0.20   # the amount of height reserved for white space between subplots
        wspace = 0.05    # the amount of width reserved for blank space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        # plot the selected fields
        namef=os.path.join(OUTPUTdir,'Observed_anomaly_{}_sigmascale.{}'.format(name_outputs, inputs['fig_format']))
        fig5.savefig(namef)#bbox_inches='tight')

        # Making the Taylor-like graph
        # Polar plot with
        fig6 = plt.figure(figsize=(8,6))
        ax = fig6.add_subplot(111, polar = True)
        plt.title('Taylor diagram: predictions vs observed')

        ax.set_thetamin(0)
        ax.set_thetamax(180)

        sigmas_pred = np.array([np.std(var) for var in vartoplot])
        sigma_obs = np.std(observation)
        corrs_pred = np.array([Rcorr(observation, var) for var in vartoplot])
        colors_all = [colors[clu] for clu in labels]
        angles = np.arccos(corrs_pred)
        print(corrs_pred.max(), corrs_pred.min())

        ax.scatter(angles, sigmas_pred, s = 10, color = colors_all)
        ax.scatter([0.], [sigma_obs], color = 'black', s = 40, clip_on=False)

        repr_ens = []
        for clu in range(numclus):
            repr_ens.append(ens_mindist[clu][0])

        ax.scatter(angles[repr_ens], sigmas_pred[repr_ens], color = colors, edgecolor = 'black', s = 40)

        ok_cos = np.array([-0.99, -0.95, -0.9, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99])
        labgr = ['{:4.2f}'.format(co) for co in ok_cos]
        anggr = np.rad2deg(np.arccos(ok_cos))

        #ax.grid()
        plt.thetagrids(anggr, labels=labgr, frac = 1.1)

        for sig in [1., 2., 3.]:
            circle = plt.Circle((sigma_obs, 0.), sig*sigma_obs, transform=ax.transData._b, fill = False, edgecolor = 'black', linestyle = '--')# color="red", alpha=0.1-0.03*sig)
            ax.add_artist(circle)

        top    = 0.88  # the top of the subplots of the figure
        bottom = 0.02    # the bottom of the subplots of the figure
        left   = 0.02    # the left side of the subplots of the figure
        right  = 0.98  # the right side of the subplots of the figure
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

        namef = OUTPUTdir + 'Taylor_diagram_{}.'.format(name_outputs) + inputs['fig_format']
        fig6.savefig(namef)

        fig7 = plt.figure(figsize=(8,6))
        ax = fig7.add_subplot(111)

        biases = np.array([np.mean(var) - np.mean(observation) for var in vartoplot])
        ctr_patt_RMS = np.array([E_rms_cp(var, observation) for var in vartoplot])
        RMS = np.array([E_rms(var, observation) for var in vartoplot])

        print('----------------------------\n')
        min_cprms = ctr_patt_RMS.argmin()
        print('The member with smallest centered-pattern RMS is member {} of cluster {}\n'.format(min_cprms, labels[min_cprms]))
        print('----------------------------\n')
        min_rms = RMS.argmin()
        print('The member with smallest absolute RMS is member {} of cluster {}\n'.format(min_rms, labels[min_rms]))
        print('----------------------------\n')
        min_bias = np.abs(biases).argmin()
        print('The member with smallest absolute bias is member {} of cluster {}\n'.format(min_bias, labels[min_bias]))
        print('----------------------------\n')
        max_corr = corrs_pred.argmax()
        print('The member with largest correlation coefficient is member {} of cluster {}\n'.format(max_corr, labels[max_corr]))

        ax.scatter(biases, ctr_patt_RMS, color = colors_all, s =10)
        ax.scatter(biases[repr_ens], ctr_patt_RMS[repr_ens], color = colors, edgecolor = 'black', s = 40)

        plt.xlabel('Bias (K)')
        plt.ylabel('Centered-pattern RMS difference [E\'] (K)')

        for sig in [1., 2., 3.]:
            circle = plt.Circle((0., 0.), sig*sigma_obs, fill = False, edgecolor = 'black', linestyle = '--')
            ax.add_artist(circle)

        plt.scatter(0., 0., color = 'black', s = 40)
        #plt.grid()

        # if plt.ylim()[1] < 0.:
        #     ax.set_ylim(ymax=0.)
        # elif plt.ylim()[0] > 0.:
        #     ax.set_ylim(ymin=0.)
        # if plt.xlim()[1] < 0.:
        #     ax.set_xlim(xmax=0.)
        # elif plt.xlim()[0] > 0.:
        #     ax.set_xlim(xmin=0.)

        namef = OUTPUTdir + 'Taylor_bias_vs_cpRMS_{}.'.format(name_outputs) + inputs['fig_format']
        fig7.savefig(namef)

        plt.close('all')

    return


#========================================================

if __name__ == '__main__':
    print('This program is being run by itself')

    print('**************************************************************')
    print('Running {}'.format(sys.argv[0]))
    print('**************************************************************')
    dir_OUTPUT    = sys.argv[1]          # OUTPUT DIRECTORY
    name_outputs  = sys.argv[2]          # name of the outputs
    numclus       = int(sys.argv[3])  # number of clusters
    field_to_plot = sys.argv[4]          #field to plot ('climatologies', 'anomalies', '75th_percentile', 'mean', 'maximum', 'std', 'trend')

    ens_plots(dir_OUTPUT,name_outputs,numclus,field_to_plot)

else:
    print('ens_plots is being imported from another module')
