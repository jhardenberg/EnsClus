# Standard packages
from netCDF4 import Dataset, num2date, date2num
import numpy as np
import os
import pandas as pd

def check_increasing_latlon(var, lat, lon):
    """
    Checks that the latitude and longitude are in increasing order. Returns ordered arrays.

    Assumes that lat and lon are the second-last and last dimensions of the array var.
    """
    lat = np.array(lat)
    lon = np.array(lon)
    var = np.array(var)

    revlat = False
    revlon = False
    if lat[1] < lat[0]:
        revlat = True
        print('Latitude is in reverse order! Ordering..\n')
    if lon[1] < lon[0]:
        revlon = True
        print('Longitude is in reverse order! Ordering..\n')

    if revlat and not revlon:
        var = var[..., ::-1, :]
        lat = lat[::-1]
    elif revlon and not revlat:
        var = var[..., :, ::-1]
        lon = lon[::-1]
    elif revlat and revlon:
        var = var[..., ::-1, ::-1]
        lat = lat[::-1]
        lon = lon[::-1]

    return var, lat, lon


def readxDncfield(ifile, extract_level = None, select_var = None, compress_dummy_dim = True, pressure_in_Pa = True, force_level_units = None, verbose = True, keep_only_Ndim_vars = True):
    """
    Read a netCDF file as it is, preserving all dimensions and multiple variables.

    < extract_level > : float. If set, only the corresponding level is extracted.
    < select_var > : str or list. For a multi variable file, only variable names corresponding to those listed in select_var are read. Redundant definition are treated safely: variable is extracted only one time.

    < pressure_in_Pa > : bool. If True (default) pressure levels are converted to Pa.
    < force_level_units > : str. Set units of levels to avoid errors in reading. To be used with caution, always check the level output to ensure that the units are correct.
    < keep_only_Ndim_vars > : keeps only variables with correct size (excludes variables like time_bnds, lat_bnds, ..)
    """

    fh = Dataset(ifile)
    dimensions = fh.dimensions.keys()
    if verbose: print('Dimensions: {}\n'.format(dimensions))
    ndim = len(dimensions)

    variab_names = fh.variables.keys()
    if 'time' in variab_names and 'dim0' in dimensions:
        dimensions[dimensions.index('dim0')] = 'time'

    for nam in dimensions:
        if nam in variab_names: variab_names.remove(nam)
    if verbose: print('Variables: {}\n'.format(variab_names))
    nvars = len(variab_names)
    print('Field as {} dimensions and {} vars. All keys: {}'.format(ndim, nvars, fh.variables.keys()))

    try:
        lat_o         = fh.variables['lat'][:]
        lon_o         = fh.variables['lon'][:]
    except KeyError as ke:
        #print(repr(ke))
        lat_o         = fh.variables['latitude'][:]
        lon_o         = fh.variables['longitude'][:]
    true_dim = 2

    vars = dict()
    if select_var is None:
        for varna in variab_names:
            var = fh.variables[varna][:]
            var, lat, lon = check_increasing_latlon(var, lat_o, lon_o)
            vars[varna] = var
    else:
        print('Extracting {}\n'.format(select_var))
        for varna in variab_names:
            if varna in select_var:
                var = fh.variables[varna][:]
                var, lat, lon = check_increasing_latlon(var, lat_o, lon_o)
                vars[varna] = var
        if len(vars.keys()) == 0:
            raise KeyError('No variable corresponds to names: {}. All variabs: {}'.format(select_var, variab_names))


    if 'time' in dimensions:
        true_dim += 1
        time        = fh.variables['time'][:]
        time_units  = fh.variables['time'].units
        time_cal    = fh.variables['time'].calendar

        time = list(time)
        dates = num2date(time,time_units,time_cal)

        if time_cal == '365_day' or time_cal == 'noleap':
            dates = adjust_noleap_dates(dates)
        elif time_cal == '360_day':
            dates = adjust_360day_dates(dates)

        print('calendar: {0}, time units: {1}'.format(time_cal,time_units))

    if true_dim == 3 and ndim > 3:
        lev_names = ['level', 'lev', 'pressure', 'plev', 'plev8']
        found = False
        for levna in lev_names:
            if levna in dimensions:
                oklevname = levna
                level = fh.variables[levna][:]
                nlevs = len(level)
                found = True
                break

        if not found:
            print('Level name not found among: {}\n'.format(lev_names))
            print('Does the variable have levels?')
        else:
            true_dim += 1

            try:
                level_units = fh.variables[oklevname].units
            except AttributeError as atara:
                print('level units not found in file {}\n'.format(ifile))
                if force_level_units is not None:
                    level_units = force_level_units
                    print('setting level units to {}\n'.format(force_level_units))
                    print('levels are {}\n'.format(level))
                else:
                    raise atara

            print('level units are {}\n'.format(level_units))
            if pressure_in_Pa:
                if level_units in ['millibar', 'millibars','hPa']:
                    level = 100.*level
                    level_units = 'Pa'
                    print('Converting level units from hPa to Pa\n')

            if extract_level is not None:
                lvel = extract_level
                if nlevs > 1:
                    if level_units=='millibar' or level_units=='hPa':
                        l_sel=int(np.where(level==lvel)[0])
                        print('Selecting level {0} millibar'.format(lvel))
                    elif level_units=='Pa':
                        l_sel=int(np.where(level==lvel*100)[0])
                        print('Selecting level {0} Pa'.format(lvel*100))
                    level = lvel
                else:
                    level = level[0]
                    l_sel = 0

                for varna in vars.keys():
                    vars[varna] = vars[varna][:,l_sel, ...]
            else:
                levord = level.argsort()
                level = level[levord]
                for varna in vars.keys():
                    vars[varna] = vars[varna][:, levord, ...]

    var_units = dict()
    for varna in vars.keys():
        try:
            var_units[varna] = fh.variables[varna].units
        except:
            var_units[varna] = None

        if var_units[varna] == 'm**2 s**-2':
            print('From geopotential (m**2 s**-2) to geopotential height (m)')   # g0=9.80665 m/s2
            vars[varna] = vars[varna]/9.80665
            var_units[varna] = 'm'


    # if len(vars.keys()) == 1:
    #     vars = vars.values()[0]
    print('Dimension of variables is {}\n'.format(true_dim))
    if keep_only_Ndim_vars:
        for varna in vars.keys():
            if len(vars[varna].shape) < true_dim:
                print('Erasing variable {}\n'.format(varna))
                vars.pop(varna)
                var_units.pop(varna)

    n_compressed = 0
    if compress_dummy_dim:
        for varna in vars.keys():
            if 1 in vars[varna].shape:
                n_compressed = np.sum(np.array(vars[varna].shape) == 1)
                vars[varna] = vars[varna].squeeze()

    true_dim -= n_compressed

    if true_dim == 2:
        return vars, lat, lon, var_units
    elif true_dim == 3:
        return vars, lat, lon, dates, time_units, var_units, time_cal
    elif true_dim == 4:
        return vars, level, lat, lon, dates, time_units, var_units, time_cal


def read4Dncfield(ifile, extract_level = None, compress_dummy_dim = True, increasing_plev = True):
    '''
    GOAL
        Read netCDF file of 4Dfield, optionally selecting a level.
    USAGE
        var, dates = read4Dncfield(ifile, extract_level = level)
        ifile: filename
        extract_level: level to be selected in hPa
    '''
    #----------------------------------------------------------------------------------------
    print('__________________________________________________________')
    print('Reading the 4D field [time x level x lat x lon]: \n{0}'.format(ifile))
    #----------------------------------------------------------------------------------------
    fh = Dataset(ifile, mode='r')
    variabs=[]
    for variab in fh.variables:
        variabs.append(variab)
    #print('The variables in the nc file are: ', variabs)
    lev_names = ['level', 'lev', 'pressure', 'plev', 'plev8']
    for levna in lev_names:
        if levna in variabs:
            oklevname = levna
            level = fh.variables[levna][:]
            nlevs = len(level)
            break

    try:
        lat         = fh.variables['lat'][:]
        lon         = fh.variables['lon'][:]
    except KeyError as ke:
        print(repr(ke))
        lat         = fh.variables['latitude'][:]
        lon         = fh.variables['longitude'][:]

    time        = fh.variables['time'][:]
    time_units  = fh.variables['time'].units
    time_cal    = fh.variables['time'].calendar

    try:
        var_units   = fh.variables[variabs[-1]].units
    except:
        var_units = None

    if extract_level is not None:
        lvel = extract_level
        if nlevs > 1:
            level_units = fh.variables[levna].units
            if level_units=='millibar' or level_units=='hPa':
                l_sel=int(np.where(level==lvel)[0])
                print('Selecting level {0} millibar'.format(lvel))
            elif level_units=='Pa':
                l_sel=int(np.where(level==lvel*100)[0])
                print('Selecting level {0} Pa'.format(lvel*100))
            level = lvel
        else:
            level = level[0]
            l_sel = 0

        var         = fh.variables[variabs[-1]][:,l_sel,:,:]
        txt='{0}{1} dimension for a single ensemble member [time x lat x lon]: {2}'.format(variabs[-1],lvel,var.shape)
    else:
        var         = fh.variables[variabs[-1]][:,:,:,:]
        txt='{0} dimension for a single ensemble member [time x lat x lon]: {1}'.format(variabs[-1],var.shape)
        if increasing_plev:
            levord = level.argsort()
            level = level[levord]
            var = var[:, levord, ...]

    print(txt)
    #print(fh.variables)
    if var_units == 'm**2 s**-2':
        print('From geopotential (m**2 s**-2) to geopotential height (m)')   # g0=9.80665 m/s2
        var=var/9.80665
        var_units='m'
    print('calendar: {0}, time units: {1}'.format(time_cal,time_units))

    time = list(time)
    dates = num2date(time,time_units,time_cal)
    fh.close()

    if time_cal == '365_day' or time_cal == 'noleap':
        dates = adjust_noleap_dates(dates)
    elif time_cal == '360_day':
        dates = adjust_360day_dates(dates)

    if compress_dummy_dim:
        var = var.squeeze()

    var, lat, lon = check_increasing_latlon(var, lat, lon)

    return var, level, lat, lon, dates, time_units, var_units, time_cal


def adjust_noleap_dates(dates):
    """
    When the time_calendar is 365_day or noleap, num2date() returns a cftime array which is not convertible to datetime (and to pandas DatetimeIndex). This fixes this problem, returning the usual datetime array.
    """
    dates_ok = []
    #for ci in dates: dates_ok.append(datetime.strptime(ci.strftime(), '%Y-%m-%d %H:%M:%S'))
    # diffs = []
    for ci in dates:
        # coso = ci.isoformat()
        coso = ci.strftime()
        nudat = pd.Timestamp(coso).to_pydatetime()
        # print(coso, nudat)
        # if ci-nudat >= pd.Timedelta('1 days'):
        #     raise ValueError
        # diffs.append(ci-nudat)
        dates_ok.append(nudat)

    dates_ok = np.array(dates_ok)
    # print(diffs)

    return dates_ok


def adjust_360day_dates(dates):
    """
    When the time_calendar is 360_day (please not!), num2date() returns a cftime array which is not convertible to datetime (obviously)(and to pandas DatetimeIndex). This fixes this problem in a completely arbitrary way, missing one day each two months. Returns the usual datetime array.
    """
    dates_ok = []
    #for ci in dates: dates_ok.append(datetime.strptime(ci.strftime(), '%Y-%m-%d %H:%M:%S'))
    strindata = '{:4d}-{:02d}-{:02d} 12:00:00'

    for ci in dates:
        firstday = strindata.format(ci.year, 1, 1)
        num = ci.dayofyr-1
        add_day = num/72 # salto un giorno ogni 72
        okday = pd.Timestamp(firstday)+pd.Timedelta('{} days'.format(num+add_day))
        dates_ok.append(okday.to_pydatetime())

    dates_ok = np.array(dates_ok)

    return dates_ok


def read3Dncfield(ifile, compress_dummy_dim = True):
    '''
    GOAL
        Read netCDF file of 3Dfield
    USAGE
        var, dates = read3Dncfield(fname)
        fname: filname
    '''
    #----------------------------------------------------------------------------------------
    print('__________________________________________________________')
    print('Reading the 3D field [time x lat x lon]: \n{0}'.format(ifile))
    #----------------------------------------------------------------------------------------
    fh = Dataset(ifile, mode='r')
    variabs=[]
    for variab in fh.variables:
        variabs.append(variab)
    #print('The variables in the nc file are: ', variabs)

    try:
        lat         = fh.variables['lat'][:]
        lon         = fh.variables['lon'][:]
    except KeyError as ke:
        print(repr(ke))
        lat         = fh.variables['latitude'][:]
        lon         = fh.variables['longitude'][:]

    time        = fh.variables['time'][:]
    time_units  = fh.variables['time'].units
    time_cal    = fh.variables['time'].calendar

    try:
        var_units   = fh.variables[variabs[-1]].units
    except:
        var_units = None

    var         = fh.variables[variabs[-1]][:,:,:]
    txt='{0} dimension [time x lat x lon]: {1}'.format(variabs[-1],var.shape)

    if compress_dummy_dim and var.ndim > 3:
        var = var.squeeze()
    #print(fh.variables)
    time = list(time)
    dates = num2date(time, time_units, time_cal)
    fh.close()

    if time_cal == '365_day' or time_cal == 'noleap':
        dates = adjust_noleap_dates(dates)

    var, lat, lon = check_increasing_latlon(var, lat, lon)

    print(txt)

    return var, lat, lon, dates, time_units, var_units


def read2Dncfield(ifile):
    '''
    GOAL
        Read netCDF file of 2Dfield
    USAGE
        var = read2Dncfield(fname)
        fname: filename
    '''
    #----------------------------------------------------------------------------------------
    print('__________________________________________________________')
    print('Reading the 2D field [lat x lon]: \n{0}'.format(ifile))
    #----------------------------------------------------------------------------------------
    fh = Dataset(ifile, mode='r')
    variabs=[]
    for variab in fh.variables:
        variabs.append(variab)
    #print('The variables in the nc file are: ', variabs)

    try:
        lat         = fh.variables['lat'][:]
        lon         = fh.variables['lon'][:]
    except KeyError as ke:
        print(repr(ke))
        lat         = fh.variables['latitude'][:]
        lon         = fh.variables['longitude'][:]

    #var_units   = fh.variables[variabs[2]].units
    var         = fh.variables[variabs[2]][:,:]
    var_units   = fh.variables[variabs[3]].units

    txt='{0} dimension [lat x lon]: {1}'.format(variabs[2],var.shape)
    #print(fh.variables)
    fh.close()

    #print('\n'+txt)

    return var, var_units, lat, lon


def read_N_2Dfields(ifile):
    '''
    GOAL
        Read var in ofile netCDF file
    USAGE
        read a number N of 2D fields [latxlon]
        fname: output filname
    '''
    fh = Dataset(ifile, mode='r')
    variabs=[]
    for variab in fh.variables:
        variabs.append(variab)
    #print('The variables in the nc file are: ', variabs)

    num         = fh.variables['num'][:]

    try:
        lat         = fh.variables['lat'][:]
        lon         = fh.variables['lon'][:]
    except KeyError as ke:
        print(repr(ke))
        lat         = fh.variables['latitude'][:]
        lon         = fh.variables['longitude'][:]

    var         = fh.variables[variabs[3]][:,:,:]
    var_units   = fh.variables[variabs[3]].units
    txt='{0} dimension [num x lat x lon]: {1}'.format(variabs[3],var.shape)
    #print(fh.variables)
    fh.close()

    #print('\n'+txt)

    return var, var_units, lat, lon


def save2Dncfield(lats,lons,variab,varname,ofile):
    '''
    GOAL
        Save var in ofile netCDF file
    USAGE
        save2Dncfield(var,ofile)
        fname: output filname
    '''
    try:
        os.remove(ofile) # Remove the outputfile
    except OSError:
        pass
    dataset = Dataset(ofile, 'w', format='NETCDF4_CLASSIC')
    #print(dataset.file_format)

    lat = dataset.createDimension('lat', variab.shape[0])
    lon = dataset.createDimension('lon', variab.shape[1])

    # Create coordinate variables for 2-dimensions
    lat = dataset.createVariable('lat', np.float32, ('lat',))
    lon = dataset.createVariable('lon', np.float32, ('lon',))
    # Create the actual 2-d variable
    var = dataset.createVariable(varname, np.float64,('lat','lon'))

    #print('variable:', dataset.variables[varname])

    #for varn in dataset.variables.keys():
    #    print(varn)
    # Variable Attributes
    lat.units='degree_north'
    lon.units='degree_east'
    #var.units = varunits

    lat[:]=lats
    lon[:]=lons
    var[:,:]=variab

    dataset.close()

    #----------------------------------------------------------------------------------------
    print('The 2D field [lat x lon] is saved as \n{0}'.format(ofile))
    print('__________________________________________________________')
    #----------------------------------------------------------------------------------------

def save3Dncfield(lats,lons,variab,varname,varunits,dates,timeunits,time_cal,ofile):
    '''
    GOAL
        Save var in ofile netCDF file
    USAGE
        save3Dncfield(var,ofile)
        fname: output filname
    '''
    try:
        os.remove(ofile) # Remove the outputfile
    except OSError:
        pass
    dataset = Dataset(ofile, 'w', format='NETCDF4_CLASSIC')
    #print(dataset.file_format)

    time = dataset.createDimension('time', None)
    lat = dataset.createDimension('lat', variab.shape[1])
    lon = dataset.createDimension('lon', variab.shape[2])

    # Create coordinate variables for 3-dimensions
    time = dataset.createVariable('time', np.float64, ('time',))
    lat = dataset.createVariable('lat', np.float32, ('lat',))
    lon = dataset.createVariable('lon', np.float32, ('lon',))
    # Create the actual 3-d variable
    var = dataset.createVariable(varname, np.float64,('time','lat','lon'))

    #print('variable:', dataset.variables[varname])

    #for varn in dataset.variables.keys():
    #    print(varn)
    # Variable Attributes
    time.units=timeunits
    time.calendar=time_cal
    lat.units='degree_north'
    lon.units='degree_east'
    var.units = varunits

    # Fill in times.
    time[:] = date2num(dates, units = timeunits, calendar = time_cal)#, calendar = times.calendar)
    print(time_cal)
    print('time values (in units {0}): {1}'.format(timeunits,time[:]))
    print(dates)

    #print('time values (in units %s): ' % time)

    lat[:]=lats
    lon[:]=lons
    var[:,:,:]=variab

    dataset.close()

    #----------------------------------------------------------------------------------------
    print('The 3D field [time x lat x lon] is saved as \n{0}'.format(ofile))
    print('__________________________________________________________')
    #----------------------------------------------------------------------------------------

def save_N_2Dfields(lats,lons,variab,varname,varunits,ofile):
    '''
    GOAL
        Save var in ofile netCDF file
    USAGE
        save a number N of 2D fields [latxlon]
        fname: output filname
    '''
    try:
        os.remove(ofile) # Remove the outputfile
    except OSError:
        pass
    dataset = Dataset(ofile, 'w', format='NETCDF4_CLASSIC')
    #print(dataset.file_format)

    num = dataset.createDimension('num', variab.shape[0])
    lat = dataset.createDimension('lat', variab.shape[1])
    lon = dataset.createDimension('lon', variab.shape[2])

    # Create coordinate variables for 3-dimensions
    num = dataset.createVariable('num', np.int32, ('num',))
    lat = dataset.createVariable('lat', np.float32, ('lat',))
    lon = dataset.createVariable('lon', np.float32, ('lon',))
    # Create the actual 3-d variable
    var = dataset.createVariable(varname, np.float64,('num','lat','lon'))

    #print('variable:', dataset.variables[varname])

    #for varn in dataset.variables.keys():
    #    print(varn)
    # Variable Attributes
    lat.units='degree_north'
    lon.units='degree_east'

    try:
        var.units = varunits
    except:
        var.units = 'unknown'
        print('WARNING in save_N_2D: unknown varunits\n')

    num[:]=np.arange(variab.shape[0])
    lat[:]=lats
    lon[:]=lons
    var[:,:,:]=variab

    dataset.close()

    #----------------------------------------------------------------------------------------
    print('The {0} 2D fields [num x lat x lon] are saved as \n{1}'.format(variab.shape[0], ofile))
    print('__________________________________________________________')
    #----------------------------------------------------------------------------------------
#
#
# def read3Dncfield(ifile):
#     '''
#     GOAL
#         Read netCDF file of 3Dfield
#     USAGE
#         var, lat, lon, dates = read3Dncfield(filename)
#     '''
#     #----------------------------------------------------------------------------------------
#     #print('__________________________________________________________')
#     #print('Reading the 3D field [time x lat x lon]: \n{0}'.format(ifile))
#     #----------------------------------------------------------------------------------------
#     fh = Dataset(ifile, mode='r')
#     variabs=[]
#     for variab in fh.variables:
#         variabs.append(variab)
#     #print('The variables in the nc file are: ', variabs)
#
#     lat         = fh.variables['lat'][:]
#     lon         = fh.variables['lon'][:]
#     time        = fh.variables['time'][:]
#     time_units  = fh.variables['time'].units
#     var_units   = fh.variables[variabs[3]].units
#     var         = fh.variables[variabs[3]][:,:,:]
#     txt='{0} dimension [time x lat x lon]: {1}'.format(variabs[3],var.shape)
#     #print(fh.variables)
#     dates=num2date(time,time_units)
#     fh.close()
#
#     #print('\n'+txt)
#
#     return var, var_units, lat, lon, dates, time_units
#
#
# def save_N_2Dfields(lats,lons,variab,varname,varunits,ofile):
#     '''
#     GOAL
#         Save var in ofile netCDF file
#     USAGE
#         save a number N of 2D fields [latxlon]
#     '''
#     try:
#         os.remove(ofile) # Remove the outputfile
#     except OSError:
#         pass
#     dataset = Dataset(ofile, 'w', format='NETCDF4_CLASSIC')
#     #print(dataset.file_format)
#
#     num = dataset.createDimension('num', variab.shape[0])
#     lat = dataset.createDimension('lat', variab.shape[1])
#     lon = dataset.createDimension('lon', variab.shape[2])
#
#     # Create coordinate variables for 3-dimensions
#     num = dataset.createVariable('num', np.int32, ('num',))
#     lat = dataset.createVariable('lat', np.float32, ('lat',))
#     lon = dataset.createVariable('lon', np.float32, ('lon',))
#     # Create the actual 3-d variable
#     var = dataset.createVariable(varname, np.float64,('num','lat','lon'))
#
#     #print('variable:', dataset.variables[varname])
#
#     #for varn in dataset.variables.keys():
#     #    print(varn)
#     # Variable Attributes
#     lat.units='degree_north'
#     lon.units='degree_east'
#     var.units = varunits
#
#     num[:]=np.arange(variab.shape[0])
#     lat[:]=lats
#     lon[:]=lons
#     var[:,:,:]=variab
#
#     dataset.close()
#
#     #----------------------------------------------------------------------------------------
#     print('The {0} 2D fields [num x lat x lon] are saved as \n{1}'.format(variab.shape[0], ofile))
#     print('__________________________________________________________')
#     #----------------------------------------------------------------------------------------
#
#
# def read_N_2Dfields(ifile):
#     '''
#     GOAL
#         read a number N of 2D fields [latxlon]
#     USAGE
#         var, lat, lon, dates = read_N_2Dfields(filename)
#     '''
#     fh = Dataset(ifile, mode='r')
#     variabs=[]
#     for variab in fh.variables:
#         variabs.append(variab)
#     #print('The variables in the nc file are: ', variabs)
#
#     num         = fh.variables['num'][:]
#     lat         = fh.variables['lat'][:]
#     lon         = fh.variables['lon'][:]
#     var         = fh.variables[variabs[3]][:,:,:]
#     var_units   = fh.variables[variabs[3]].units
#     txt='{0} dimension [num x lat x lon]: {1}'.format(variabs[3],var.shape)
#     #print(fh.variables)
#     fh.close()
#
#     #print('\n'+txt)
#
#     return var, var_units, lat, lon
