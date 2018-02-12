# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:30:30 2018

@author: chrwil
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:44:27 2018

@author: chrwil
"""

# Create overlapping tiles of raster
"""
* start anaconda prompt
* cd C:/Users/chrwil/Downloads/LFMapper-master/LFMapper-master (creassemap is stored in here)
* activate LFMapper_work (conda environment -- see README)
* start spyder
"""

import os
import sys
import numpy as np
from osgeo import gdal, gdalconst 
from osgeo.gdalconst import * 

import georaster 

## Funcs
def raster_binary_to_2d_array(file_name, gdal_driver='GTiff'):
    '''
    Converts a binary file of ENVI type to a numpy arra.
    Lack of an ENVI .hdr file will cause this to crash.
    VARIABLES
    file_name : file name and path of your file
    RETURNS
    geotransform, inDs, cols, rows, bands, originX, originY, pixelWidth, pixelHeight, image_array, image_array_name
    '''
    driver = gdal.GetDriverByName(gdal_driver) ## http://www.gdal.org/formats_list.html
    driver.Register()

    inDs = gdal.Open(file_name, GA_ReadOnly)
    
    if inDs is None:
        print("Couldn't open this file: %s" %(file_name))
        print('/nPerhaps you need an ENVI .hdr file? A quick way to do this is to just open the binary up in ENVI and one will be created for you.')
        sys.exit("Try again!")
    else:
        print("%s opened successfully" %file_name)
        
        
    #print( )'Get image size')
    cols = inDs.RasterXSize
    rows = inDs.RasterYSize
    bands = inDs.RasterCount

                    
    #print( )'Get georeference information')
    geotransform = inDs.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
        
        
    # Set pixel offset.....
    band = inDs.GetRasterBand(1)
    image_array = band.ReadAsArray(0, 0, cols, rows)
    image_array_name = file_name

    
    return geotransform, inDs, cols, rows, bands, originX, originY, pixelWidth, pixelHeight, image_array, image_array_name


def load_dem(file_name):
    '''
    Loads an ENVI binary as a numpy image array also returning a tuple including map and projection info
    VARIABLES
    file_name : file name and path of your file
    RETURNS
    image_array, post, (geotransform, inDs)
    '''
    geotransform, inDs, _, _, _, _, _, post, _, image_array, _ = raster_binary_to_2d_array(file_name)
    return image_array, post, geotransform, inDs


## IO
file_name = "C:/Users/chrwil/Desktop/Hofsjokull_data/hofsjokull-2008-cmb-v1/hofsjokull_FLAT_5m_SUBSET.tif"
out_path = 'C:/Users/chrwil/Desktop/Hofsjokull_data/outputs_5m/tiles'
dem_original, post, dem_data, inDs = load_dem(file_name)

## Calc extent 
nrows=dem_original.shape[0]
ncols=dem_original.shape[1]
pixel_dim=np.abs(dem_data[5])

#<<<<<<<<<< make into function
easting_min=dem_data[0]                          # tl_x
northing_max=dem_data[3]                         # tl_y
northing_min = northing_max - (post*nrows)        # bl_x
easting_max  = easting_min + (post*ncols)         # br_y

assert northing_min < northing_max
assert easting_min < easting_max

## Set tile dimensions

### in metres
tile_dim_m=200
tile_overlap=tile_dim_m/2 # 50% overlap - need to see how this affects the variables tl_moving_window_*_RANGE

### convert to pixels
tile_dim_px=tile_dim_m/pixel_dim


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CROP ORIGINAL IMAGE, REST TRANSFORM AND CALC SUB-IMAGE
## Crop image so cleanly divisible by tile_dim_m

nrows_cropped=nrows
while nrows_cropped%tile_dim_px != 0:
    nrows_cropped=nrows_cropped-1

ncols_cropped=ncols
while ncols_cropped%tile_dim_px != 0:
    ncols_cropped=ncols_cropped-1

# crop image 
dem_crop=dem_original
dem_crop=dem_crop[0:nrows_cropped,0:ncols_cropped] # currently crops from the top left corner... make it crop so only edges are lost?

#reassign geotransform

## Re-calc extent 
nrows_crop=dem_crop.shape[0]
ncols_crop=dem_crop.shape[1]
northing_min = northing_max - (post*nrows_crop)        # bl_x
easting_max  = easting_min + (post*ncols_crop)         # br_y


## Sub image extent (within border so that moving window doesn;t have any edge effects)
sub_easting_min =easting_min+(tile_overlap)
sub_easting_max =easting_max-(tile_overlap)
sub_northing_min=northing_min+(tile_overlap)
sub_northing_max=northing_max-(tile_overlap)
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MAIN BIT....

################## Now use same loop and instead of each plot, save each tile
################## When tiles are brought back together, just merge using a priority function as values should be the same
################## If using another convolution layer, make sure the overlaps are big enough to make sure there are no gaps...
import gdal, ogr, osr, os
import numpy as np

tl_moving_window_min_easting_RANGE=np.arange(sub_easting_min,sub_easting_max+(tile_overlap),(tile_overlap))
tl_moving_window_max_northing_RANGE=np.arange(sub_northing_max,sub_northing_min-(tile_overlap),(tile_overlap)*-1)

tiles_x=[]
tiles_y=[]

tiles_x_c1=[]
tiles_y_c1=[]

eastings=[]
northings=[]
# write raster from array to new extent - same CRS and pixel size as original dataset
def array2raster(newRasterfn,z_array,inDs,originX, originY, driver='GTiff'):

    cols=z_array.shape[1], 
    rows=z_array.shape[0]

    driver = gdal.GetDriverByName(driver)
    driver.Register()

    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32) ### bug here...
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)

    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(inDs.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())

    outband.FlushCache()

count=0

for tl_easting_TILE_initial in tl_moving_window_min_easting_RANGE:
    for tl_northing_TILE_initial in tl_moving_window_max_northing_RANGE:

        count+=1

        print ("  ")
        print("Tile %i" %count)

        print("left easting initial: %f" %tl_easting_TILE_initial)
        print("top northing initial: %f" %tl_northing_TILE_initial)
        
        def assign_new_tile_dims(dem_crop, tl_easting_TILE, br_easting_TILE, tl_northing_TILE, br_northing_TILE, tiles_x, tiles_y):
            
            
            dem_tile_tl_easting=np.int(np.floor(tl_easting_TILE))
            dem_tile_br_easting=np.int(np.ceil(br_easting_TILE))

            dem_tile_br_northing=np.int(np.floor(br_northing_TILE))
            dem_tile_tl_northing=np.int(np.ceil(tl_northing_TILE))

            #dem_tile=dem_crop[dem_tile_tl_easting:dem_tile_br_easting, 
            #                dem_tile_br_northing:dem_tile_tl_northing]

            # for plotting
            tiles_x.append((tl_easting_TILE, br_easting_TILE,br_easting_TILE,tl_easting_TILE, tl_easting_TILE))
            tiles_y.append((br_northing_TILE,br_northing_TILE,tl_northing_TILE,tl_northing_TILE, br_northing_TILE))
            
            #return (dem_tile, tiles_x, tiles_y, (dem_tile_tl_easting, dem_tile_br_easting, dem_tile_br_northing, dem_tile_tl_northing))
            return (tiles_x, tiles_y, (dem_tile_tl_easting, dem_tile_br_easting, dem_tile_br_northing, dem_tile_tl_northing))

        tl_easting_TILE=tl_easting_TILE_initial-(tile_overlap)
        br_easting_TILE=tl_easting_TILE+(tile_dim_m)
        tl_northing_TILE=tl_northing_TILE_initial+(tile_overlap)
        br_northing_TILE=tl_northing_TILE-(tile_dim_m)

        print("left easting: %f" %tl_easting_TILE)
        print("right easting: %f" %br_easting_TILE)

        print("top northing: %f" %tl_northing_TILE)
        print("bottom northing: %f" %br_northing_TILE)
        #dem_tile, tiles_x, tiles_y, tile_xy_dims = assign_new_tile_dims(dem_crop, tl_easting_TILE, br_easting_TILE, tl_northing_TILE, br_northing_TILE, tiles_x, tiles_y)
        tiles_x, tiles_y, tile_xy_dims = assign_new_tile_dims(dem_crop, tl_easting_TILE, br_easting_TILE, tl_northing_TILE, br_northing_TILE, tiles_x, tiles_y)

        # Write raster

        #*** USE GEORASTER LIBRARY TO READ IN DTM TO THE EXTENT SPECIFIED BY THE TILE COORDINATES (tile_xy_dims) ***
        #*** read me here: http://georaster.readthedocs.io/en/latest/api.html?highlight=sub#georaster.__Raster.read_single_band_subset ***
        #*** no need to transform the coordinates to pixels yourself (use georaster which uses gdal to do this - probably with some bilinear interpolatin to sort the edges...) ***

        
        #dd=georaster.SingleBandRaster(file_name, load_data=False, latlon=True)    
        dd_tile=georaster.SingleBandRaster(file_name,load_data=tile_xy_dims, latlon=False)
        ofile="%s/tile_%i.tif" %(out_path, count)
        dd_tile.save_geotiff(ofile)
        
print("COMPLETE")


# plot tiles over main extent
def plot_it():
    fig, ax = plt.subplots()
    ax.plot((easting_min,easting_max,easting_max,easting_min, easting_min),
                (northing_min,northing_min,northing_max,northing_max, northing_min), color='red')

    plt.plot((sub_easting_min,sub_easting_max,sub_easting_max,sub_easting_min,sub_easting_min),
                (sub_northing_min,sub_northing_min,sub_northing_max,sub_northing_max,sub_northing_min), color='blue')

    ax.set_xlim(easting_min-500, easting_max+500)
    ax.set_ylim(northing_min-500, northing_max+500)

    ax.plot(tiles_x[0], tiles_y[0])
    ax.plot(tiles_x[1], tiles_y[1])
    #ax.plot(tiles_x[2], tiles_y[2])
    #ax.plot(tiles_x[3], tiles_y[3])
    #ax.plot(tiles_x[4], tiles_y[4])
    #ax.plot(tiles_x[5], tiles_y[5])
    #ax.plot(tiles_x[6], tiles_y[6])
    #ax.plot(tiles_x[7], tiles_y[7])
    #ax.plot(tiles_x[8], tiles_y[8])
    #ax.plot(tiles_x[9], tiles_y[9])
    #ax.plot(tiles_x[10], tiles_y[10])

    #ax.plot(tiles_x[11], tiles_y[11])
    #ax.plot(tiles_x[12], tiles_y[12])
    #ax.plot(tiles_x[13], tiles_y[13])
    #ax.plot(tiles_x[14], tiles_y[14])

    #ax.plot(tiles_x[15], tiles_y[15])
    #ax.plot(tiles_x[16], tiles_y[16])
    #ax.plot(tiles_x[17], tiles_y[17])
    #ax.plot(tiles_x[18], tiles_y[18])
    #ax.plot(tiles_x[19], tiles_y[19])
    #ax.plot(tiles_x[20], tiles_y[20])
    #ax.plot(tiles_x[20], tiles_y[21])

    #ax.plot(tiles_x[110], tiles_y[110])
    #ax.plot(tiles_x[111], tiles_y[111])
    #ax.plot(tiles_x[112], tiles_y[112])
    #ax.plot(tiles_x[113], tiles_y[113])
    #ax.plot(tiles_x[114], tiles_y[114])
    #ax.plot(tiles_x[115], tiles_y[115])
    #ax.plot(tiles_x[116], tiles_y[116])
    #ax.plot(tiles_x[117], tiles_y[117])
    #ax.plot(tiles_x[118], tiles_y[118])
    #ax.plot(tiles_x[119], tiles_y[119])
    #ax.plot(tiles_x[120], tiles_y[120])
    #ax.plot(tiles_x[121], tiles_y[121])

    ax.axes.set_aspect('equal', 'datalim')
    ax.set_title("Single tile locations")
    plt.show()

plot_it()

"""
* extend to bottom row and furthest column (adjust the tl_moving_window_min_easting_RANGE and tl_moving_window_max_northing_RANGE variables to go a cell further)
* check how changing the overlap works - the step should also change (affects the tl_moving_window_min_easting_RANGE and tl_moving_window_max_northing_RANGE variables)
* check tiles overlap (open in QGIS or something)
* send to Andy
"""

