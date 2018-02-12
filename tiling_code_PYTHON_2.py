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
easting_min=dem_data[0]      					# tl_x
northing_max=dem_data[3]     					# tl_y
northing_min = northing_max - (post*nrows)		# bl_x
easting_max  = easting_min + (post*ncols) 		# br_y

assert northing_min < northing_max
assert easting_min < easting_max

## Set tile dimensions

### in metres
tile_dim_m=200

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
northing_min = northing_max - (post*nrows_crop)		# bl_x
easting_max  = easting_min + (post*ncols_crop) 		# br_y


## Sub image extent (within border so that moving window doesn;t have any edge effects)
sub_easting_min =easting_min+(tile_dim_m/2)
sub_easting_max =easting_max-(tile_dim_m/2)
sub_northing_min=northing_min+(tile_dim_m/2)
sub_northing_max=northing_max-(tile_dim_m/2)
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


## Plot original and "internal" extents
import matplotlib.pyplot as plt

plt.plot((easting_min,easting_max,easting_max,easting_min, easting_min),
			(northing_min,northing_min,northing_max,northing_max, northing_min), color='blue')

plt.plot((sub_easting_min,sub_easting_max,sub_easting_max,sub_easting_min,sub_easting_min),
			(sub_northing_min,sub_northing_min,sub_northing_max,sub_northing_max,sub_northing_min), color='red')

plt.xlabel("easting")
plt.ylabel("northing")
plt.axes().set_aspect('equal', 'datalim')

# Iterate through easting and northign extents

# top left corner
original_tl_moving_window_min_easting_RANGE=np.arange(easting_min,easting_max,tile_dim_m)
original_tl_moving_window_max_northing_RANGE=np.arange(northing_max,northing_min,tile_dim_m*-1)

tl_moving_window_min_easting_RANGE=np.arange(sub_easting_min,sub_easting_max,tile_dim_m)
tl_moving_window_max_northing_RANGE=np.arange(sub_northing_max,sub_northing_min,tile_dim_m*-1)


easting_step=np.diff(tl_moving_window_min_easting_RANGE, n=2)
northing_step=np.diff(tl_moving_window_max_northing_RANGE, n=2)
# iterate through top left and calcualte  bottom right corner
import matplotlib.patches as patches

count=0

fig, ax = plt.subplots()
ax.plot((easting_min,easting_max,easting_max,easting_min, easting_min),
			(northing_min,northing_min,northing_max,northing_max, northing_min), color='red')
ax.set_xlim(easting_min-500, easting_max+500)
ax.set_ylim(northing_min-500, northing_max+500)

for tl_easting_TILE in tl_moving_window_min_easting_RANGE:
	for tl_northing_TILE in tl_moving_window_max_northing_RANGE:

		count+=1
		br_easting_TILE=tl_easting_TILE+tile_dim_m
		br_northing_TILE=tl_northing_TILE-tile_dim_m

		print("Tile %i" %count)
		print(tl_easting_TILE, tl_northing_TILE, br_easting_TILE, br_northing_TILE)
		
		# Iteritively add tile to plot which has extent equal to original raster (with original raster drawn)
		eastings=(tl_easting_TILE, br_easting_TILE,br_easting_TILE,tl_easting_TILE, tl_easting_TILE)
		northings=(br_northing_TILE,br_northing_TILE,tl_northing_TILE,tl_northing_TILE, br_northing_TILE)
		ax.plot(eastings, northings)#, color='blue')
		
ax.plot((sub_easting_min,sub_easting_max,sub_easting_max,sub_easting_min, sub_easting_min),
			(sub_northing_min,sub_northing_min,sub_northing_max,sub_northing_max, sub_northing_min), color='black')

ax.plot((easting_min,easting_max,easting_max,easting_min, easting_min),
			(northing_min,northing_min,northing_max,northing_max, northing_min), color='red')

ax.axes.set_aspect('equal', 'datalim')
ax.set_title("Tiling scheme")

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MAIN BIT....

################## Now use same loop and instead of each plot, save each tile
################## When tiles are brought back together, just merge using a priority function as values should be the same
################## If using another convolution layer, make sure the overlaps are big enough to make sure there are no gaps...
import gdal, ogr, osr, os
import numpy as np
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

		#<<<<<<<<<<
		# first cell
		if (tl_easting_TILE_initial == tl_moving_window_min_easting_RANGE.min()) & (tl_northing_TILE_initial == tl_moving_window_max_northing_RANGE.max()):
			print("min easting and max northing!")
			tl_easting_TILE=tl_easting_TILE_initial
			br_easting_TILE=tl_easting_TILE+tile_dim_m
			tl_northing_TILE=tl_northing_TILE_initial
			br_northing_TILE=tl_northing_TILE-tile_dim_m
		# first column (not first row)
		elif ((tl_easting_TILE_initial == tl_moving_window_min_easting_RANGE.min()) & (tl_northing_TILE_initial != tl_moving_window_max_northing_RANGE.max())):
			print("min easting!")
			tl_easting_TILE=tl_easting_TILE_initial
			br_easting_TILE=tl_easting_TILE+tile_dim_m			
			tl_northing_TILE=tl_northing_TILE_initial+(tile_dim_m/2)
			br_northing_TILE=tl_northing_TILE-tile_dim_m	
		# first row (not first column)
		elif ((tl_northing_TILE_initial == tl_moving_window_max_northing_RANGE.max()) & (tl_easting_TILE_initial != tl_moving_window_min_easting_RANGE.min())):
			print("max northing!")
			tl_northing_TILE=tl_northing_TILE_initial
			br_northing_TILE=tl_northing_TILE-tile_dim_m		
			tl_easting_TILE=tl_easting_TILE_initial-(tile_dim_m/2)
			br_easting_TILE=tl_easting_TILE+tile_dim_m
		# all other cells
		elif ((tl_northing_TILE_initial != tl_moving_window_max_northing_RANGE.max()) & (tl_easting_TILE_initial != tl_moving_window_min_easting_RANGE.min())):
			print("NOT max northing OR min easting!")
			tl_easting_TILE=tl_easting_TILE_initial-(tile_dim_m/2)
			br_easting_TILE=tl_easting_TILE+tile_dim_m
			tl_northing_TILE=tl_northing_TILE_initial+(tile_dim_m/2)
			br_northing_TILE=tl_northing_TILE-tile_dim_m	

		#<<<<<<<<<<

		#<<< this was working...
		#br_easting_TILE=tl_easting_TILE+tile_dim_m/2
		#br_northing_TILE=tl_northing_TILE-tile_dim_m/2
		
		print("Tile %i" %count)

		dem_tile=dem_crop[np.int(np.floor(tl_easting_TILE)):np.int(np.ceil(br_easting_TILE)), 
					np.int(np.floor(br_northing_TILE)):np.int(np.ceil(tl_northing_TILE))]
	
		# for plotting
		tiles_x.append((tl_easting_TILE, br_easting_TILE,br_easting_TILE,tl_easting_TILE, tl_easting_TILE))
		tiles_y.append((br_northing_TILE,br_northing_TILE,tl_northing_TILE,tl_northing_TILE, br_northing_TILE))
	
		# Crop DEM to tile
		#convert eastings and northings to pixel....
		#tl_easting_TILE_px=
		#br_easting_TILE_px=
		#br_northing_TILE_px=
		#tl_northing_TILE_px=

		# Write raster
	#	newRasterfn = "%s/tile_%i.tif"%(out_path, count)
	#	array2raster(newRasterfn=newRasterfn,
	#				z_array=dem_tile,
	#				inDs=inDs,
	#				originX=tl_easting_TILE, 
	#				originY=tl_northing_TILE) 

		


print("COMPLETE")


# plot tiles over main extent

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
#ax.plot(tiles_x[21], tiles_y[21])
#ax.plot(tiles_x[22], tiles_y[22])
#ax.plot(tiles_x[23], tiles_y[23])
#ax.plot(tiles_x[24], tiles_y[24])

#ax.plot(tiles_x[8], tiles_y[8])
ax.axes.set_aspect('equal', 'datalim')
ax.set_title("Single tile locations")

* fix overlapping
	>> 3rd row onwrds isn;t overlapping...
* map ROUNDED tile northings and eastings to pixels
* fix array2raster code (look at your old scripts too)
* send to Andy


