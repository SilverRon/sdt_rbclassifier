# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# Setting
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# %%
import os, sys, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")

# Astro
from astropy.table import Table, vstack, hstack
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.wcs.utils import proj_plane_pixel_scales

# Matplotlib global settings
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# Data
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# %%
path_data = f"/data/data1/processed_1x1_gain2750"
path_save = "../data/raw"
if os.path.exists(path_save) == False:
	os.makedirs(path_save)
path_sex_config = f"../config/SourceEXtractor"
path_psfex_config = f"../config/PSFEx"

# (old) IMS tiles
IMS_tiles = [
	"T00138",
	"T00139",
	"T00174",
	"T00175",
	"T00176",
	"T00215",
	"T00216"
]

print(f"Found {len(IMS_tiles):,} IMS_tiles")
for ii, tile in enumerate(IMS_tiles):
	print(f"[{ii:>2}] {tile}")
	if ii >= 10:
		print(f"...")
		break
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Images
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# %%
# image_dict = {}

images = []

for ii, tile in enumerate(IMS_tiles):
	tileimages = glob.glob(f"{path_data}/{tile}/7DT??/*/calib*com.fits")
	print(f"[{ii:>2}] {tile} ({len(tileimages):,} images found)")
	# image_dict[tile] = tileimages
	images += tileimages

images.sort()
print(f"Total {len(images):,} images found")
# %%

rows = []

for ii, image in enumerate(images):
	basename = os.path.basename(image)
	parts = basename.split("_")
	# 
	subtimage = image.replace(".com.fits", ".com.subt.fits")
	exist_subtimage = os.path.exists(subtimage)
	unitname = parts[1]
	tilename = parts[2]
	datename = parts[3]
	timename = parts[4]
	filtername = parts[5]
	exptime = int(parts[6].split(".")[0])
	# 
	row = dict(
		path_image = image,
		path_subtimage = subtimage,
		exist_subtimage = exist_subtimage,
		# 
		image = basename,
		unit = unitname,
		tile = tilename,
		date = datename,
		time = timename,
		filter = filtername,
		exptime = exptime,
	)

	rows.append(row)
outbl = Table(rows=rows)

n_valid = len(outbl[outbl['exist_subtimage']])
print(f"Found {n_valid:,} ({n_valid/len(images):.1%}%) valid images among {len(images):,} images")

valtbl = outbl[outbl['exist_subtimage']]

for tt, tile in enumerate(IMS_tiles):
	print(f"[{tt:>2}] {tile} ({len(valtbl[valtbl['tile'] == tile]):,} valid images)")
	

# %%
outbl.write(f"{path_save}/images.csv", overwrite=True)
valtbl.write(f"{path_save}/images_subt_exist.csv", overwrite=True)
outbl[:3]
# %%
