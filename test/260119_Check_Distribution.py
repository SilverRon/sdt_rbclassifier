# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# %%
# Library
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
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
# %%
# Function
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# %% 
# Data
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
path_data = f"/data/data1/processed_1x1_gain2750"
path_save = f"../data/stacked_meta"
os.makedirs(path_save, exist_ok=True)
paths_tile = sorted(glob.glob("../data/split/T?????"))
print(f"Found {len(paths_tile):,} tiles")
for ii, path_tile in enumerate(paths_tile):
	print(f"[{ii:>2}] {path_tile}")
	if ii >= 10:
		print(f"...")
		break

# %%
pp = 0
path_tile = paths_tile[pp]
tilename = os.path.basename(path_tile)
print(f"[{pp:>2}] {path_tile}")

summary_table = Table.read(f"{path_tile}/summary.csv")
n_real = np.sum(summary_table['n_selected_sci'])
n_bogus = np.sum(summary_table['n_selected_sub'])

print(f"Number of real sources : {n_real:,}")
print(f"Number of bogus sources: {n_bogus:,}")
print(f"Total number of sources: {n_real + n_bogus:,}")
print(f"Ratio of real sources  : {n_real / (n_real + n_bogus):.1%}")
print(f"Ratio of bogus sources : {n_bogus / (n_real + n_bogus):.1%}")
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# %%
# Collect Catalogs
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
path_tile_catalog_scidata = f"{path_tile}/sci/meta"
path_tile_catalog_subdata = f"{path_tile}/sub/meta"

rcats = sorted(glob.glob(f"{path_tile_catalog_scidata}/*.meta.cat"))
bcats = sorted(glob.glob(f"{path_tile_catalog_subdata}/*.meta.cat"))

cats = rcats+bcats
print(f"Found {len(cats):,} catalogs")
print(f"- {len(rcats):,} real catalogs")
print(f"- {len(bcats):,} bogus catalogs")

check_all_match = True

for ii, (rcat, bcat) in enumerate(zip(rcats, bcats)):
    basename_rcat = os.path.basename(rcat)
    basename_bcat = os.path.basename(bcat)
    check_match = basename_rcat == basename_bcat.replace(".subt", "")
    if check_match:
        # print(f"[{ii:>2}] {basename_rcat},{basename_bcat} (MATCH)")
        pass
    else:
        print(f"[{ii:>2}] {basename_rcat},{basename_bcat} (MISMATCH)")
        check_all_match = False
    # if ii >= 10:
    if ii == len(rcats)-1:
        # print(f"...")
        keys = Table.read(bcat, format='ascii').keys()
        break

if check_all_match:
    print("All catalogs match")
else:
    print("Not all catalogs match")
# %%
keys_for_record = [
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # FLUX & MAG
    "FLUX_ISO", "FLUXERR_ISO", "MAG_ISO", "MAGERR_ISO", 
    "FLUX_AUTO", "FLUXERR_AUTO", "MAG_AUTO", "MAGERR_AUTO",
    "FLUX_PETRO", "FLUXERR_PETRO", "MAG_PETRO", "MAGERR_PETRO",
    "SNR_WIN", 
    "MU_THRESHOLD",
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # (X,Y), (RA,DEC)
    "XPEAK_IMAGE", "YPEAK_IMAGE", "X_IMAGE", "Y_IMAGE",
    "XWIN_IMAGE", "YWIN_IMAGE",
    "ALPHA_J2000", "DELTA_J2000", "ALPHAPEAK_J2000", "DELTAPEAK_J2000", 
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Morphology
    "ELONGATION", "ELLIPTICITY", # -> Convert to ratio
    "CLASS_STAR",
    "FWHM_WORLD", "FWHM_IMAGE", # -> Convert to ratio
    "A_IMAGE", "B_IMAGE", "AWIN_IMAGE", "BWIN_IMAGE",
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Value
    "BACKGROUND",
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # ETC
    "NUMBER", "snapshot",
]
keys_for_ML = [key for key in keys if key not in keys_for_record]
print(f"Keywords for ML: {len(keys_for_ML)}")
for kk, key in enumerate(keys_for_ML):
    print(f"{key}," if (kk % 8 != 0) | (kk == 0) else f"{key},\n", end='')
# %%
# Read Tables
rintbl = Table.read(rcat, format='ascii')
bintbl = Table.read(bcat, format='ascii')
# %%
key = "BACKGROUND"
bins_bkg = np.arange(-0.2, +0.3+0.01, 0.01)
plt.hist(rintbl[key], bins=bins_bkg, histtype='step', label='real', density=True)
plt.hist(bintbl[key], bins=bins_bkg, histtype='step', label='bogus', density=True)
plt.xlabel(key, fontsize=10)
plt.legend()
plt.show()
# %%
parts = os.path.basename(rcat).split("_")
unitname = parts[1]
filtername = parts[5]
print(unitname, filtername)

path_image = f"{path_data}/{tilename}/{unitname}/{filtername}/{os.path.basename(rcat).replace('.meta.cat', '.fits')}"
print(path_image, os.path.exists(path_image))
# %%
# keys_to_check = [
#     "ELLIPTICITY", 
#     "FWHM_IMAGE", 
#     "ELONGATION", 
#     "FLAGS", 
#     "BACKGROUND", 
#     "CLASS_STAR",
#     "MU_MAX", "FLUX_MAX"
#     ]
keys_to_check = keys_for_ML
fig = plt.figure(figsize=(20, 20))
for kk, key in enumerate(keys_to_check):
    ax = fig.add_subplot(7, 7, kk+1)
    plt.hist(rintbl[key], bins=30, histtype='step', label='real', density=True)
    plt.hist(bintbl[key], bins=30, histtype='step', label='bogus', density=True)
    plt.xlabel(key, fontsize=10)
    # plt.ylabel("Density")
    plt.legend()
plt.tight_layout()
plt.show()
# 
feature_engineering_dict = {
    "RATIO_ELLIPTICITY": "ELLIPTICITY",
    "RATIO_FWHM_IMAGE": "FWHM_IMAGE",

}
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# %%
r_ximages, r_yimages, r_xpeaks, r_ypeaks = rintbl['X_IMAGE'], rintbl['Y_IMAGE'], rintbl['XPEAK_IMAGE'], rintbl['YPEAK_IMAGE']
b_ximages, b_yimages, b_xpeaks, b_ypeaks = bintbl['X_IMAGE'], bintbl['Y_IMAGE'], bintbl['XPEAK_IMAGE'], bintbl['YPEAK_IMAGE']

r_offsets = np.sqrt((r_ximages-r_xpeaks)**2 + (r_yimages-r_ypeaks)**2)
b_offsets = np.sqrt((b_ximages-b_xpeaks)**2 + (b_yimages-b_ypeaks)**2)

bins_offset = np.arange(0, 10+0.1, 0.1)

plt.hist(r_offsets, bins=bins_offset, histtype='step', label='real', density=True)
plt.hist(b_offsets, bins=bins_offset, histtype='step', label='bogus', density=True)
plt.xlabel("Offset",)
plt.legend()
plt.show()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# %%
margin = 1e-3
rvals = rintbl['FLUX_MAX'] / (rintbl['MU_MAX']+margin)
bvals = bintbl['FLUX_MAX'] / (bintbl['MU_MAX']+margin)

bins_flux_mu = np.arange(-100, 0+2.5, 2.5)

plt.hist(rvals, bins=bins_flux_mu, histtype='step', label='real', density=True)
plt.hist(bvals, bins=bins_flux_mu, histtype='step', label='bogus', density=True)
plt.xlabel("Ratio",)
plt.legend()
plt.show()
# %%
r_ellip_moment = abs(rintbl['X2_IMAGE'] - rintbl['Y2_IMAGE']) / (rintbl['X2_IMAGE'] + rintbl['Y2_IMAGE'])
r_shear_moment = abs(rintbl['XY_IMAGE']) / np.sqrt(rintbl['X2_IMAGE'] * rintbl['Y2_IMAGE'])
b_ellip_moment = abs(bintbl['X2_IMAGE'] - bintbl['Y2_IMAGE']) / (bintbl['X2_IMAGE'] + bintbl['Y2_IMAGE'])
b_shear_moment = abs(bintbl['XY_IMAGE']) / np.sqrt(bintbl['X2_IMAGE'] * bintbl['Y2_IMAGE'])

r_ellip_cxx = abs(rintbl['CXX_IMAGE'] - rintbl['CYY_IMAGE']) / (rintbl['CXX_IMAGE'] + rintbl['CYY_IMAGE'])
r_shear_cxx = abs(rintbl['CXY_IMAGE']) / np.sqrt(rintbl['CXX_IMAGE'] * rintbl['CYY_IMAGE'])
b_ellip_cxx = abs(bintbl['CXX_IMAGE'] - bintbl['CYY_IMAGE']) / (bintbl['CXX_IMAGE'] + bintbl['CYY_IMAGE'])
b_shear_cxx = abs(bintbl['CXY_IMAGE']) / np.sqrt(bintbl['CXX_IMAGE'] * bintbl['CYY_IMAGE'])

plt.hist(r_ellip_moment, bins=30, histtype='step', label='real', density=True)
plt.hist(b_ellip_moment, bins=30, histtype='step', label='bogus', density=True)
plt.xlabel("Ellipticity",)
plt.legend()
plt.show()

plt.hist(r_ellip_cxx, bins=30, histtype='step', label='real', density=True)
plt.hist(b_ellip_cxx, bins=30, histtype='step', label='bogus', density=True)
plt.xlabel("Ellipticity",)
plt.legend()
plt.show()
# %%
r_kron_flux_ratio  = rintbl['KRON_RADIUS'] / rintbl['FLUX_RADIUS']
r_petro_flux_ratio = rintbl['PETRO_RADIUS'] / rintbl['FLUX_RADIUS']
r_iso_norm_area    = rintbl['ISOAREA_WORLD'] / rintbl['FLUX_RADIUS']**2

b_kron_flux_ratio  = bintbl['KRON_RADIUS'] / bintbl['FLUX_RADIUS']
b_petro_flux_ratio = bintbl['PETRO_RADIUS'] / bintbl['FLUX_RADIUS']
b_iso_norm_area    = bintbl['ISOAREA_WORLD'] / bintbl['FLUX_RADIUS']**2

plt.hist(r_kron_flux_ratio, bins=30, histtype='step', label='real', density=True)
plt.hist(b_kron_flux_ratio, bins=30, histtype='step', label='bogus', density=True)
plt.xlabel("Kron/Flux",)
plt.legend()
plt.show()

plt.hist(r_petro_flux_ratio, bins=30, histtype='step', label='real', density=True)
plt.hist(b_petro_flux_ratio, bins=30, histtype='step', label='bogus', density=True)
plt.xlabel("Petro/Flux",)
plt.legend()
plt.show()
# %%
r_iso_gradient = np.max(np.abs(rintbl['ISOAREA_WORLD'] - rintbl['ISOAREA_WORLD'].shift(1)), axis=0)
b_iso_gradient = np.max(np.abs(bintbl['ISOAREA_WORLD'] - bintbl['ISOAREA_WORLD'].shift(1)), axis=0)

plt.hist(r_iso_gradient, bins=30, histtype='step', label='real', density=True)
plt.hist(b_iso_gradient, bins=30, histtype='step', label='bogus', density=True)
plt.xlabel("ISO Gradient",)
plt.legend()
plt.show()
# %%
r_theta_diff = abs(rintbl['THETA_IMAGE'] - rintbl['THETAWIN_IMAGE'])
b_theta_diff = abs(bintbl['THETA_IMAGE'] - bintbl['THETAWIN_IMAGE'])

plt.hist(r_theta_diff, bins=30, histtype='step', label='real', density=True)
plt.hist(b_theta_diff, bins=30, histtype='step', label='bogus', density=True)
plt.xlabel("Theta Diff",)
plt.legend()
plt.show()
# %%
r_theta_sin = np.sin(2 * rintbl['THETA_IMAGE'])
b_theta_sin = np.sin(2 * bintbl['THETA_IMAGE'])

plt.hist(r_theta_sin, bins=30, histtype='step', label='real', density=True)
plt.hist(b_theta_sin, bins=30, histtype='step', label='bogus', density=True)
plt.xlabel("Theta Sin",)
plt.legend()
plt.show()
# %%
for nn in range(6):
    r_values = rintbl[f'ISO{nn}'] - rintbl[f'ISO{nn+1}']
    b_values = bintbl[f'ISO{nn}'] - bintbl[f'ISO{nn+1}']
    plt.hist(r_values, bins=30, histtype='step', label='real', density=True)
    plt.hist(b_values, bins=30, histtype='step', label='bogus', density=True)
    plt.xlabel(f"ISO {nn} - ISO {nn+1}",)
    plt.legend()
    plt.show()
# %%
r_norm_flux_radius = rintbl['FLUX_RADIUS'] / (rintbl['FWHM_IMAGE']+1e-3)
r_norm_kron        = rintbl['KRON_RADIUS'] / (rintbl['FWHM_IMAGE']+1e-3)

b_norm_flux_radius = bintbl['FLUX_RADIUS'] / (bintbl['FWHM_IMAGE']+1e-3)
b_norm_kron        = bintbl['KRON_RADIUS'] / (bintbl['FWHM_IMAGE']+1e-3)

bins_nradius = np.arange(0, 10+0.1, 0.1)

plt.hist(r_norm_flux_radius, bins=bins_nradius, histtype='step', label='real', density=True)
plt.hist(b_norm_flux_radius, bins=bins_nradius, histtype='step', label='bogus', density=True)
plt.xlabel("Norm Flux Radius",)
plt.legend()
plt.show()

plt.hist(r_norm_kron, bins=bins_nradius, histtype='step', label='real', density=True)
plt.hist(b_norm_kron, bins=bins_nradius, histtype='step', label='bogus', density=True)
plt.xlabel("Norm Kron",)
plt.legend()
plt.show()

