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
from multiprocessing import Pool
from functools import partial
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
from astropy.stats import sigma_clip

# Matplotlib global settings
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# %%
# Function
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 

def robust_hist(ax, x, bins=30, p=(1,99), log=False):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return

    lo, hi = np.percentile(x, p)
    x = x[(x >= lo) & (x <= hi)]

    if log:
        x = x[x > 0]
        x = np.log10(x)

    ax.hist(
        x, bins=bins,
        histtype="step", linewidth=1.5,
        density=True,
    )

def process_catalog_pair(pair):
    """
    Process a single (real, bogus) catalog pair: feature engineering and splitting.
    """
    rcat, bcat = pair
    try:
        rintbl = Table.read(rcat, format='ascii')
        bintbl = Table.read(bcat, format='ascii')
    except Exception as e:
        print(f"Error reading {rcat} or {bcat}: {e}")
        return None

    # Labels
    rintbl['label'] = 'r' # real
    bintbl['label'] = 'b' # bogus

    # Robust type conversion for columns that may have mixed types across catalogs (e.g. snapshot)
    for colname in ['snapshot']:
        if colname in rintbl.colnames:
            rintbl[colname] = rintbl[colname].astype(str)
        if colname in bintbl.colnames:
            bintbl[colname] = bintbl[colname].astype(str)

    intbl = vstack([rintbl, bintbl])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # AS IS
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    keys_as_is = [
        "CXX_IMAGE", "CYY_IMAGE", "CXY_IMAGE", "THETA_IMAGE",
        "XY_IMAGE", "FLAGS",
    ]
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # RATIO
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    keys_engineering = []
    basekey = "ISO0"
    for k in np.arange(1, 8, 1):
        key = f"ISO{k:g}"
        if key in intbl.keys() and basekey in intbl.keys():
            ratiokey = f"RATIO_{key}_{basekey}"
            intbl[ratiokey] = np.asarray(intbl[key]) / np.asarray(intbl[basekey])
            keys_engineering.append(ratiokey)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Position
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    xbasekey, ybasekey = "XPEAK_IMAGE", "YPEAK_IMAGE"
    middle_position_keys = ["", "PSF", "WIN"]
    for mkey in middle_position_keys:
        xposkey, yposkey = f"X{mkey}_IMAGE", f"Y{mkey}_IMAGE"
        if xposkey in intbl.keys() and yposkey in intbl.keys():
            dist = np.sqrt((intbl[xposkey] - intbl[xbasekey])**2 + (intbl[yposkey] - intbl[ybasekey])**2)
            distkey = f"DIST_{mkey}" if mkey != "" else "DIST"
            intbl[distkey] = dist
            keys_engineering.append(distkey)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Size
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    basesizekey = "FWHM_IMAGE"
    sizekeys = ["KRON_RADIUS", "PETRO_RADIUS", "FLUX_RADIUS", "FWHMPSF_IMAGE"]
    for skey in sizekeys:
        if skey in intbl.keys() and basesizekey in intbl.keys():
            ratiokey = f"RATIO_{skey}_{basesizekey}"
            intbl[ratiokey] = np.asarray(intbl[skey]) / (np.asarray(intbl[basesizekey]) + 1e-6)
            keys_engineering.append(ratiokey)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # FLUX RATIO
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    basefluxkey = "FLUX_MAX"
    fluxkeys = ["ISO", "AUTO", "PETRO", "PSF"]
    for fkey in fluxkeys:
        colkey = f"FLUX_{fkey}"
        if colkey in intbl.keys() and basefluxkey in intbl.keys():
            ratiokey = f"RATIO_{fkey}_{basefluxkey}"
            intbl[ratiokey] = np.asarray(intbl[colkey]) / (np.asarray(intbl[basefluxkey]) + 1e-6)
            keys_engineering.append(ratiokey)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Per-Image Normalization
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    keys_perimage = ["ELONGATION", "ELLIPTICITY", "X2_IMAGE", "Y2_IMAGE"]
    for key in keys_perimage:
        if key in rintbl.keys():
            values = np.asarray(rintbl[key], dtype=float)
            values = values[np.isfinite(values)]
            if len(values) > 0:
                clipped = sigma_clip(values, sigma=3, maxiters=5)
                median_value = np.nanmedian(clipped)
                if median_value != 0 and np.isfinite(median_value):
                    ratiokey = f"RATIO_{key}"
                    intbl[ratiokey] = np.asarray(intbl[key]) / median_value
                    keys_engineering.append(ratiokey)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    keys_for_ML = keys_as_is + keys_engineering
    new_all_keys = intbl.keys()
    keys_for_record = [key for key in new_all_keys if key not in keys_for_ML]

    # Split again
    rout_ml = intbl[intbl['label'] == 'r'][keys_for_ML]
    rout_rec = intbl[intbl['label'] == 'r'][keys_for_record]
    bout_ml = intbl[intbl['label'] == 'b'][keys_for_ML]
    bout_rec = intbl[intbl['label'] == 'b'][keys_for_record]

    return (rout_ml, rout_rec, bout_ml, bout_rec)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# %% 
# Main
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
if __name__ == "__main__":
    n_core = 10
    path_data = f"/data/data1/processed_1x1_gain2750"
    path_save = f"../data/stacked_meta"
    os.makedirs(path_save, exist_ok=True)
    paths_tile = sorted(glob.glob("../data/split/T?????"))
    print(f"Found {len(paths_tile):,} tiles")

    for ii, path_tile in enumerate(paths_tile):
        tilename = os.path.basename(path_tile)
        print(f"\nProcessing Tile [{ii:>2}]: {tilename}")
        
        # Path
        path_save_tile = f"{path_save}/{tilename}"
        path_save_tile_check = f"{path_save_tile}/check"
        os.makedirs(path_save_tile_check, exist_ok=True)

        summary_file = f"{path_tile}/summary.csv"
        if os.path.exists(summary_file):
            summary_table = Table.read(summary_file)
            n_real = np.sum(summary_table['n_selected_sci'])
            n_bogus = np.sum(summary_table['n_selected_sub'])
            print(f"  Real sources: {n_real:,} | Bogus sources: {n_bogus:,}")
        
        # Collect Catalogs
        rcats = sorted(glob.glob(f"{path_tile}/sci/meta/*.meta.cat"))
        bcats = sorted(glob.glob(f"{path_tile}/sub/meta/*.meta.cat"))
        
        # Matching
        pairs = []
        for rcat in rcats:
            basename = os.path.basename(rcat)
            expected_bcat = f"{path_tile}/sub/meta/{basename.replace('.meta.cat', '.subt.meta.cat')}"
            if os.path.exists(expected_bcat):
                pairs.append((rcat, expected_bcat))
        
        if not pairs:
            print(f"  No matching catalogs found for tile {tilename}")
            continue
            
        print(f"  Processing {len(pairs):,} catalog pairs with {n_core} cores...")
        
        # Parallel processing
        with Pool(n_core) as pool:
            results = pool.map(process_catalog_pair, pairs)
        
        # Filter None
        results = [r for r in results if r is not None]
        if not results:
            continue
            
        # Aggregate results
        tables_real_ml = [r[0] for r in results]
        tables_real_record = [r[1] for r in results]
        tables_bogus_ml = [r[2] for r in results]
        tables_bogus_record = [r[3] for r in results]
        
        print(f"  Stacking and saving results...")
        fin_rintbl4ml = vstack(tables_real_ml)
        fin_bintbl4ml = vstack(tables_bogus_ml)
        fin_rintbl4record = vstack(tables_real_record)
        fin_bintbl4record = vstack(tables_bogus_record)

        fin_rintbl4ml.write(f"{path_save_tile}/real_ml.parquet", format="parquet", overwrite=True)
        fin_bintbl4ml.write(f"{path_save_tile}/bogus_ml.parquet", format="parquet", overwrite=True)
        fin_rintbl4record.write(f"{path_save_tile}/real_record.cat", format="ascii", overwrite=True)
        fin_bintbl4record.write(f"{path_save_tile}/bogus_record.cat", format="ascii", overwrite=True)

        # Histogram
        print(f"  Generating summary histogram...")
        keys_for_ML = fin_rintbl4ml.keys()
        fig = plt.figure(figsize=(40, 20))
        for kk, key in enumerate(keys_for_ML):
            ax = plt.subplot(4, 7, kk + 1)
            robust_hist(ax, fin_rintbl4ml[key], log=False)
            robust_hist(ax, fin_bintbl4ml[key], log=False)
            ax.legend(["real", "bogus"], fontsize=12)
            ax.set_title(key, fontsize=14)
            ax.set_ylabel("density", fontsize=12)
            ax.set_xlabel(key, fontsize=12)

        plt.tight_layout()
        plt.savefig(f"{path_save_tile_check}/histogram.png")
        plt.close()
        
        print(f"  Tile {tilename} completed.")