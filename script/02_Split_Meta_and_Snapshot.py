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
import multiprocessing


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
# %%
# Function
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
def generate_snapshot(row, keys_vignet, image_size):
    n_array = image_size**2
    snap = np.zeros(n_array, dtype=float)
    for kk, key in enumerate(keys_vignet):
        snap[kk] = row[key]
    snap = snap.reshape((image_size, image_size))
    return snap

def process_catalog_pair(args):
    path_sci_cat, path_subt_cat, path_split_tile, reftbl_ra, reftbl_dec, snrcut, flagcut, r_match, keys_vignet, image_size = args
    
    # Reconstruct SkyCoord from passed arrays to avoid potential pickling issues with complex objects if any, 
    # though SkyCoord is generally pickleable. Passing arrays is safer for multiprocessing memory efficiency sometimes.
    c_ref = SkyCoord(reftbl_ra, reftbl_dec, unit='deg')

    result_stats = {}
    
    # Two sci & sub catalogs
    for cc, (path_cat, imagetyp) in enumerate(zip([path_sci_cat, path_subt_cat], ['sci', 'sub'])):
        intbl = Table.read(path_cat, format='ascii')
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        if path_cat == path_sci_cat:
            c_sci = SkyCoord(intbl['ALPHA_J2000'], intbl['DELTA_J2000'], unit='deg')
            indx_match, sep, _ = c_sci.match_to_catalog_sky(c_ref)
            indx_select = np.where(
                (intbl["SNR_WIN"] > snrcut) & (intbl["FLAGS"] == flagcut)
                    & (sep.arcsec < r_match)
                )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Apply 'SNR cut' Only for Subtracted image
        elif path_cat == path_subt_cat:
            indx_select = np.where(
                (intbl["SNR_WIN"] > snrcut)
                )
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
        n_total = len(intbl)
        seltbl = intbl[indx_select]
        n_select = len(seltbl)
        
        if imagetyp == 'sci':
            result_stats['n_original_sci'] = n_total
            result_stats['n_selected_sci'] = n_select
        elif imagetyp == 'sub':
            result_stats['n_original_sub'] = n_total
            result_stats['n_selected_sub'] = n_select
        
        # print(f"Number of selected sources ({imagetyp}): {n_select:,} ({n_select/n_total:.1%}) among {n_total:,}")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Snapshot Generation
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        prefix_meta_output = f"{path_split_tile}/{imagetyp}/meta/{os.path.basename(path_cat).replace('.final.cat', '')}"
        prefix_image_output = f"{path_split_tile}/{imagetyp}/image/{os.path.basename(path_cat).replace('.final.cat', '')}"
        os.makedirs(os.path.dirname(prefix_meta_output), exist_ok=True)
        os.makedirs(os.path.dirname(prefix_image_output), exist_ok=True)
        n_row_fig, n_col_fig = 10, 10
        n_max_fig = n_row_fig * n_col_fig
        fig = plt.figure(figsize=(15, 15))

        n_snap_count = 0
        n_fig_count = 0
        snapshotnames = []
        for nn, (number, fluxmax) in enumerate(zip(seltbl['NUMBER'], seltbl['FLUX_MAX'])):
            row = seltbl[nn]
            # Fits snap
            snapname = f"{prefix_image_output}_{number:0>6}.fits"
            snapshotnames.append(snapname)
            snap = generate_snapshot(row, keys_vignet, image_size)
            if os.path.exists(snapname):
                fits.writeto(snapname, snap, overwrite=True)
            else:
                pass
            # PNG mosaic
            n_snap_count += 1
            plt.subplot(n_row_fig, n_col_fig, n_snap_count)
            if fluxmax > 0:
                plt.imshow(snap, vmin=0, vmax=fluxmax)
            else:
                plt.imshow(snap)
            plt.text(0, 1, f"{number}", ha='left', va='top', color='w', fontsize=16)
            plt.xticks([])
            plt.yticks([])

            if n_snap_count >= n_max_fig:
                n_snap_count = 0
                plt.tight_layout()
                fig.savefig(f"{prefix_image_output}_mosaic_{n_fig_count:0>2}.png", dpi=100, bbox_inches="tight")
                plt.close(fig)
                n_fig_count += 1
                fig = plt.figure(figsize=(15, 15))

        # Fill remaining subplots
        if n_snap_count > 0:
            for n_rest in range(n_snap_count + 1, n_max_fig + 1):
                plt.subplot(n_row_fig, n_col_fig, n_rest)
                plt.axis('off')
            plt.tight_layout()
            fig.savefig(f"{prefix_image_output}_mosaic_{n_fig_count:0>2}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Extract Snapshot from Table
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        seltbl.remove_columns(keys_vignet)
        seltbl['snapshot'] = snapshotnames
        seltbl.write(f"{prefix_meta_output}.meta.cat", format='ascii', overwrite=True)
    
    return result_stats

import os
import math
import numpy as np
import matplotlib.pyplot as plt

def plot_snapshots_paged(
    seltbl,
    keys_vignet,
    image_size,
    prefix_output,
    n_per_fig=100,
    ncols=10,
    panel_in=1.5,          # 패널(서브플롯) 1개당 inch 크기
    cmap="gray",
    save_dir=None,         # None이면 저장 안 함
    dpi=150,
):
    """
    seltbl: astropy Table or pandas-like (seltbl['NUMBER'] 등 인덱싱 가능)
    n_per_fig: figure 한 장에 최대 몇 개를 그릴지
    ncols: 한 행에 몇 개를 둘지
    panel_in: 서브플롯 1개당 inch 크기(대략적인 가독성/해상도 스케일)
    """

    n_total = len(seltbl)
    if n_total == 0:
        return

    nrows = int(math.ceil(n_per_fig / ncols))
    figsize = (ncols * panel_in, nrows * panel_in)

    n_pages = int(math.ceil(n_total / n_per_fig))

    for page in range(n_pages):
        start = page * n_per_fig
        end = min((page + 1) * n_per_fig, n_total)
        n_this = end - start

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True
        )
        axes = np.atleast_1d(axes).ravel()

        # 페이지에 해당하는 것만 채우기
        for i, nn in enumerate(range(start, end)):
            row = seltbl[nn]
            number = row["NUMBER"]
            fluxmax = row["FLUX_MAX"]

            snap = generate_snapshot(row, keys_vignet, image_size)
            snapname = f"{prefix_output}_{int(number):0>6}.fits"
            title = os.path.basename(snapname)

            ax = axes[i]
            if fluxmax > 0:
                ax.imshow(snap, vmin=0, vmax=fluxmax, cmap=cmap, origin="lower")
            else:
                ax.imshow(snap, cmap=cmap, origin="lower")
            ax.set_title(title, fontsize=6)   # 100개면 글씨 작게
            ax.set_xticks([])
            ax.set_yticks([])

        # 남는 서브플롯은 제거(깔끔)
        for j in range(n_this, nrows * ncols):
            fig.delaxes(axes[j])

        # 필요하면 페이지 제목(선택)
        fig.suptitle(f"Snapshots ({start+1}-{end} / {n_total})", fontsize=12)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            outpath = os.path.join(save_dir, f"{prefix_output}_page{page+1:03d}.png")
            fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        else:
            plt.show()

        plt.close(fig)

# %%
# Data
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
path_raw_data = "../data/raw"
path_split_data = "../data/split"
path_ref_cat = "../data/ref_cat"

summary_table = pd.read_csv(f"{path_raw_data}/after_run_images.csv")
summary_table['run_status'].value_counts()

suc_table = summary_table[summary_table['run_status'] == 'S'] # Successed images

# %%
# Fill out the summary table
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
paths_sci_cat = []
paths_subt_cat = []
exists_sci_cat = []
exists_subt_cat = []
# 
ss = 0
path_image = suc_table['path_image'][ss]
path_subtimage = suc_table['path_subtimage'][ss]
tilename = suc_table['tile'][ss]

verbose = False

for ss, (path_image, path_subtimage, tilename) in enumerate(zip(suc_table['path_image'], suc_table['path_subtimage'], suc_table['tile'])):    
    # 
    if verbose: 
        print(f"SCI IMAGE: {path_image}")
        print(f"SUB IMAGE: {path_subtimage}")
    # 
    path_sci_cat = f"{path_raw_data}/{tilename}/sci/{os.path.basename(path_image).replace('.fits', '.final.cat')}"
    path_subt_cat = f"{path_raw_data}/{tilename}/sub/{os.path.basename(path_subtimage).replace('.fits', '.final.cat')}"

    exist_sci_cat = os.path.exists(path_sci_cat)
    exist_subt_cat = os.path.exists(path_subt_cat)

    if verbose: 
        print(f"SCI CAT: {path_sci_cat} -> {exist_sci_cat}")
        print(f"SUB CAT: {path_subt_cat} -> {exist_subt_cat}")

    paths_sci_cat.append(path_sci_cat)
    paths_subt_cat.append(path_subt_cat)
    exists_sci_cat.append(exist_sci_cat)
    exists_subt_cat.append(exist_subt_cat)

suc_table['path_sci_cat'] = paths_sci_cat
suc_table['path_subt_cat'] = paths_subt_cat
suc_table['exist_sci_cat'] = exists_sci_cat
suc_table['exist_subt_cat'] = exists_subt_cat

# Summary Table
sumtbl = suc_table[suc_table['exist_sci_cat'] & suc_table['exist_subt_cat']]
print(f"Number of valid data: {len(sumtbl)} among {len(suc_table)}")

# VIGNET KEYS with dummy file
# Extract VIGNET columns to 2D array

keys_vignet = []
for key in Table.read(path_sci_cat, format='ascii').keys():
    if 'VIGNET' in key:
        keys_vignet.append(key)
print(f"Found {len(keys_vignet)} VIGNET keys")
# %%
# Split Image and Meta Data
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# Parameters
snrcut = 5
image_size = 25
flagcut = 0
r_match = 1.0
# 
n_cores = 10
# 
tiles = np.unique(sumtbl['tile'])
print(f"Found {len(tiles):,} tiles")
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Tile Iteration
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
for ii, tile in enumerate(tiles):
    path_split_tile = f"{path_split_data}/{tile}"
    final_summary_table = f"{path_split_tile}/summary.csv"
    if os.path.exists(final_summary_table):
        print(f"{final_summary_table} already exists")
        continue
    os.makedirs(path_split_tile, exist_ok=True)

    sub_sumtbl = sumtbl[sumtbl['tile'] == tile].copy()
    n_subtable = len(sub_sumtbl)
    print(f"[{ii:>2}] {tile} -> {n_subtable:,}")
    refcat = f"{path_ref_cat}/gaiaxp_dr3_synphot_{tile}.csv"
    exist_refcat = os.path.exists(refcat)
    if exist_refcat:
        print(f"{refcat} found")
        reftbl = pd.read_csv(refcat)
        c_ref = SkyCoord(reftbl['ra'], reftbl['dec'], unit='deg')
    else:
        print(f"Reference catalog not found: {refcat}")
        continue

    numbers_original_sci = []
    numbers_selected_sci = []
    numbers_original_sub = []
    numbers_selected_sub = []
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Catalog Iteration (multiprocessing)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    processing_args = []
    for path_sci_cat, path_subt_cat in zip(sub_sumtbl['path_sci_cat'], sub_sumtbl['path_subt_cat']):
        processing_args.append((
            path_sci_cat, 
            path_subt_cat, 
            path_split_tile, 
            reftbl['ra'].values, 
            reftbl['dec'].values, 
            snrcut, 
            flagcut, 
            r_match, 
            keys_vignet, 
            image_size
        ))

    print(f"Starting multiprocessing with {n_cores} cores for {len(processing_args)} items...")
    
    with multiprocessing.Pool(n_cores) as pool:
        results = pool.map(process_catalog_pair, processing_args)
    
    # Collect results
    for res in results:
        numbers_original_sci.append(res['n_original_sci'])
        numbers_selected_sci.append(res['n_selected_sci'])
        numbers_original_sub.append(res['n_original_sub'])
        numbers_selected_sub.append(res['n_selected_sub'])


    # Add Statistics
    sub_sumtbl['n_original_sci'] = numbers_original_sci
    sub_sumtbl['n_selected_sci'] = numbers_selected_sci
    sub_sumtbl['n_original_sub'] = numbers_original_sub
    sub_sumtbl['n_selected_sub'] = numbers_selected_sub

    sub_sumtbl.to_csv(final_summary_table, index=False)
