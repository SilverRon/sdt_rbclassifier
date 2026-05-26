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
from multiprocessing import Pool
from tqdm import tqdm

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

def normalize_snapshot_mad_asinh(
    img,
    k=2.0,
    subtract_median=False,
    eps=1e-8,
    post_clip=None
):
    """
    Recommended normalization for real/bogus snapshot-based ML.

    Parameters
    ----------
    img : 2D numpy array
        Background-subtracted snapshot image.
    k : float, default=2.0
        asinh compression scale. (1.5–3 is a reasonable range)
    subtract_median : bool, default=False
        If True, subtract median(img) before scaling.
        Usually False if background is already ~0.
    eps : float
        Numerical stability term.
    post_clip : float or None
        If not None, apply symmetric clipping AFTER asinh, e.g. post_clip=3.

    Returns
    -------
    norm_img : 2D numpy array
        Normalized snapshot.
    """

    x = img.astype(np.float32, copy=False)

    # finite mask
    m = np.isfinite(x)
    if not np.any(m):
        return np.zeros_like(x, dtype=np.float32)

    # (optional) median subtraction
    if subtract_median:
        med = np.median(x[m])
        x = x - med

    # robust scale: MAD
    med = np.median(x[m])
    mad = np.median(np.abs(x[m] - med))
    sigma = 1.4826 * mad

    if sigma < eps:
        return np.zeros_like(x, dtype=np.float32)

    # robust linear scaling
    x = x / sigma

    # mild non-linearity (key step)
    x = np.arcsinh(x / k)

    # optional post-asinh clipping (VERY mild)
    if post_clip is not None:
        x = np.clip(x, -post_clip, post_clip)

    # clean non-finite
    x[~m] = 0.0

    return x

def process_snapshot(args):
    snap, path_tile_save, post_clip = args
    if "sci/" in snap:
        imagetyp = "sci"
    elif "sub/" in snap:
        imagetyp = "sub"
    else:
        imagetyp = "who_are_you"
        # print(imagetyp)

    norm_data = normalize_snapshot_mad_asinh(fits.getdata(snap), post_clip=post_clip)
    fits.writeto(f"{path_tile_save}/{imagetyp}/{os.path.basename(snap)}", norm_data, overwrite=True)

# %% 
path_save = f"../data/norm_snapshot"
os.makedirs(path_save, exist_ok=True)
paths_tile = sorted(glob.glob("../data/split/T?????"))
print(f"Found {len(paths_tile):,} tiles")
for ii, path_tile in enumerate(paths_tile):
	print(f"[{ii:>2}] {path_tile}")
	if ii >= 10:
		print(f"...")
		break

# %%
post_clip = 3

for pp, path_tile in enumerate(paths_tile):
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
    # Collect Images
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    path_tile_image_data = f"{path_tile}/???/image"
    tilename = os.path.basename(path_tile)
    path_tile_save = f"{path_save}/{tilename}"
    os.makedirs(path_tile_save, exist_ok=True)
    os.makedirs(f"{path_tile_save}/sci", exist_ok=True)
    os.makedirs(f"{path_tile_save}/sub", exist_ok=True)

    snaps = sorted(glob.glob(f"{path_tile_image_data}/calib*.fits"))
    print(f"Found {len(snaps):,} snapshots")
    for ii, snap in enumerate(snaps):
        print(f"[{ii:>2}] {snap}")
        if ii >= 10:
            print(f"...")
            break
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Normalize Snapshots
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Create arguments for parallel processing
    process_args = [(snap, path_tile_save, post_clip) for snap in snaps]

    # Use 30 cores or less if desired
    n_core = 10

    with Pool(n_core) as pool:
        list(tqdm(pool.imap(process_snapshot, process_args), total=len(snaps), desc=f"Processing {tilename}"))