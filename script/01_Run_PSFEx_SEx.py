# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# Setting
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
import os, sys
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd
from astropy.table import Table

# Add src to sys.path to allow imports
# Assuming script is run from script/ directory, src is in ../src
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.psfex import run_sex_psfex_sex, run_sex_only, move_check_images, PsfexPipelineOutputs, SexOnlyOutputs

warnings.filterwarnings("ignore")

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# Configuration
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
PATH_SAVE = Path('../data/raw').resolve()
PATH_CONFIG = Path('../config').resolve()

# Config files
CONFIG = {
    "pre_sex_conf": PATH_CONFIG / "PSFEx/prepsfex.sex",
    "pre_param": PATH_CONFIG / "PSFEx/prepsfex.param",
    "psfex_conf": PATH_CONFIG / "PSFEx/default.psfex",
    "post_sex_conf": PATH_CONFIG / "SourceExtractor/default.sex",
    "post_param": PATH_CONFIG / "SourceExtractor/default.param",
    "default_conv": PATH_CONFIG / "SourceExtractor/default.conv",
    "default_nnw": PATH_CONFIG / "SourceExtractor/default.nnw",
}

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# Worker Function
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
def process_one_image_set(row_data: dict) -> dict:
    """
    Process a single image and its subtraction image.
    Returns dictionary with updated path information or error info.
    """
    image_path = Path(row_data['path_image'])
    subt_path = Path(row_data['path_subtimage'])
    tile = row_data['tile']
    
    # Work directory based on tile
    workdir = PATH_SAVE / tile
    
    # Naming conventions
    # The original script used prefix based on basename
    prefix_image = image_path.stem
    pre_cat_name = f"{prefix_image}.pre.cat"
    psf_name = f"{prefix_image}.psf"
    final_cat_name = f"{prefix_image}.final.cat"
    sub_final_cat_name = f"{subt_path.stem}.final.cat"

    result = {
        'idx': row_data['idx'],
        'path_ex_cat': None,
        'path_ex_psf': None,
        'path_final_cat': None,
        'path_sub_cat': None,
        'status': 'FAILED',
        'error_msg': ''
    }

    try:
        # 1. Run Pipeline for Main Image
        out_main: PsfexPipelineOutputs = run_sex_psfex_sex(
            image_path=image_path,
            workdir=workdir,
            pre_sex_conf=CONFIG["pre_sex_conf"],
            pre_param=CONFIG["pre_param"],
            psfex_conf=CONFIG["psfex_conf"],
            post_sex_conf=CONFIG["post_sex_conf"],
            post_param=CONFIG["post_param"],
            default_conv=CONFIG["default_conv"],
            default_nnw=CONFIG["default_nnw"],
            pre_cat_name=pre_cat_name,
            psf_name=psf_name,
            final_cat_name=final_cat_name
        )
        
        # Move check images if any (samp, snap, resi)
        # Note: We check 'sci' dir for them.
        check_dir = workdir / "check"
        move_check_images(out_main.workdir, check_dir, image_path)

        # 2. Run Pipeline for Subtraction Image (SEx only)
        # using the PSF from the main run
        out_sub: SexOnlyOutputs = run_sex_only(
            image_path=subt_path,
            workdir=workdir,
            psf_path=out_main.psf,
            sex_conf=CONFIG["post_sex_conf"],
            param=CONFIG["post_param"],
            default_conv=CONFIG["default_conv"],
            default_nnw=CONFIG["default_nnw"],
            final_cat_name=sub_final_cat_name
        )
        
        # Move check images for sub if any
        move_check_images(out_sub.workdir, check_dir, subt_path)

        result['path_ex_cat'] = str(out_main.cat_pre)
        result['path_ex_psf'] = str(out_main.psf)
        result['path_final_cat'] = str(out_main.cat_final)
        result['path_sub_cat'] = str(out_sub.cat_final)
        result['status'] = 'SUCCESS'

    except Exception as e:
        result['error_msg'] = str(e)
        # print(f"Error processing {image_path}: {e}") # Optional: keep logs clean

    return result

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# Main
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
if __name__ == '__main__':
    # Load Data
    input_table_path = PATH_SAVE / "images_subt_exist.csv"
    if not input_table_path.exists():
        raise FileNotFoundError(f"Input table not found: {input_table_path}")
        
    intbl = Table.read(str(input_table_path))
    
    # Prepare rows for parallel processing
    rows_to_process = []
    for i, row in enumerate(intbl):
        row_dict = {
            'idx': i,
            'path_image': row['path_image'],
            'path_subtimage': row['path_subtimage'],
            'tile': row['tile']
        }
        rows_to_process.append(row_dict)

    # Initialize new columns
    num_rows = len(intbl)
    intbl['path_ex_cat'] = [''] * num_rows
    intbl['path_ex_psf'] = [''] * num_rows
    intbl['path_final_cat'] = [''] * num_rows
    intbl['path_sub_cat'] = [''] * num_rows
    intbl['run_status'] = [''] * num_rows
    intbl['error_msg'] = [''] * num_rows

    # Run Parallel
    N_CORES = 10
    print(f"Running on {N_CORES} cores...")
    
    with Pool(processes=N_CORES) as pool:
        results = pool.map(process_one_image_set, rows_to_process)
        
    # Collect results
    print("Collecting results...")
    count_success = 0
    for res in results:
        idx = res['idx']
        if res['status'] == 'SUCCESS':
            intbl['path_ex_cat'][idx] = res['path_ex_cat']
            intbl['path_ex_psf'][idx] = res['path_ex_psf']
            intbl['path_final_cat'][idx] = res['path_final_cat']
            intbl['path_sub_cat'][idx] = res['path_sub_cat']
            intbl['run_status'][idx] = 'SUCCESS'
            count_success += 1
        else:
            intbl['run_status'][idx] = 'FAILED'
            intbl['error_msg'][idx] = res['error_msg']
    
    print(f"Processed {len(results)} images. Success: {count_success}, Failed: {len(results)-count_success}")

    # Save Output
    output_path = PATH_SAVE / "after_run_images.csv"
    intbl.write(str(output_path), format='csv', overwrite=True)
    print(f"Saved results to {output_path}")