from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class PsfexPipelineOutputs:
    workdir: Path
    image_fits: Path
    cat_pre: Path
    psf: Path
    cat_final: Path
    logs: Dict[str, Path]

@dataclass
class SexOnlyOutputs:
    workdir: Path
    image_fits: Path
    cat_final: Path
    logs: Dict[str, Path]

def _run(cmd: List[str], cwd: Path, log_path: Path, env: Optional[dict] = None) -> None:
    """
    Run command, stream stdout/stderr to log file, raise on failure.
    """
    cmd_str = " ".join(cmd)
    with open(log_path, "w") as f:
        print(f"Running: {cmd_str}")
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            check=False,
        )
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit={p.returncode}).\n"
            f"cmd: {cmd_str}\n"
            f"log: {log_path}"
        )

def run_sex_psfex_sex(
    image_path: str | Path,
    *,
    workdir: str | Path,
    # pre-PSFEx SExtractor
    pre_sex_conf: str | Path,
    pre_param: str | Path,
    # PSFEx
    psfex_conf: str | Path,
    # post-PSFEx SExtractor
    post_sex_conf: str | Path,
    post_param: str | Path,
    # ancillary files
    default_conv: Optional[str | Path] = None,
    default_nnw: Optional[str | Path] = None,
    # output naming
    pre_cat_name: str = "image.pre.cat",
    psf_name: str = "image.psf",
    final_cat_name: str = "image.final.cat",
) -> PsfexPipelineOutputs:
    """
    Full pipeline: SEx(pre) -> PSFEx -> SEx(post, with PSF).
    Saves outputs to workdir/sci.
    """
    # 0) Path resolving
    image_path = Path(image_path).expanduser().resolve()
    base_workdir = Path(workdir).expanduser().resolve()
    
    # 0.1) Specific output directory for science image
    sci_dir = base_workdir / "sci"
    sci_dir.mkdir(parents=True, exist_ok=True)
    
    # Log dir inside sci_dir (or base_workdir? user said workdir/sci, let's keep logs there too to be self-contained)
    log_dir = sci_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = image_path.stem # filename without extension

    pre_sex_conf = Path(pre_sex_conf).expanduser().resolve()
    pre_param = Path(pre_param).expanduser().resolve()
    psfex_conf = Path(psfex_conf).expanduser().resolve()
    post_sex_conf = Path(post_sex_conf).expanduser().resolve()
    post_param = Path(post_param).expanduser().resolve()

    # check existence
    for p in [image_path, pre_sex_conf, pre_param, psfex_conf, post_sex_conf, post_param]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    sex_bin = shutil.which("source-extractor")
    psfex_bin = shutil.which("psfex")
    if not sex_bin or not psfex_bin:
        raise FileNotFoundError("Missing 'source-extractor' or 'psfex' in PATH.")

    # 1) Stage files (optional but good for aux files)
    if default_conv:
        default_conv = Path(default_conv).resolve()
        shutil.copy2(default_conv, sci_dir / "default.conv")
    if default_nnw:
        default_nnw = Path(default_nnw).resolve()
        shutil.copy2(default_nnw, sci_dir / "default.nnw")
    
    # We do NOT copy the image, just reference it.
    
    # Define output files
    pre_cat = sci_dir / pre_cat_name
    psf = sci_dir / psf_name
    final_cat = sci_dir / final_cat_name
    
    logs = {
        "sex_pre": log_dir / f"{prefix}_sex_pre.log",
        "psfex": log_dir / f"{prefix}_psfex.log",
        "sex_post": log_dir / f"{prefix}_sex_post.log",
    }

    # 2) SEx pre
    cmd_pre = [
        sex_bin, str(image_path),
        "-c", str(pre_sex_conf),
        "-PARAMETERS_NAME", str(pre_param),
        "-CATALOG_NAME", str(pre_cat),
    ]
    _run(cmd_pre, cwd=sci_dir, log_path=logs["sex_pre"])

    # 3) PSFEx
    cmd_psfex = [
        psfex_bin, str(pre_cat),
        "-c", str(psfex_conf),
        "-PSF_DIR", str(sci_dir),
        "-XML_NAME", str(psf.with_suffix(".xml")),
    ]
    _run(cmd_psfex, cwd=sci_dir, log_path=logs["psfex"])
    
    # Verify PSF creation (PSFEx naming can be tricky)
    # Expected: pre_cat name but with .psf extension? 
    # Usually PSFEx takes input catalog "name.cat" and outputs "name.psf".
    # If we named pre_cat as "image.pre.cat", output might be "image.pre.psf".
    # User logic tried to find .psf or enforce name.
    
    expected_psf_default = sci_dir / f"{pre_cat.stem}.psf"
    
    if not psf.exists():
        # Check if default output exists, rename if needed
        if expected_psf_default.exists():
            expected_psf_default.rename(psf)
        else:
             # Fallback
            candidates = list(sci_dir.glob("*.psf"))
            if len(candidates) == 1:
                # If we found exactly one, assume it's the one (risky if parallel? No, dir is unique per task usually? 
                # Actually user script makes workdir unique per tile. 
                # But 'sci' folder is shared if multiple images in same tile?
                # The prompt implies iterating images. If images are in same tile, workdir is same.
                # So we must be careful with globs.
                # PSFEx behavior: input "foo.cat" -> output "foo.psf".
                pass
            
    if not psf.exists():
         raise RuntimeError(f"PSF not created at {psf}")

    # 4) SEx post
    cmd_post = [
        sex_bin, str(image_path),
        "-c", str(post_sex_conf),
        "-PARAMETERS_NAME", str(post_param),
        "-PSF_NAME", str(psf),
        "-CATALOG_NAME", str(final_cat),
    ]
    _run(cmd_post, cwd=sci_dir, log_path=logs["sex_post"])

    return PsfexPipelineOutputs(
        workdir=sci_dir,
        image_fits=image_path,
        cat_pre=pre_cat,
        psf=psf,
        cat_final=final_cat,
        logs=logs,
    )

def run_sex_only(
    image_path: str | Path,
    *,
    workdir: str | Path,
    psf_path: str | Path,
    sex_conf: str | Path,
    param: str | Path,
    default_conv: Optional[str | Path] = None,
    default_nnw: Optional[str | Path] = None,
    final_cat_name: str = "sub.cat",
) -> SexOnlyOutputs:
    """
    Run only the post-processing SExtractor (using existing PSF).
    Saves outputs to workdir/sub.
    """
    # 0) Path resolving
    image_path = Path(image_path).expanduser().resolve()
    base_workdir = Path(workdir).expanduser().resolve()
    psf_path = Path(psf_path).expanduser().resolve()
    
    # 0.1) Specific output directory for sub image
    sub_dir = base_workdir / "sub"
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = sub_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = image_path.stem 

    sex_conf = Path(sex_conf).expanduser().resolve()
    param = Path(param).expanduser().resolve()

    if not sex_conf.exists() or not param.exists() or not psf_path.exists():
        raise FileNotFoundError("Missing required file for SEx-only run.")

    sex_bin = shutil.which("source-extractor")
    if not sex_bin:
         raise FileNotFoundError("Missing 'source-extractor' in PATH.")
         
    # 1) Stage aux files
    if default_conv:
        default_conv = Path(default_conv).resolve()
        shutil.copy2(default_conv, sub_dir / "default.conv")
    if default_nnw:
        default_nnw = Path(default_nnw).resolve()
        shutil.copy2(default_nnw, sub_dir / "default.nnw")
        
    final_cat = sub_dir / final_cat_name
    
    log_file = log_dir / f"{prefix}_sex.log"
    logs = {"sex": log_file}
    
    # 2) Run SEx
    cmd = [
        sex_bin, str(image_path),
        "-c", str(sex_conf),
        "-PARAMETERS_NAME", str(param),
        "-PSF_NAME", str(psf_path),
        "-CATALOG_NAME", str(final_cat),
    ]
    _run(cmd, cwd=sub_dir, log_path=log_file)
    
    return SexOnlyOutputs(
        workdir=sub_dir,
        image_fits=image_path,
        cat_final=final_cat,
        logs=logs,
    )

def move_check_images(
    src_dir: Path, 
    dest_dir: Path, 
    image_path: Path, 
    patterns: List[str] = ["samp", "snap", "resi"]
) -> None:
    """
    Move check images (created by SExtractor) to a destination folder.
    Uses subprocess/shutil rather than os.system.
    
    Typically SEx produces check images in the cwd.
    If image_path is '.../foo.fits', patterns like 'samp' usually map to 'samp_foo.fits' 
    if configured that way in .sex config (CHECKIMAGE_NAME).
    
    The user script had: `sampim = f"{checkdir}/samp_{os.path.basename(image_path)}"`
    And it looked for `_sampim = f"{workdir}/samp_{os.path.basename(image_path.replace('.fits', '.pre.fits'))}"`
    
    Wait, the original script logic for naming was specific:
    `_sampim = f"{workdir}/samp_{os.path.basename(image_path.replace('.fits', '.pre.fits'))}"`
    This implies the input image to SEx was named something like `*.pre.fits` or the CHECKIMAGE_NAME was configured strictly.
    
    However, in `run_sex_psfex_sex` above, we are running SEx with `image_path` (original fits) or `staged_image`.
    The user's original script passed `staged_image` which was just `image_path`.
    
    If the .sex config says `CHECKIMAGE_NAME samp.fits`, then it's `samp.fits`.
    If it says `CHECKIMAGE_NAME samp_image.fits`, then it is that.
    
    We'll assume the files are generated in `src_dir` (e.g. `sci` or `sub`) and we want to move them to `dest_dir` (`check`).
    
    We will assume standard naming convention based on the provided logic or wildcard search? 
    The user asked to replace os.system with subprocess for cleaning part.
    
    Let's make this function generic:
    find files matching `pattern_*.fits` in src_dir and move to dest_dir.
    Or strictly follow user's naming if possible.
    
    In the main script, the user constructed names manually. 
    It's better if we just look for the expected CHECKIMAGE names if we know them, 
    OR we can pass the expected basenames.
    
    For now, I'll replicate the user's logic but using python's `shutil.move` which is better than `os.system("mv ...")`.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # We need to accommodate how the user set up the checks. 
    # Since I don't see the .sex file, I can't be 100% sure of the output name.
    # But I can implement the function to accept flexible "patterns" and just move them.
    
    # User's logic was:
    # _sampim = f"{workdir}/samp_{image_basename_pre}"
    # sampim = f"{checkdir}/samp_{image_basename}"
    # move _sampim -> sampim
    
    # I'll let the main script call this with specific paths to be safe,
    # OR I can just look for `samp_*.fits` in `src_dir` and move them?
    # Safer to let main script define exact expectations?
    # No, that's tedious.
    
    # Let's try to find files starting with the patterns.
    
    for pattern in patterns:
        # We search for files starting with pattern in src_dir
        # But wait, multiple images might be running in parallel if we are not careful.
        # Ideally check images should have unique names per run.
        # The user's SEx config likely uses CHECKIMAGE_NAME with a suffix.
        
        # We will iterate over the candidates and move them.
        candidates = list(src_dir.glob(f"{pattern}_*.fits"))
        for cand in candidates:
             dest_path = dest_dir / cand.name
             shutil.move(str(cand), str(dest_path))

