import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class OutputManager:
    """
    Manages standardized output paths for RB classification experiments.
    Follows the Output Contract (output/{pipeline}/{run_name}/).
    """
    
    VALID_PIPELINES = ['meta', 'snapshot']
    
    def __init__(self, pipeline: str, run_name: str, output_root: Optional[Union[str, Path]] = None):
        if pipeline not in self.VALID_PIPELINES:
            raise ValueError(f"Invalid pipeline '{pipeline}'. Must be one of {self.VALID_PIPELINES}")
        
        if not run_name or not isinstance(run_name, str):
            raise ValueError("run_name must be a non-empty string")
            
        self.pipeline = pipeline
        self.run_name = run_name
        
        # Default root is PROJECT_ROOT/output
        if output_root is None:
            self.output_root = Path(__file__).parent.parent.parent / "output"
        else:
            self.output_root = Path(output_root)
            
        self.run_dir = self.output_root / self.pipeline / self.run_name
        self._init_dirs()
        
    def _init_dirs(self):
        """Create the basic run directory."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
    def get_path(self, artifact_type: str, filename: Optional[str] = None) -> Path:
        """
        Get path for a specific artifact type.
        
        Args:
            artifact_type: 'plots', 'checkpoints', 'logs', 'data', or 'root'
            filename: Optional filename to append
        """
        if artifact_type == 'root':
            base = self.run_dir
        elif artifact_type in ['plots', 'checkpoints', 'logs', 'data']:
            base = self.run_dir / artifact_type
        else:
            # Default to root or custom subfolder
            base = self.run_dir / artifact_type
            
        base.mkdir(parents=True, exist_ok=True)
        
        if filename:
            return base / filename
        return base

    def write_json(self, data: Dict, filename: str, artifact_type: str = 'root'):
        """Helper to write JSON metadata."""
        path = self.get_path(artifact_type, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Wrote {filename} to {path}")

    def write_args(self, args_dict: Dict):
        """Write experimental arguments to run root."""
        self.write_json(args_dict, "args.json")

    def write_features(self, features: List[str]):
        """Write features list to run root."""
        self.write_json(features, "features.json")

    def write_metrics(self, metrics: Dict):
        """Write metrics summary to run root."""
        self.write_json(metrics, "metrics.json")
