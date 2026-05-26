import os
from pathlib import Path

def migrate_snapshots():
    project_root = Path('/lyman/data1/rb_classification_meta')
    output_root = project_root / 'output'
    snapshot_root = output_root / 'snapshot'
    
    # Artifact types in old structure
    artifact_types = {
        'checkpoints': 'checkpoints',
        'logs': 'logs',
        'plots': 'plots',
        'results': 'data' # New name is 'data'
    }
    
    # 1. Collect all run names from all old artifact folders
    # Note: They are currently at output/snapshot/{artifact_type}/{run_name}
    run_names = set()
    for old_type in artifact_types.keys():
        old_dir = snapshot_root / old_type
        if old_dir.exists():
            for run_dir in old_dir.iterdir():
                if run_dir.is_dir():
                    run_names.add(run_dir.name)
    
    print(f"Found {len(run_names)} runs to migrate.")
    
    # 2. Create new consolidated structure via symlinks
    for run_name in sorted(run_names):
        new_run_dir = snapshot_root / run_name
        new_run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Migrating run: {run_name}")
        
        for old_type, new_type in artifact_types.items():
            old_path = snapshot_root / old_type / run_name
            new_path = new_run_dir / new_type
            
            if old_path.exists():
                if new_path.exists() or new_path.is_symlink():
                    # print(f"  Skipping {new_type}: Already exists")
                    continue
                
                try:
                    # Use relative symlink if possible for portability, 
                    # but absolute is safer given the project structure.
                    os.symlink(old_path, new_path)
                    print(f"  ✓ Linked {old_type} -> {new_type}")
                except Exception as e:
                    print(f"  ✗ Failed to link {old_type}: {e}")
            else:
                pass # print(f"  - No {old_type} found")

    print("\nMigration complete! Existing outputs are now accessible via output/snapshot/<run_name>/")

if __name__ == "__main__":
    migrate_snapshots()
