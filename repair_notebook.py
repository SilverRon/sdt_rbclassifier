import json
from pathlib import Path

def repair_notebook():
    notebook_path = Path("notebook/99_Summary_Compare_Results.ipynb")
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # 1. Column-aware load_run_data replacement
    load_run_data_code = [
        "def load_run_data(run_info):\n",
        "    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score\n",
        "    import pandas as pd\n",
        "    \n",
        "    run_path = Path(run_info['path'])\n",
        "    pipeline = run_info['pipeline']\n",
        "    run_id = run_info['run_id']\n",
        "    \n",
        "    res = {'run_id': run_id, 'pipeline': pipeline, 'model_family': pipeline, 'layout': run_info.get('layout', 'unknown')}\n",
        "    \n",
        "    # Initialize common keys to avoid KeyError later\n",
        "    for k in ['test_prec', 'test_rec', 'f1_real', 'f1_bogus', 'optimal_threshold', 'test_auroc', 'test_f1', 'test_acc']: \n",
        "        res[k] = None\n",
        "    \n",
        "    # Try artifact locations (root first, then data/ subdir for migrated snapshots)\n",
        "    artifact_paths = [run_path, run_path / 'data']\n",
        "    \n",
        "    for base_p in artifact_paths:\n",
        "        if not base_p.exists(): continue\n",
        "        for fname in ['args.json', 'features.json', 'metrics.json']:\n",
        "            p = base_p / fname\n",
        "            if p.exists():\n",
        "                try:\n",
        "                    with open(p, 'r') as f: data = json.load(f)\n",
        "                    if fname == 'metrics.json' and 'metrics' in data: res.update(data['metrics'])\n",
        "                    else: res.update(data)\n",
        "                except: pass\n",
        "    \n",
        "    # Fallback: Compute metrics from CSV if metrics.json missing\n",
        "    if res['test_auroc'] is None:\n",
        "        for base_p in artifact_paths:\n",
        "            if not base_p.exists(): continue\n",
        "            for csv_name in ['test_predictions.csv', 'validation_predictions.csv']:\n",
        "                csv_path = base_p / csv_name\n",
        "                if csv_path.exists():\n",
        "                    try:\n",
        "                        df_pred = pd.read_csv(csv_path)\n",
        "                        # Column mapping support\n",
        "                        for old, new in [('true_label', 'y_true'), ('score', 'y_score'), ('predicted_label', 'y_pred')]:\n",
        "                            if old in df_pred.columns: df_pred.rename(columns={old: new}, inplace=True)\n",
        "                        \n",
        "                        if 'y_true' in df_pred.columns and 'y_score' in df_pred.columns:\n",
        "                            res['test_auroc'] = roc_auc_score(df_pred['y_true'], df_pred['y_score'])\n",
        "                            if 'y_pred' in df_pred.columns:\n",
        "                                res['test_acc'] = accuracy_score(df_pred['y_true'], df_pred['y_pred'])\n",
        "                                res['test_f1'] = f1_score(df_pred['y_true'], df_pred['y_pred'], average='macro')\n",
        "                            if csv_name == 'test_predictions.csv': break\n",
        "                    except: pass\n",
        "    \n",
        "    # Infer model type if unknown\n",
        "    if res.get('model_type', 'unknown') == 'unknown':\n",
        "        if 'model' in res: res['model_type'] = res['model']\n",
        "        elif 'v3_' in run_id: res['model_type'] = run_id.split('v3_')[1]\n",
        "        elif 'v2_' in run_id: res['model_type'] = run_id.split('v2_')[1].split('_')[0]\n",
        "    \n",
        "    # Standardize\n",
        "    if 'test_auroc' in res and 'auroc' not in res: res['auroc'] = res['test_auroc']\n",
        "    if 'test_f1' in res and 'f1_macro' not in res: res['f1_macro'] = res['test_f1']\n",
        "    if 'test_acc' in res and 'accuracy' not in res: res['accuracy'] = res['test_acc']\n",
        "    \n",
        "    res['has_plots'] = (run_path / 'plots').exists() or (run_path / 'data' / 'plots').exists() or (run_path.parent / 'plots' / run_id).exists()\n",
        "    res['has_checkpoints'] = (run_path / 'checkpoints').exists() or (run_path / 'data' / 'checkpoints').exists() or (run_path.parent / 'checkpoints' / run_id).exists()\n",
        "    return res\n"
    ]

    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and any('def load_run_data' in line for line in cell['source']):
            cell['source'] = load_run_data_code
            break

    # Normalize
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []

    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print("\u2713 Notebook robust repair (v5) complete.")

if __name__ == "__main__":
    repair_notebook()
