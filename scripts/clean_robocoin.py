# 将robocoin中的track和task嵌入内容全部删除并更新状态和info
import os
import json
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

@click.command()
@click.option("--root", type=str, default="/data1/jibaixu/Datasets/Cobot_Magic_all_extracted/", help="Root directory containing Cobot_Magic_* datasets")
def main(root):
    root_path = Path(root)
    
    # Discover datasets
    datasets = [d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("Cobot_Magic_")]
    print(f"Found {len(datasets)} datasets to clean.")

    for dataset_dir in datasets:
        print(f"\nCleaning dataset: {dataset_dir.name}")
        meta_dir = dataset_dir / "meta"
        info_file = meta_dir / "info.json"
        stats_file = meta_dir / "episodes_stats.jsonl"
        
        # 1. Clean Parquet files
        parquet_files = list((dataset_dir / "data_clipped").rglob("*.parquet"))
        if parquet_files:
            for parquet_path in tqdm(parquet_files, desc="Cleaning Parquet files"):
                df = pd.read_parquet(parquet_path)
                
                # Identify columns added by the preprocess script
                cols_to_drop = [
                    col for col in df.columns 
                    if col == "task_emb_bert" 
                    or col.startswith("observation.tracks.") 
                    or col.startswith("observation.vis.")
                ]
                
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                    df.to_parquet(parquet_path, engine="pyarrow")

        # 2. Clean info.json
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
            
            if "features" in info_data:
                keys_to_remove = [
                    key for key in info_data["features"].keys()
                    if key == "task_emb_bert" 
                    or key.startswith("observation.tracks.") 
                    or key.startswith("observation.vis.")
                ]
                
                modified = False
                for key in keys_to_remove:
                    del info_data["features"][key]
                    modified = True
                
                if modified:
                    with open(info_file, 'w', encoding='utf-8') as f:
                        json.dump(info_data, f, indent=4)
                    print(f"Cleaned info.json (removed {len(keys_to_remove)} keys).")

        # 3. Clean episodes_stats.jsonl
        if stats_file.exists():
            cleaned_stats = []
            modified_stats = False
            
            with open(stats_file, 'r', encoding='utf-8') as f:
                for line in f:
                    ep_data = json.loads(line.strip())
                    
                    if "stats" in ep_data:
                        keys_to_remove = [
                            key for key in ep_data["stats"].keys()
                            if key.startswith("observation.tracks.") 
                            or key.startswith("observation.vis.")
                        ]
                        
                        for key in keys_to_remove:
                            del ep_data["stats"][key]
                            modified_stats = True
                            
                    cleaned_stats.append(ep_data)
            
            if modified_stats:
                with open(stats_file, 'w', encoding='utf-8') as f:
                    for ep_data in cleaned_stats:
                        f.write(json.dumps(ep_data) + "\n")
                print("Cleaned episodes_stats.jsonl.")

    print("\nDataset cleanup completed.")

if __name__ == "__main__":
    main()
