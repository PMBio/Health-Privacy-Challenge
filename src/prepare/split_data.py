#!/usr/bin/env python3
"""Generate real-data folds + MIA membership labels from split indices.

Reads:  original counts (TSV, genes×samples) + subtype labels + <dataset>_splits.yaml
Writes: data_splits/<dataset>/real/{X,y}_{train,test}_real_split_*.csv
        data_splits/<dataset>/real/MIA_lbl_split_*.csv
        data_splits/<dataset>/real/column_names.csv
"""

import os
import yaml
import click
import numpy as np
import pandas as pd
from typing import Any, Dict, List  
from pathlib import Path


class RealDataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.home_dir = Path(config["dir_list"]["home"])
        #self.home_dir = Path(config["dir_list"]["home"]).expanduser()
        print(f"Home directory: {self.home_dir}")
        dc = config["dataset_config"]
        self.dataset_name = dc["name"]
        self.num_splits = dc["num_splits"]
        self.random_seed = dc["random_seed"]
        self.sample_col_name = dc["sample_col_name"]
        self.subtype_col_name = dc["subtype_col_name"]

        self.original_data_path = self.home_dir / dc["count_file"]
        self.original_lbl_path = self.home_dir / dc["annot_file"]
        self.data_save_dir = self.home_dir / config["dir_list"]["data_save_dir"]
        self.split_indices_dir = self.home_dir / config["dir_list"]["split_save_dir"]

        print(f"Random seed set to {self.random_seed}.")
        self.read_original_data()
        self.read_subtype_labels()

    def read_original_data(self):
        if not self.original_data_path.exists():
            raise FileNotFoundError(f"Original data is missing: {self.original_data_path}")
        # TSV is genes×samples; transpose so samples are rows (index = sample IDs)
        self.original_data = pd.read_csv(self.original_data_path, sep="\t", index_col=0).T

    def read_subtype_labels(self):
        if not self.original_lbl_path.exists():
            raise FileNotFoundError(f"Labels file is missing: {self.original_lbl_path}")
        self.subtype_labels = pd.read_csv(self.original_lbl_path)
        
        # COMBINED: the fine-grained label lives in `project`, but downstream expects
        # it under `cancer_type` (which in this annotation file holds coarse tissue
        # names). Overwrite cancer_type with the project-derived label, keeping the
        # column name so downstream (subtype_col_name = cancer_type) is unchanged.
        source_col = self.config["dataset_config"].get("label_source_col")
        if source_col and source_col != self.subtype_col_name:
            if source_col not in self.subtype_labels.columns:
                raise ValueError(
                    f"label_source_col '{source_col}' not in labels file; "
                    f"have {list(self.subtype_labels.columns)}"
                )
            self.subtype_labels[self.subtype_col_name] = self.subtype_labels[source_col]

        indexed = self.subtype_labels.set_index(self.sample_col_name)
        # guard: every data sample must have exactly one label
        missing = self.original_data.index.difference(indexed.index)
        if len(missing):
            raise ValueError(f"{len(missing)} samples have no label, e.g. {list(missing[:5])}")
        if indexed.index.duplicated().any():
            raise ValueError("Duplicate sample IDs in labels file")

        ordered_subtype_labels = indexed.loc[self.original_data.index]
        self.subtype_labels = ordered_subtype_labels.reset_index()
        self.subtype_labels = self.subtype_labels.rename(columns={"index": self.sample_col_name})
        self.subtype_col_ix = self.subtype_labels.columns.get_loc(self.subtype_col_name)

    

    # -----------------------------------------------------------------
    def save_membership_dataset(self, i: int, train_index, test_index, split_save_dir: str):
        train_pos = self.original_data.index.get_indexer(train_index)
        test_pos = self.original_data.index.get_indexer(test_index)
        # get_indexer returns -1 for labels not found -> would corrupt the last row
        if (train_pos < 0).any() or (test_pos < 0).any():
            raise ValueError(f"Split {i+1}: some indices not found in original_data")

        mmb_labels = np.zeros(len(self.original_data))
        mmb_labels[train_pos] = 1
        mmb_labels[test_pos] = 0

        mmb_labels_df = pd.DataFrame(
            {"membership_label": mmb_labels}, index=self.original_data.index
        )
        mmb_labels_df.to_csv(os.path.join(split_save_dir, f"MIA_lbl_split_{i+1}.csv"), index=True)

        # MIA_ds (full matrix + label) is unused downstream — kept commented for reference
        # mmb_df = self.original_data.copy()
        # mmb_df["membership_label"] = mmb_labels
        # mmb_df.to_csv(os.path.join(split_save_dir, f"MIA_ds_split_{i+1}.csv"), index=True)

    # -----------------------------------------------------------------
    def save_split_data(self):
        splits_file = self.split_indices_dir / f"{self.dataset_name}_splits.yaml"
        if not splits_file.exists():
            raise FileNotFoundError(f"Split indices not found: {splits_file}. Generate them first.")
        with open(splits_file) as fh:
            loaded = yaml.safe_load(fh)
        splits = loaded.get("splits", {})
        if not splits:
            raise ValueError(f"No 'splits' found in {splits_file}")

        split_indices_named = [
            (np.array(s["train_index"]), np.array(s["test_index"]))
            for s in splits.values()
        ]

        split_save_dir = self.data_save_dir / self.dataset_name / "real"
        split_save_dir.mkdir(parents=True, exist_ok=True)

        subtypes_indexed = self.subtype_labels.set_index(self.sample_col_name)

        for i, (train_index, test_index) in enumerate(split_indices_named):
            # index (sample IDs) preserved on every output — used downstream
            X_train = self.original_data.loc[train_index]
            X_test = self.original_data.loc[test_index]
            y_train = subtypes_indexed.loc[train_index, [self.subtype_col_name]]
            y_test = subtypes_indexed.loc[test_index, [self.subtype_col_name]]

            X_train.to_csv(split_save_dir / f"X_train_real_split_{i+1}.csv", index=False)
            X_test.to_csv(split_save_dir / f"X_test_real_split_{i+1}.csv", index=False)
            y_train.to_csv(split_save_dir / f"y_train_real_split_{i+1}.csv", index=False)
            y_test.to_csv(split_save_dir / f"y_test_real_split_{i+1}.csv", index=False)

            self.save_membership_dataset(i, train_index, test_index, split_save_dir)

        # column names are split-independent — write once, outside the loop
        pd.DataFrame(self.original_data.columns.values, columns=["column_names"]).to_csv(
            split_save_dir / "column_names.csv", index=False
        )
        print(f"Wrote {len(split_indices_named)} splits to {split_save_dir}")




@click.command()
@click.option("--config", "config_path", default="config.yaml",
              type=click.Path(exists=True), help="Path to config YAML.")
def main(config_path):
    """Generate real folds + MIA membership labels from split indices."""
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    RealDataLoader(config).save_split_data()


if __name__ == "__main__":
    main()