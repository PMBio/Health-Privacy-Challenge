#!/usr/bin/env python3
"""Build scoring manifests by walking the (already generated) synthetic tree.
Run AFTER generation completes — the synthetic dirs must exist.

Usage: python src/prepare/get_manifest.py --config submit_jobs/config_brca.yaml

Writes under manifests/<dataset>/:
    eval.txt          (gen exp split)            eval, MIA, pathway
    pairs.txt         (gen exp)                  eval / pathway fan-in
    de.txt            (gen exp split lfc)         DE
    pairs_lfc.txt     (gen exp lfc)               DE fan-in
    coexpr.txt        (gen exp split cutoff)      coexpression
    pairs_cutoff.txt  (gen exp cutoff)            coexpression fan-in
"""
import os
from pathlib import Path
import click
import yaml

SKIP = {"checkpoint", "checkpoints", "log", "logs"}


def keep(name: str) -> bool:
    n = name.lower()
    return not name.startswith("_") and n not in SKIP and "checkpoint" not in n


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--force", is_flag=True, default=False,
              help="Rebuild manifests even if they already exist.")
def main(config_path, force):
    config = yaml.safe_load(open(config_path))
    dataset = config["dataset_config"]["name"]
    split_num = config["dataset_config"]["num_splits"]
    home = config["dir_list"]["home"]
    data_split_dir = os.path.join(home, config["dir_list"]["data_splits"])

    lfcs = config.get("log_fold_changes", [])
    cutoffs = config.get("coexp_cutoffs", [])

    out = Path(home) / "manifests" / dataset
    manifests = {
        name: out / name for name in (
            "eval.txt", "pairs.txt",
            "de.txt", "pairs_lfc.txt",
            "coexpr.txt", "pairs_cutoff.txt",
        )
    }

    # skip only if ALL manifests already exist, unless --force
    if all(p.exists() for p in manifests.values()) and not force:
        print(f"{dataset}: all manifests exist, skipping. Use --force to rebuild.")
        return

    synth_root = Path(data_split_dir) / dataset / "synthetic"
    if not synth_root.exists():
        raise SystemExit(f"No synthetic dir: {synth_root}. Run generation first.")

    pairs = sorted({
        (gen.name, exp.name)
        for gen in synth_root.iterdir() if gen.is_dir() and keep(gen.name)
        for exp in gen.iterdir()         if exp.is_dir() and keep(exp.name)
    })
    if not pairs:
        raise SystemExit(f"No (generator, experiment) dirs found under {synth_root}")

    out.mkdir(parents=True, exist_ok=True)
    splits = range(1, split_num + 1)

    # --- eval / MIA / pathway: (gen, exp, split) and (gen, exp) ---
    with manifests["eval.txt"].open("w") as fh:
        for g, e in pairs:
            for s in splits:
                fh.write(f"{g} {e} {s}\n")
    with manifests["pairs.txt"].open("w") as fh:
        for g, e in pairs:
            fh.write(f"{g} {e}\n")

    # --- DE: (gen, exp, split, lfc) ; fan-in (gen, exp, lfc) ---
    with manifests["de.txt"].open("w") as fh:
        for g, e in pairs:
            for s in splits:
                for l in lfcs:
                    fh.write(f"{g} {e} {s} {l}\n")
    with manifests["pairs_lfc.txt"].open("w") as fh:
        for g, e in pairs:
            for l in lfcs:
                fh.write(f"{g} {e} {l}\n")

    # --- coexpression: (gen, exp, split, cutoff) ; fan-in (gen, exp, cutoff) ---
    with manifests["coexpr.txt"].open("w") as fh:
        for g, e in pairs:
            for s in splits:
                for c in cutoffs:
                    fh.write(f"{g} {e} {s} {c}\n")
    with manifests["pairs_cutoff.txt"].open("w") as fh:
        for g, e in pairs:
            for c in cutoffs:
                fh.write(f"{g} {e} {c}\n")

    n = len(pairs)
    print(f"{dataset}: {n} pairs -> {out}")
    print(f"  eval/mia/pathway: {n * split_num} tasks")
    print(f"  de:               {n * split_num * len(lfcs)} tasks ({len(lfcs)} lfc)")
    print(f"  coexpr:           {n * split_num * len(cutoffs)} tasks ({len(cutoffs)} cutoff)")


if __name__ == "__main__":
    main()