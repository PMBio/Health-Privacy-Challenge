# lib.sh — shared helpers for submit scripts. Source after setting CONFIG.
#   CONFIG must be set by the caller before calling resolve_paths.

cfg() {
    python -c "import yaml,sys; d=yaml.safe_load(open('$CONFIG')); print(eval('d'+''.join(f'[\"{k}\"]' for k in sys.argv[1:])))" "$@"
}

resolve_paths() {
    HOME_DIR=$(cfg dir_list home)
    SRC_DIR="${HOME_DIR}/src"
    DATASET=$(cfg dataset_config name)
    BIO_ROOT="${HOME_DIR}/$(cfg dir_list bio_files)/${DATASET}"
    RES_ROOT="${HOME_DIR}/$(cfg dir_list res_files)/${DATASET}"
    MANIFEST_DIR="${HOME_DIR}/manifests/${DATASET}"
    JOBS_DIR="${HOME_DIR}/workflows/jobs"
    DATA_SPLIT_DIR="${HOME_DIR}/$(cfg dir_list data_splits)"
    P_VALUE=$(cfg evaluator_config bio_params p_value)
}
