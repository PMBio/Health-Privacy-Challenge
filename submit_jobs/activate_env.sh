set +u
MAMBA_EXE="~/software/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
micromamba activate health-privacy-env
set -u