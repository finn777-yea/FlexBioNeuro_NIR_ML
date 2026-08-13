#!/bin/bash
#SBATCH --job-name=smoke_test_nir_ml
#SBATCH --time=2:00:00
#SBATCH --mail-user=zhehao.ma@tum.de
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

#SBATCH -o log_%j.out                   # File to store standard output
#SBATCH -e log_%j.err                   # File to store standard error

#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1

# Container configuration (use srun flags — #SBATCH does not expand shell vars)
CONTAINER_IMAGE="/dss/dsshome1/0B/go96rol2/enroot/nvidia+pytorch+23.10-py3.sqsh"
PROJECT_DIR="$(pwd)"
CONTAINER_WORKDIR="/workspace"

echo "Start on $(hostname) at $(date)"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${PROJECT_DIR}:${CONTAINER_WORKDIR}" \
     --container-workdir="${CONTAINER_WORKDIR}" \
     --no-container-remap-root \
     bash -c '
set -euo pipefail
echo "Inside container cwd: $(pwd)"
python -c "import torch; print(\"torch\", torch.__version__); print(\"cuda\", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no gpu\")"
python Classification/choose_hyperparameter/classification_torch_NN_GA_hyperparam_acid.py --smoke
'
