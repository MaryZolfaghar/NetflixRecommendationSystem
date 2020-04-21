#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=2G

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate /home/mazlfghr/.conda/envs/MLprj

sbatch scripts/eCortex/mc_default.sh &
sbatch scripts/eCortex/sc_default.sh

wait
