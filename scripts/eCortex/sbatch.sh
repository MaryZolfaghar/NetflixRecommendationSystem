#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=2G

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate /home/mazlfghr/.conda/envs/MLprj

# sbatch scripts/eCortex/mc_U.sh &
# sbatch scripts/eCortex/mc_U_test_prc25.sh &
sbatch scripts/eCortex/mc_M.sh &
# sbatch scripts/eCortex/mc_M_test_prc25.sh &

### sbatch scripts/eCortex/sc_U.sh &
### sbatch scripts/eCortex/sc_U_test_prc25.sh &
# sbatch scripts/eCortex/sc_M.sh &
# sbatch scripts/eCortex/sc_M_test_prc25.sh &

wait
