#!/bin/bash
# Run chunk_plots_gen.py as a SLURM batch job instead of on the login node.
#
# Usage (from /blue/avery/justin.heyman/bbzz):
#   sbatch plot_gen.sh -i output/2l2nu/*.root -o plots/2l2nu_gen
# All arguments are passed straight to chunk_plots_gen.py.
# Progress: tail -f logs/plots_gen_<jobid>.log
#
#SBATCH --job-name=plots_gen
#SBATCH --account=avery
#SBATCH --qos=avery
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/plots_gen_%j.log

cd /blue/avery/justin.heyman/bbzz
source setuproot.sh

python3 -u chunk_plots_gen.py "$@"
