#!/bin/bash
#SBATCH -t 0-2:00                          # Runtime in D-HH:MM format
#SBATCH -p bch-compute                        # Partition to run in
#SBATCH --mem=8GB 


variant_peak_pair_file="${1}"
peak_gtf_file="${2}"
borzoi_target_file="${3}"
borzoi_peak_targets_file="${4}"

source ~/.bashrc
conda activate borzoi

# One exon line per peak, built from the unique peaks in the pair file
python make_chromatin_peak_gtf_file.py $variant_peak_pair_file $peak_gtf_file

# Borzoi targets file restricted to blood DNase/ATAC tracks
python make_peak_accessibility_targets_file.py $borzoi_target_file $borzoi_peak_targets_file
