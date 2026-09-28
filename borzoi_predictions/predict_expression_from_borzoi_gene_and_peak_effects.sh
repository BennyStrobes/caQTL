#!/bin/bash
#SBATCH -t 0-12:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                        # Partition to run in
#SBATCH --mem=40GB


borzoi_gene_effects_file="${1}"
borzoi_peak_results_dir="${2}"
borzoi_peak_targets_file="${3}"
peak_gene_links_file="${4}"
peak_re_scaling_file="${5}"
gtex_expression_file="${6}"
gtex_genotype_plink_stem="${7}"
gtex_genotype_sample_mapping_file="${8}"
peak_track_set="${9}"
link_scaling="${10}"
link_significance="${11}"
output_file="${12}"

source ~/.bashrc
conda activate plink_env

mkdir -p $(dirname $output_file)

python predict_expression_from_borzoi_gene_and_peak_effects.py $borzoi_gene_effects_file $borzoi_peak_results_dir $borzoi_peak_targets_file $peak_gene_links_file $peak_re_scaling_file $gtex_expression_file $gtex_genotype_plink_stem $gtex_genotype_sample_mapping_file $peak_track_set $link_scaling $link_significance $output_file
