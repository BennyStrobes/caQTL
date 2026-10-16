#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-2:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=20G                        # Memory total in MiB (for all cores)



cell_type="${1}"
pred_beta_combined_dir="${2}"
visualization_dir="${3}"
loeuf_file="${4}"

source ~/.bash_profile
conda activate plink_env

if false; then
combination_version="all_links_hurdle"
scaling_version="robust_sd"  # one of: a_p_robust_sd, robust_sd, none
beta_combined_file=${pred_beta_combined_dir}${cell_type}"_beta_combined_"${combination_version}"_"${scaling_version}".tsv.gz"
n_bins="50"
output_prefix=${visualization_dir}${cell_type}"_beta_combined_"${combination_version}"_"${scaling_version}
python visualize_beta_combined.py \
    --beta_combined_file ${beta_combined_file} \
    --cell_type ${cell_type} \
    --n_bins ${n_bins} \
    --output_prefix ${output_prefix} \
    --loeuf_file ${loeuf_file}
fi

combination_version="all_links_hurdle"
scaling_version="a_p_robust_sd"  # one of: a_p_robust_sd, robust_sd, none
beta_combined_file=${pred_beta_combined_dir}${cell_type}"_beta_combined_"${combination_version}"_"${scaling_version}".tsv.gz"
n_bins="20"
output_prefix=${visualization_dir}${cell_type}"_beta_combined_"${combination_version}"_"${scaling_version}
if false; then
python visualize_beta_combined.py \
    --beta_combined_file ${beta_combined_file} \
    --cell_type ${cell_type} \
    --n_bins ${n_bins} \
    --output_prefix ${output_prefix} \
    --loeuf_file ${loeuf_file}
fi
# Two-panel grant figure (3.8 x 2.3 in) from the summary TSVs written above; panel b calls genes by BH FDR < 0.05
python grant_figure_scatter_loeuf.py \
    --scatter_summary ${output_prefix}"_"${n_bins}"bin_scatter_symlog_centered_summary.tsv" \
    --loeuf_summary ${output_prefix}"_qtl_fraction_by_loeuf_decile_summary.tsv" \
    --per_gene ${output_prefix}"_per_gene.tsv" \
    --output_file ${output_prefix}"_grant_scatter_loeuf.pdf" \
    --call bh \
    --fdr 0.05 \
    --width 3.4 \
    --height 1.7


if false; then
combination_version="all_links_hurdle"
scaling_version="exact_a_p_robust_sd"  # one of: a_p_robust_sd, robust_sd, none
beta_combined_file=${pred_beta_combined_dir}${cell_type}"_beta_combined_"${combination_version}"_"${scaling_version}".tsv.gz"
n_bins="50"
output_prefix=${visualization_dir}${cell_type}"_beta_combined_"${combination_version}"_"${scaling_version}
python visualize_beta_combined.py \
    --beta_combined_file ${beta_combined_file} \
    --cell_type ${cell_type} \
    --n_bins ${n_bins} \
    --output_prefix ${output_prefix} \
    --loeuf_file ${loeuf_file}
fi

if false; then
combination_version="all_links_hurdle"
scaling_version="none"  # one of: a_p_robust_sd, robust_sd, none
beta_combined_file=${pred_beta_combined_dir}${cell_type}"_beta_combined_"${combination_version}"_"${scaling_version}".tsv.gz"
n_bins="50"
output_prefix=${visualization_dir}${cell_type}"_beta_combined_"${combination_version}"_"${scaling_version}
python visualize_beta_combined.py \
    --beta_combined_file ${beta_combined_file} \
    --cell_type ${cell_type} \
    --n_bins ${n_bins} \
    --output_prefix ${output_prefix} \
    --loeuf_file ${loeuf_file}
fi