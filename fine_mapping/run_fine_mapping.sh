#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-60:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=40G                        # Memory total in MiB (for all cores)


fm_input_summary_file=${1}
cell_type=${2}
fine_mapping_results_dir=${3}

source ~/.bash_profile
conda activate plink_env
mkdir -p ${fine_mapping_results_dir}

date

# Peak IDs for the plots (from the prep step's caQTL summary, next to the original fine-mapping input summary)
caqtl_sumstat_summary_file="$(dirname ${fm_input_summary_file})/${cell_type}_caqtl_mediated_sumstats_summary.txt"




##########################
# 1. Standard, eQTL-only SuSiE fine-mapping (z-score model; no sample size needed)
##########################
eqtl_only_finemapping_output_file="${fine_mapping_results_dir}${cell_type}_eqtl_only_finemapping_results.txt"
if false; then
python run_standard_susie_finemapping_of_eqtls.py --sumstat_summary_file ${fm_input_summary_file} --fine_mapping_output_file ${eqtl_only_finemapping_output_file}
fi

##########################
# 1. eQTL only expression prediction
##########################
eqtl_only_expression_prediction_output_file="${fine_mapping_results_dir}${cell_type}_eqtl_only_expression_prediction_results.txt"
python run_eqtl_only_expression_prediction.py --sumstat_summary_file ${fm_input_summary_file} --expression_prediction_output_file ${eqtl_only_expression_prediction_output_file}


##########################
# 2. Joint eQTL/caQTL expression prediction model (spike-and-slab eQTL effects with prior mean = lambda x caQTL effect x peak-gene effect; fit once across all genes)
# This is an expression prediction model, not fine-mapping (inclusion probabilities are not valid PIPs under tight LD)
##########################
joint_expression_prediction_output_file="${fine_mapping_results_dir}${cell_type}_joint_eqtl_caqtl_expression_prediction_results.txt"
python run_joint_eqtl_caqtl_expression_prediction.py --sumstat_summary_file ${fm_input_summary_file} --expression_prediction_output_file ${joint_expression_prediction_output_file}

# Add mediation term
joint_expression_prediction_output_file="${fine_mapping_results_dir}${cell_type}_joint_eqtl_caqtl_mediated_expression_prediction_results.txt"
python run_joint_eqtl_caqtl_expression_prediction.py --mediation_indicator --sumstat_summary_file ${fm_input_summary_file} --expression_prediction_output_file ${joint_expression_prediction_output_file}



##########################
# 3. Plots comparing eQTL-only and caQTL-mediated (Model E, elements) fine-mapping results
##########################
if false; then
comparison_plot_stem="${fine_mapping_results_dir}${cell_type}_eqtl_only_vs_caqtl_mediated_elements"
python visualize_fine_mapping_comparison.py --eqtl_only_results_file ${eqtl_only_finemapping_output_file} --mediated_results_file ${elements_finemapping_output_file} --output_stem ${comparison_plot_stem}

# Stacked eQTL / caQTL Manhattan plots for every variant with a caQTL-mediated PIP > 0.5 (genes with linked peaks), in their own folder
joint_pip_locus_plot_dir="${fine_mapping_results_dir}${cell_type}_caqtl_mediated_elements_pip_gt_0.5_locus_plots/"
mkdir -p ${joint_pip_locus_plot_dir}
python plot_fine_mapping_example_loci.py --eqtl_only_results_file ${eqtl_only_finemapping_output_file} --mediated_results_file ${elements_finemapping_output_file} --fm_input_summary_file ${fm_input_summary_file} --caqtl_sumstat_summary_file ${caqtl_sumstat_summary_file} --output_stem ${joint_pip_locus_plot_dir}${cell_type} --all_qualifying --min_eqtl_only_pip 0.0 --max_eqtl_only_pip 1.0 --min_mediated_pip 0.5
fi

##########################
# 4. Per-link support table (Model E): implied eQTL z at each peak's caQTL lead vs the observed eQTL z, with the link probability and P_k
##########################
if false; then
python tabulate_link_support.py --mediated_results_file ${elements_finemapping_output_file} --shared_hyperparameter_file ${elements_hyperparameter_output_file} --fm_input_summary_file ${fm_input_summary_file} --output_file ${fine_mapping_results_dir}${cell_type}_link_support.txt
fi

date
