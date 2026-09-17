#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-20:00                         # Runtime in D-HH:MM format
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
# 0. LD-consistency screen (DENTIST-S / SLALOM outliers vs the lead variant, per trait per gene)
# Genes whose eQTL disagrees with the LD panel are dropped (with all their peaks); peaks whose caQTL disagrees are dropped from their gene.
# Both fine-mappers then run on the screened summary file.
##########################
screened_fm_input_summary_file="${fine_mapping_results_dir}${cell_type}_fine_mapping_input_summary_ld_screened.txt"
ld_screen_diagnostics_file="${fine_mapping_results_dir}${cell_type}_ld_consistency_screen.txt"
python screen_ld_consistency.py --dentist_p_threshold 1e-2  --fm_input_summary_file ${fm_input_summary_file} --screened_summary_file ${screened_fm_input_summary_file} --diagnostics_file ${ld_screen_diagnostics_file} --drop_failed_eqtl_genes


fm_input_summary_file=${screened_fm_input_summary_file}


##########################
# 1. Standard, eQTL-only SuSiE fine-mapping (z-score model; no sample size needed)
##########################
eqtl_only_finemapping_output_file="${fine_mapping_results_dir}${cell_type}_eqtl_only_finemapping_results.txt"
python run_standard_susie_finemapping_of_eqtls.py --sumstat_summary_file ${fm_input_summary_file} --fine_mapping_output_file ${eqtl_only_finemapping_output_file}


##########################
# 2. caQTL-mediated eQTL fine-mapping (z-score model; link scale lambda FIXED from data, link bias variance tau_u^2 FIXED at 0)
# Estimating lambda by EM is degenerate (lambda -> 0, tau2_g -> inf); lambda ~ 1 / slope of eQTL effects on caQTL x link products (~0.1)
##########################
# lambda on the z-score scale, estimated from the prepared inputs (slope of eQTL z-effects on caQTL z-effects x links)
link_scale_file="${fine_mapping_results_dir}${cell_type}_link_scale_estimate.txt"
link_scale=$(python estimate_link_scale.py --sumstat_summary_file ${fm_input_summary_file} --output_file ${link_scale_file} --lambda_only)
echo "link scale lambda: ${link_scale}"
mediated_finemapping_output_file="${fine_mapping_results_dir}${cell_type}_caqtl_mediated_finemapping_results.txt"
mediated_hyperparameter_output_file="${fine_mapping_results_dir}${cell_type}_caqtl_mediated_finemapping_shared_hyperparameters.txt"
if false; then
python run_ca_qtl_mediated_eqtl_finemapping.py --sumstat_summary_file ${fm_input_summary_file} --fine_mapping_output_file ${mediated_finemapping_output_file} --shared_hyperparameter_output_file ${mediated_hyperparameter_output_file} --link_scale_init ${link_scale} --no_component_output --no_estimate_link_scale
fi

# Model D (link inclusion): lambda fixed, tau_u^2 estimated, pi_u (fraction of non-zero links that mediate) estimated; the eQTL data decide which hurdle-implied effects are included
inclusion_finemapping_output_file="${fine_mapping_results_dir}${cell_type}_caqtl_mediated_link_inclusion_finemapping_results.txt"
inclusion_hyperparameter_output_file="${fine_mapping_results_dir}${cell_type}_caqtl_mediated_link_inclusion_finemapping_shared_hyperparameters.txt"
if false; then
python run_ca_qtl_mediated_eqtl_finemapping.py --sumstat_summary_file ${fm_input_summary_file} --fine_mapping_output_file ${inclusion_finemapping_output_file} --shared_hyperparameter_output_file ${inclusion_hyperparameter_output_file} --link_scale_init ${link_scale} --link_inclusion_prior_strength 0.0 --link_inclusion_prob_delay 0 --link_inclusion_prob_init 0.9 --no_estimate_link_scale --hyper_tol "5e-4" --link_inclusion --max_outer_iter 200
fi



if false; then
python run_ca_qtl_mediated_eqtl_finemapping.py --sumstat_summary_file ${fm_input_summary_file} --fine_mapping_output_file ${mediated_finemapping_output_file} --shared_hyperparameter_output_file ${mediated_hyperparameter_output_file} --link_scale_init ${link_scale} --no_estimate_link_scale --no_estimate_link_bias_variance --no_component_output

python run_ca_qtl_mediated_eqtl_finemapping.py --sumstat_summary_file ${fm_input_summary_file} --fine_mapping_output_file ${mediated_finemapping_output_file} --shared_hyperparameter_output_file ${mediated_hyperparameter_output_file} --link_scale_init ${link_scale} --no_component_output
fi


##########################
# 2b. Model E (variants and real-link peak genetic components as competing elements): lambda fixed, tau_u^2 estimated, peak weight ratio estimated
##########################
if false; then

elements_finemapping_output_file="${fine_mapping_results_dir}${cell_type}_caqtl_mediated_elements_finemapping_results2.txt"
elements_hyperparameter_output_file="${fine_mapping_results_dir}${cell_type}_caqtl_mediated_elements_finemapping_shared_hyperparameters2.txt"
python run_ca_qtl_elements_eqtl_finemapping.py --sumstat_summary_file ${fm_input_summary_file} --fine_mapping_output_file ${elements_finemapping_output_file} --shared_hyperparameter_output_file ${elements_hyperparameter_output_file} --link_scale_init ${link_scale} --estimate_link_scale --no_estimate_link_bias_variance --max_outer_iter 50

# E-free comparison (selected peaks get a free effect; no lambda)
python run_ca_qtl_elements_eqtl_finemapping.py --sumstat_summary_file ${fm_input_summary_file} --fine_mapping_output_file ${fine_mapping_results_dir}${cell_type}_caqtl_mediated_elements_free_finemapping_results.txt --shared_hyperparameter_output_file ${fine_mapping_results_dir}${cell_type}_caqtl_mediated_elements_free_finemapping_shared_hyperparameters.txt --link_scale_init ${link_scale} --free_peak_effects --max_outer_iter 80
fi

##########################
# 2c. Model J (cTWAS on peaks): fixed standardized peak predictors, free effects, group priors by hurdle significance and sign concordance
##########################
ctwas_finemapping_output_file="${fine_mapping_results_dir}${cell_type}_caqtl_mediated_ctwas_finemapping_results.txt"
ctwas_hyperparameter_output_file="${fine_mapping_results_dir}${cell_type}_caqtl_mediated_ctwas_finemapping_shared_hyperparameters.txt"
if false; then
python run_ca_qtl_ctwas_eqtl_finemapping.py --sumstat_summary_file ${fm_input_summary_file} --fine_mapping_output_file ${ctwas_finemapping_output_file} --no_sign_groups --t_bin_edges 1,2,4,6 --shared_hyperparameter_output_file ${ctwas_hyperparameter_output_file}
fi

##########################
# 2d. Model L (predicted-mean slab): causal eQTL effects are a spike plus a Gaussian centred on the chromatin-predicted effect;
#     caQTL (SuSiE) and hurdle (spike-and-slab) posteriors marginalized inside a variant-only eQTL fit; lambda per hurdle bin and
#     two slab variances learned by EM (caqtl_mediated_fine_mapping_model_proposal_predicted_mean.md)
##########################
predicted_mean_finemapping_output_file="${fine_mapping_results_dir}${cell_type}_caqtl_mediated_predicted_mean_finemapping_results.txt"
predicted_mean_hyperparameter_output_file="${fine_mapping_results_dir}${cell_type}_caqtl_mediated_predicted_mean_finemapping_shared_hyperparameters.txt"
python run_ca_qtl_predicted_mean_eqtl_finemapping.py --sumstat_summary_file ${fm_input_summary_file} --fine_mapping_output_file ${predicted_mean_finemapping_output_file} --shared_hyperparameter_output_file ${predicted_mean_hyperparameter_output_file} --t_bin_edges 2,4 --lambda_prior_sd 10



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
