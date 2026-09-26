#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-10:00                         # Runtime in D-HH:MM format
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
if false; then
python run_eqtl_only_expression_prediction.py --sumstat_summary_file ${fm_input_summary_file} --expression_prediction_output_file ${eqtl_only_expression_prediction_output_file}
fi

##########################
# 2. Joint eQTL/caQTL expression prediction model (spike-and-slab eQTL effects with prior mean = lambda x caQTL effect x peak-gene effect; fit once across all genes)
# This is an expression prediction model, not fine-mapping (inclusion probabilities are not valid PIPs under tight LD)
##########################
joint_expression_prediction_output_file="${fine_mapping_results_dir}${cell_type}_joint_eqtl_caqtl_expression_prediction_results.txt"
if false; then
python run_joint_eqtl_caqtl_expression_prediction.py --sumstat_summary_file ${fm_input_summary_file} --expression_prediction_output_file ${joint_expression_prediction_output_file}
fi

# Add mediation term
joint_expression_prediction_output_file="${fine_mapping_results_dir}${cell_type}_joint_eqtl_caqtl_mediated_expression_prediction_results.txt"
if false; then
python run_joint_eqtl_caqtl_expression_prediction.py --mediation_indicator --sumstat_summary_file ${fm_input_summary_file} --expression_prediction_output_file ${joint_expression_prediction_output_file}
fi

##########################
# 3. Chromatin-only expression prediction (stage 1 of the chromatin-informed model; see chromatin_informed_expression_prediction_model.md)
# Uses caQTL and peak-gene data only (no eQTL data): SuSiE caQTL effects of each peak x spike-and-slab peak-gene effects
# Prediction weights are the chromatin-predicted eQTL effects. Also writes the per-variant chromatin path probabilities used by step 4
##########################
chromatin_only_expression_prediction_output_file="${fine_mapping_results_dir}${cell_type}_chromatin_only_expression_prediction_results.txt"
if false; then
python run_chromatin_only_expression_prediction.py --sumstat_summary_file ${fm_input_summary_file} --expression_prediction_output_file ${chromatin_only_expression_prediction_output_file}
fi

##########################
# 4. Chromatin-informed expression prediction (stage 2; requires output of step 3)
# Spike-and-slab eQTL effects where variants with a chromatin path get their own prior inclusion probability and a slab centred on lambda x chromatin-predicted effect
# No feedback from eQTL data to the caQTL or peak-gene effects
##########################
chromatin_informed_expression_prediction_output_file="${fine_mapping_results_dir}${cell_type}_chromatin_informed_expression_prediction_results.txt"
if false; then
python run_chromatin_informed_expression_prediction.py --sumstat_summary_file ${fm_input_summary_file} --chromatin_only_results_file ${chromatin_only_expression_prediction_output_file} --expression_prediction_output_file ${chromatin_informed_expression_prediction_output_file}
fi


##########################
# 5. Chromatin-informed SuSiE fine-mapping of eQTLs (simple version; requires output of step 3)
# Standard SuSiE on the eQTL data with prior weight of each variant proportional to 1 + (enrichment - 1) x probability of chromatin path
# This is the normalized prior pi0*(1 - path_prob) + pi1*path_prob, with enrichment = pi1/pi0
# Enrichment is FIXED in each run (not learned by SuSiE). No sign gate: the chromatin-predicted direction is kept out of the prior so it can be used as a held-out check
# enrichment of 1 is exactly standard (eQTL-only) SuSiE, and is the baseline for the comparisons
##########################
# pi1/pi0 from the variational chromatin-informed expression prediction model (step 4): take pi0 and pi1 from the LAST iteration in that run's log
# 31.86 is the value at iteration 10 of the CD4_T run (pi0=0.00562, pi1=0.17905), when it was still climbing. REPLACE with the final value
variational_enrichment="46.927847745086694"

baseline_finemapping_output_file="${fine_mapping_results_dir}${cell_type}_chromatin_informed_finemapping_enrichment_1_results.txt"
for enrichment in 1 ${variational_enrichment}; do
	echo "enrichment ${enrichment}"
	chromatin_informed_finemapping_output_file="${fine_mapping_results_dir}${cell_type}_chromatin_informed_finemapping_enrichment_${enrichment}_results.txt"
	python run_chromatin_informed_susie_finemapping_of_eqtls.py --sumstat_summary_file ${fm_input_summary_file} --chromatin_only_results_file ${chromatin_only_expression_prediction_output_file} --fine_mapping_output_file ${chromatin_informed_finemapping_output_file} --no_sign_gate --enrichment ${enrichment}

	# Compare credible sets and PIPs to the baseline (enrichment of 1 = standard eQTL-only SuSiE)
	if [ "${enrichment}" != "1" ]; then
		python compare_susie_finemapping_results.py ${baseline_finemapping_output_file} ${chromatin_informed_finemapping_output_file}
	fi
done



##########################
# 6. Plots comparing eQTL-only and caQTL-mediated (Model E, elements) fine-mapping results
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
# 7. Per-link support table (Model E): implied eQTL z at each peak's caQTL lead vs the observed eQTL z, with the link probability and P_k
##########################
if false; then
python tabulate_link_support.py --mediated_results_file ${elements_finemapping_output_file} --shared_hyperparameter_file ${elements_hyperparameter_output_file} --fm_input_summary_file ${fm_input_summary_file} --output_file ${fine_mapping_results_dir}${cell_type}_link_support.txt
fi

date
