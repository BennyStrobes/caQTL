#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-20:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=20G                        # Memory total in MiB (for all cores)

#####################
# Input data
#####################

# Directory containing fingen multiome data
fingen_data_dir="/lab-share/CHIP-Strober-e2/Public/finngen/public_multiome/finngen_multiome/"

# eQTL data dir
fingen_eqtl_dir=${fingen_data_dir}"eQTL/cis_nominal/"

# caQTL data dir
fingen_caqtl_dir=${fingen_data_dir}"caQTL/cis_nominal/"

# peak-gene links data
fingen_peak_gene_links_dir=${fingen_data_dir}"peak_gene_links/"

# finngen LD data
fingen_ld_dir="/lab-share/CHIP-Strober-e2/Public/finngen/ld/finngen_ld/"

# LOEUF file
loeuf_file="/lab-share/CHIP-Strober-e2/Public/gene_annotation_files/gnomad.v4.1.constraint_metrics.tsv"

# Gene annotation file (used to get gene TSS)
gene_annotation_file="/lab-share/CHIP-Strober-e2/Public/gene_annotation_files/gencode.v39.GRCh38.genes.gtf"

# Directory containing re-scaling information for peak data
peak_re_scaling_dir="/lab-share/CHIP-Strober-e2/Public/finngen/public_multiome/rescaling_data_from_masa/"

# Directory containing gtex genotype data
gtex_genotype_data_dir="/lab-share/CHIP-Strober-e2/Public/ben/gdl_uncertainty_paper/gtex_eqtl_expression_processing/plink_processed_genotype/"

# Directory containing gtex expression data
gtex_expression_data_dir="/lab-share/CHIP-Strober-e2/Public/ben/gdl_uncertainty_paper/gtex_eqtl_expression_processing/residualized_expression/"


#####################
# Output directories
#####################

fine_mapping_output_root="/lab-share/CHIP-Strober-e2/Public/ben/caQTL/fine_mapping/"

sum_stats_fm_input_dir=${fine_mapping_output_root}"sum_stats_input/"

LD_fm_input_dir=${fine_mapping_output_root}"LD_input/"

fine_mapping_results_dir=${fine_mapping_output_root}"fine_mapping_results/"

gtex_expression_prediction_evaluation_dir=${fine_mapping_output_root}"gtex_expression_prediction_evaluation/"


if false; then
cell_type="CD4_T"

    eqtl_file=${fingen_eqtl_dir}"finngen_multiome_v1.eQTL.l1."${cell_type}".cis_nominal.tsv.gz"
    caqtl_file=${fingen_caqtl_dir}"finngen_multiome_v1.caQTL.l1."${cell_type}".cis_nominal.tsv.gz"
    peak_gene_links_file=${fingen_peak_gene_links_dir}"finngen_multiome_v1.peak_gene_links.l1."${cell_type}".tsv.gz"
    caqtl_rescaling_file=${peak_re_scaling_dir}"l1."${cell_type}".rescaling.tsv.gz"


    sbatch prepare_input_data_for_fine_mapping.sh ${eqtl_file} ${caqtl_file} ${peak_gene_links_file} ${fingen_ld_dir} ${cell_type} ${gene_annotation_file} ${caqtl_rescaling_file} ${sum_stats_fm_input_dir} ${LD_fm_input_dir}
fi



#####################
# Run fine-mapping (standard eQTL-only SuSiE and caQTL-mediated) on the prepared input
# Also runs generates genetically predicted expression
#####################
cell_type="CD4_T"
    fm_input_summary_file=${sum_stats_fm_input_dir}${cell_type}"_fine_mapping_input_summary_ld_screened_tmp.txt"
    sh run_fine_mapping.sh ${fm_input_summary_file} ${cell_type} ${fine_mapping_results_dir}



#####################
# Evaluate expresssion predictions in gtex data
#####################
fingen_cell_type="CD4_T"
gtex_tissue_type="Whole_Blood"
gtex_expression_file=${gtex_expression_data_dir}${gtex_tissue_type}".v8.residualized_expression_renormalized.bed"
gtex_genotype_plink_stem=${gtex_genotype_data_dir}"gtex_v9_eqtl_chr"
gtex_genotype_sample_mapping_file=${gtex_genotype_data_dir}"genotype_sample_mapping_to_"${gtex_tissue_type}"_expression_samples.txt"
gtex_expression_prediction_evaluation_output_file=${gtex_expression_prediction_evaluation_dir}${fingen_cell_type}"_expression_prediction_evaluation_in_"${gtex_tissue_type}".txt"

eqtl_only_prediction_file=${fine_mapping_results_dir}${fingen_cell_type}"_eqtl_only_expression_prediction_results.txt"
joint_prediction_file=${fine_mapping_results_dir}${fingen_cell_type}"_joint_eqtl_caqtl_expression_prediction_results.txt"
chromatin_only_prediction_file=${fine_mapping_results_dir}${fingen_cell_type}"_chromatin_only_expression_prediction_results.txt"
chromatin_informed_prediction_file=${fine_mapping_results_dir}${fingen_cell_type}"_chromatin_informed_expression_prediction_results.txt"
if false; then
sh evaluate_expression_prediction_in_gtex.sh ${fingen_cell_type} ${gtex_tissue_type} ${gtex_expression_file} ${gtex_genotype_plink_stem} ${gtex_genotype_sample_mapping_file} ${gtex_expression_prediction_evaluation_output_file} ${eqtl_only_prediction_file} ${joint_prediction_file} ${chromatin_only_prediction_file} ${chromatin_informed_prediction_file}

fi
