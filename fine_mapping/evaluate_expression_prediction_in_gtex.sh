#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-10:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=20G                        # Memory total in MiB (for all cores)



fingen_cell_type=${1}
gtex_tissue_type=${2}
gtex_expression_file=${3}
gtex_genotype_plink_stem=${4}
gtex_genotype_sample_mapping_file=${5}
gtex_expression_prediction_evaluation_output_file=${6}
eqtl_only_prediction_file=${7}
joint_prediction_file=${8}
chromatin_only_prediction_file=${9}
chromatin_informed_prediction_file=${10}


source ~/.bash_profile
conda activate plink_env


python evaluate_expression_prediction_in_gtex.py ${fingen_cell_type} ${gtex_tissue_type} ${gtex_expression_file} ${gtex_genotype_plink_stem} ${gtex_genotype_sample_mapping_file} ${gtex_expression_prediction_evaluation_output_file} ${eqtl_only_prediction_file} ${joint_prediction_file} ${chromatin_only_prediction_file} ${chromatin_informed_prediction_file}
