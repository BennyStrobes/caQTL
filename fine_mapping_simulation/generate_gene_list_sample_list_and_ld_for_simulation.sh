#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-2:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=70G                        # Memory total in MiB (for all cores)


gtex_genotype_data_dir="${1}"
gene_annotation_file="${2}"
eqtl_sample_size="${3}"
LD_dir="${4}"
n_genes="${5}"

source ~/.bash_profile
conda activate plink_env


python generate_gene_list_sample_list_and_ld_for_simulation.py ${gtex_genotype_data_dir} ${gene_annotation_file} ${eqtl_sample_size} ${LD_dir} ${n_genes}