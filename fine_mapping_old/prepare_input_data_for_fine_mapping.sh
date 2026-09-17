#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-20:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=20G                        # Memory total in MiB (for all cores)


eqtl_file=${1}
caqtl_file=${2}
peak_gene_links_file=${3}
fingen_ld_dir=${4}
cell_type=${5}
gene_annotation_file=${6}
sum_stats_fm_input_dir=${7}
LD_fm_input_dir=${8}


date

source ~/.bash_profile
conda activate plink_env

python prepare_input_data_for_fine_mapping.py --eqtl_file ${eqtl_file} --caqtl_file ${caqtl_file} --peak_gene_links_file ${peak_gene_links_file} --fingen_ld_dir ${fingen_ld_dir} --cell_type ${cell_type} --gene_annotation_file ${gene_annotation_file} --sum_stats_fm_input_dir ${sum_stats_fm_input_dir} --LD_fm_input_dir ${LD_fm_input_dir}


date