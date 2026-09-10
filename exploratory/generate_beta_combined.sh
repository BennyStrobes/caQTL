#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-2:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=70G                        # Memory total in MiB (for all cores)



cell_type="${1}"
fingen_eqtl_dir="${2}"
fingen_caqtl_dir="${3}"
fingen_peak_gene_links_dir="${4}"
pred_beta_combined_dir="${5}"

source ~/.bash_profile
conda activate plink_env


combination_version="all_links"
output_file=${pred_beta_combined_dir}${cell_type}"_beta_combined_"${combination_version}".tsv.gz"
python generate_beta_combined.py \
    --fingen_eqtl_file ${fingen_eqtl_dir}"finngen_multiome_v1.eQTL.l1."${cell_type}".cis_nominal.tsv.gz" \
    --fingen_caqtl_file ${fingen_caqtl_dir}"finngen_multiome_v1.caQTL.l1."${cell_type}".cis_nominal.tsv.gz" \
    --fingen_peak_gene_links_file ${fingen_peak_gene_links_dir}"finngen_multiome_v1.peak_gene_links.l1."${cell_type}".tsv.gz" \
    --combination_version ${combination_version} \
    --output_file ${output_file}