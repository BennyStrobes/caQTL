#!/bin/bash
#SBATCH -t 0-5:30                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                        # Partition to run in
#SBATCH --mem=10GB 




eqtl_sumstats_dir="${1}"
variant_peak_pair_file="${2}"
variant_output_stem="${3}"
gtex_v10_pc_genes_gtf="${4}"
peak_gene_reference_file="${5}"

source ~/.bashrc
conda activate borzoi

python extract_variant_peak_pairs_to_test.py $eqtl_sumstats_dir $variant_peak_pair_file $variant_output_stem $gtex_v10_pc_genes_gtf $peak_gene_reference_file



vcf_file=$variant_output_stem"all_variant.vcf"
split -n l/15 -d --additional-suffix=.vcf "$vcf_file" ${variant_output_stem}chunked_variants_

for f in ${variant_output_stem}chunked_variants_*.vcf; do
    mv "$f" "$(echo "$f" | sed 's/_0\([0-9]\.vcf\)$/_\1/')"
done
