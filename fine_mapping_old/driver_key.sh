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

#####################
# Output directories
#####################

fine_mapping_output_root="/lab-share/CHIP-Strober-e2/Public/ben/caQTL/fine_mapping/"

sum_stats_fm_input_dir=${fine_mapping_output_root}"sum_stats_input/"

LD_fm_input_dir=${fine_mapping_output_root}"LD_input/"


fine_mapping_results_dir=${fine_mapping_output_root}"fine_mapping_results/"
fine_mapping_results_dir=${fine_mapping_output_root}"fine_mapping_results_debug/"
fine_mapping_results_dir=${fine_mapping_output_root}"fine_mapping_results_debug_alt_init/"


if false; then
cell_type="Mono"

    eqtl_file=${fingen_eqtl_dir}"finngen_multiome_v1.eQTL.l1."${cell_type}".cis_nominal.tsv.gz"
    caqtl_file=${fingen_caqtl_dir}"finngen_multiome_v1.caQTL.l1."${cell_type}".cis_nominal.tsv.gz"
    peak_gene_links_file=${fingen_peak_gene_links_dir}"finngen_multiome_v1.peak_gene_links.l1."${cell_type}".tsv.gz"

    sh prepare_input_data_for_fine_mapping.sh ${eqtl_file} ${caqtl_file} ${peak_gene_links_file} ${fingen_ld_dir} ${cell_type} ${gene_annotation_file} ${sum_stats_fm_input_dir} ${LD_fm_input_dir}




#####################
# Run fine-mapping (standard eQTL-only SuSiE and caQTL-mediated) on the prepared input
#####################
fi

cell_type="Mono"
    fm_input_summary_file=${sum_stats_fm_input_dir}${cell_type}"_fine_mapping_input_summary_tmp.txt"
    sh run_fine_mapping.sh ${fm_input_summary_file} ${cell_type} ${fine_mapping_results_dir}


