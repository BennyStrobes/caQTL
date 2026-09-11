##################
# Input data
##################
# Genotype data for GTEx
gtex_genotype_data_dir="/lab-share/CHIP-Strober-e2/Public/ben/process_gtex_genotype_data/processed_genotype/"

# Gene annotation file
gene_annotation_file="/lab-share/CHIP-Strober-e2/Public/gene_annotation_files/genecode.v39.GRCh38.bed"


##################
# Output data
##################
output_root="/lab-share/CHIP-Strober-e2/Public/ben/caQTL/fine_mapping_simulation/"

LD_dir="${output_root}LD/"

sumstats_output_dir="${output_root}simulated_sumstats/"



##################
# Run analysis
##################
eqtl_sample_size="200"
n_genes="100"

sh generate_gene_list_sample_list_and_ld_for_simulation.sh ${gtex_genotype_data_dir} ${gene_annotation_file} ${eqtl_sample_size} ${LD_dir} $n_genes
