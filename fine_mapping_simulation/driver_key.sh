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

simulation_fine_mapping_results_dir="${output_root}simulated_fine_mapping_results/"


##################
# Run analysis
##################

######################
# Generate genotype and LD at a given sample size
# Only done once (and shared across all simulations) for memory purposes
eqtl_sample_size="200"
n_genes="100"
if false; then
sh generate_gene_list_sample_list_and_ld_for_simulation.sh ${gtex_genotype_data_dir} ${gene_annotation_file} ${eqtl_sample_size} ${LD_dir} $n_genes
fi

gene_summary_file="${LD_dir}n_genes_${n_genes}_eqtl_sample_size_${eqtl_sample_size}_cross_gene_summary.txt"

########################
# Loop through simulation numbers and run simulation
simulation_number="1"

    sh run_fine_mapping_simulation.sh ${simulation_number} ${gene_summary_file} ${sumstats_output_dir} $simulation_fine_mapping_results_dir $eqtl_sample_size