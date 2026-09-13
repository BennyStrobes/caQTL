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

visualization_dir="${output_root}visualizations/"


##################
# Run analysis
##################

######################
# Generate genotype and LD at a given sample size
# Only done once (and shared across all simulations) for memory purposes
eqtl_sample_size="200"
n_genes="300"
if false; then
sh generate_gene_list_sample_list_and_ld_for_simulation.sh ${gtex_genotype_data_dir} ${gene_annotation_file} ${eqtl_sample_size} ${LD_dir} $n_genes
fi

gene_summary_file="${LD_dir}n_genes_${n_genes}_eqtl_sample_size_${eqtl_sample_size}_cross_gene_summary.txt"
########################
# Loop through simulation numbers and run simulation
n_simulations="20"
if false; then

for simulation_number in $(seq 1 ${n_simulations}); do
    sbatch run_fine_mapping_simulation.sh ${simulation_number} ${gene_summary_file} ${sumstats_output_dir} $simulation_fine_mapping_results_dir $eqtl_sample_size
done
fi


########################
# Organize results across simulations and make calibration / power plots
organized_results_file="${simulation_fine_mapping_results_dir}organized_fine_mapping_results_n_simulations_${n_simulations}_eqtl_sample_size_${eqtl_sample_size}.txt"
sh organize_and_visualize_fine_mapping_simulation_results.sh ${simulation_fine_mapping_results_dir} ${sumstats_output_dir} ${n_simulations} ${eqtl_sample_size} ${organized_results_file} ${visualization_dir}

