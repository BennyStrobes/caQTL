###############
# Input data
###############

# Directory containing fingen multiome data
fingen_data_dir="/lab-share/CHIP-Strober-e2/Public/finngen/public_multiome/finngen_multiome/"

# eQTL data dir
fingen_eqtl_dir=${fingen_data_dir}"eQTL/cis_nominal/"

# caQTL data dir
fingen_caqtl_dir=${fingen_data_dir}"caQTL/cis_nominal/"

# peak-gene links data
fingen_peak_gene_links_dir=${fingen_data_dir}"peak_gene_links/"

###############
# Output directories
###############
# Output root directory
output_root="/lab-share/CHIP-Strober-e2/Public/ben/caQTL/exploratory_analyses/"

# Directory containing ca-QTL-predicted eqtl effects (beta_combined)
pred_beta_combined_dir=${output_root}"pred_beta_combined/"

# Directory containing visualizations
visualization_dir=${output_root}"visualizations/"



###############
# Code
###############


# First create beta combined for each cell type
if false; then
for cell_type in "B" "CD4_T" "Mono"; do
    sbatch generate_beta_combined.sh ${cell_type} ${fingen_eqtl_dir} ${fingen_caqtl_dir} ${fingen_peak_gene_links_dir} ${pred_beta_combined_dir}
done
fi


# Second, plot observed vs chromatin-predicted eqtl effects by percentile of predicted effect
if false; then
for cell_type in "B" "CD4_T" "Mono"; do
    sh visualize_beta_combined.sh ${cell_type} ${pred_beta_combined_dir} ${visualization_dir}
done
fi

cell_type="B"
    sh visualize_beta_combined.sh ${cell_type} ${pred_beta_combined_dir} ${visualization_dir}


