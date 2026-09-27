
###################
# Input data
###################

# Directory containing fingen multiome data
fingen_data_dir="/lab-share/CHIP-Strober-e2/Public/finngen/public_multiome/finngen_multiome/"

# eQTL data dir
fingen_eqtl_dir=${fingen_data_dir}"eQTL/cis_nominal/"

# caQTL data dir
fingen_caqtl_dir=${fingen_data_dir}"caQTL/cis_nominal/"

# peak-gene links data
fingen_peak_gene_links_dir=${fingen_data_dir}"peak_gene_links/"

# LOEUF file
loeuf_file="/lab-share/CHIP-Strober-e2/Public/gene_annotation_files/gnomad.v4.1.constraint_metrics.tsv"

# Directory containing re-scaling information for peak data
peak_re_scaling_dir="/lab-share/CHIP-Strober-e2/Public/finngen/public_multiome/rescaling_data_from_masa/"

# Directory containing pre-trained borzoi models
borzoi_training_dir="/lab-share/CHIP-Strober-e2/Public/ben/s2e_uncertainty/borzoi_input_data/models/"

# eQTL summary statistics
gtex_v10_eqtl_sumstats_dir="/lab-share/CHIP-Strober-e2/Public/GTEx/eqtl_sumstats/"

# Gtex v10 protein coding genes
gtex_v10_pc_genes_gtf="/lab-share/CHIP-Strober-e2/Public/gene_annotation_files/gencode.v39.gtex.protein_coding.genes.gtf"


# Borzoi target fiel
borzoi_target_file="/lab-share/CHIP-Strober-e2/Public/ben/s2e_uncertainty/borzoi_input_data/models/targets_human.txt"

# GTEx sample attributes files
# Contains tissue identity information
gtex_sample_attributes_file="/lab-share/CHIP-Strober-e2/Public/GTEx/gtex_sample_attributes/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt"


#####################
# Output data
#####################

# Output root
output_root="/lab-share/CHIP-Strober-e2/Public/ben/caQTL/borzoi/"

# Directory containing variant-gene pairs and variants to test
variant_peak_dir=${output_root}"input_variant_peak_pairs/"

# Directory containing variant-gene pairs and variants to test
borzoi_pred_dir=${output_root}"borzoi_predictions/"


peak_gene_reference_file=${fingen_peak_gene_links_dir}"finngen_multiome_v1.peak_gene_links.l1.CD4_T.tsv.gz"

variant_peak_pair_file=${variant_peak_dir}"variant_peak_pairs_to_test.txt"
variant_output_stem=${variant_peak_dir}"variants_to_test_"

# Peak "gtf" (one exon line per peak) built from the variant-peak pair file
peak_gtf_file=${variant_peak_dir}"chromatin_peaks.gtf"

# Borzoi targets file restricted to blood accessibility (DNase/ATAC) tracks
borzoi_peak_targets_file=${variant_peak_dir}"targets_human_blood_accessibility.txt"

if false; then
sh extract_variant_peak_pairs_to_test.sh $gtex_v10_eqtl_sumstats_dir $variant_peak_pair_file $variant_output_stem $gtex_v10_pc_genes_gtf $peak_gene_reference_file
fi

# Build peak gtf (from pair file) and blood-accessibility targets file for the borzoi run
if false; then
sh make_borzoi_peak_input_files.sh $variant_peak_pair_file $peak_gtf_file $borzoi_target_file $borzoi_peak_targets_file
fi



model_num="0"


if false; then
for chunk_num in {0..14}
do
	variant_vcf_file=$variant_output_stem"chunked_variants_"${chunk_num}".vcf"
	sbatch fast_borzoi_sed.sh $borzoi_pred_dir"model_"${model_num}"_chunk_"${chunk_num}"_borzoi_results.h5" ${variant_vcf_file} $borzoi_training_dir $model_num $variant_peak_pair_file $peak_gtf_file $borzoi_peak_targets_file
done
fi

