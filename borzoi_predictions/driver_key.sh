
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

# Borzoi variant gene predictions
borzoi_variant_gene_predictions_dir="/lab-share/CHIP-Strober-e2/Public/ben/borzoi_genome_wide_run/genome_wide/borzoi_predictions/"

# Following directory contains expression files per tissue like: Whole_Blood.v8.residualized_expression_renormalized.bed
gtex_expression_directory="/lab-share/CHIP-Strober-e2/Public/ben/gdl_uncertainty_paper/gtex_eqtl_expression_processing/residualized_expression/"

# Following contains genotype data
# Chrom specific plink files: gtex_v9_eqtl_chr10.bed
# And then file containing sample indices: genotype_sample_mapping_to_Whole_Blood_expression_samples.txt
gtex_genotype_directory="/lab-share/CHIP-Strober-e2/Public/ben/gdl_uncertainty_paper/gtex_eqtl_expression_processing/plink_processed_genotype/"



#####################
# Output data
#####################

# Output root
output_root="/lab-share/CHIP-Strober-e2/Public/ben/caQTL/borzoi/"

# Directory containing variant-gene pairs and variants to test
variant_peak_dir=${output_root}"input_variant_peak_pairs/"

# Directory containing variant-gene pairs and variants to test
borzoi_pred_dir=${output_root}"borzoi_predictions/"


# Personalized expression predictions
personalized_expression_predictions_dir=${output_root}"personalized_expression_predictions/"


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



#################
# expression predictions
#################
# GTEx whole blood expression (residualized, renormalized) and genotypes, from gdl_uncertainty_paper/gtex_eqtl_expression_processing
gtex_tissue_type="Whole_Blood"
gtex_expression_data_dir="/lab-share/CHIP-Strober-e2/Public/ben/gdl_uncertainty_paper/gtex_eqtl_expression_processing/residualized_expression/"
gtex_genotype_data_dir="/lab-share/CHIP-Strober-e2/Public/ben/gdl_uncertainty_paper/gtex_eqtl_expression_processing/plink_processed_genotype/"
gtex_expression_file=${gtex_expression_data_dir}${gtex_tissue_type}".v8.residualized_expression_renormalized.bed"
gtex_genotype_plink_stem=${gtex_genotype_data_dir}"gtex_v9_eqtl_chr"
gtex_genotype_sample_mapping_file=${gtex_genotype_data_dir}"genotype_sample_mapping_to_"${gtex_tissue_type}"_expression_samples.txt"

# Borzoi variant-to-gene effects in whole blood (logAlt - logRef per alt allele), from gdl_uncertainty_paper/sldmc_analysis
# preprocessing of the genome-wide run. One file per borzoi target sample.
borzoi_gene_effects_dir="/lab-share/CHIP-Strober-e2/Public/ben/gdl_uncertainty_paper/sldmc_analysis/processed_borzoi/"
borzoi_whole_blood_target_sample="GTEX-1LB8K-0005-SM-DIPED.1"
borzoi_gene_effects_file=${borzoi_gene_effects_dir}${gtex_tissue_type}"_"${borzoi_whole_blood_target_sample}"_borzoi_effects.txt.gz"

# Peak re-scaling file (mean ATAC counts per nucleus per peak); only read when link_scaling is a_p
peak_re_scaling_file=${peak_re_scaling_dir}"l1.CD4_T.rescaling.tsv.gz"

# Options for the peak-mediated prediction
peak_track_set="cd4_t_dnase"          # lineage: cd4_t (CD4 T tracks; peaks and links are CD4 T) or all (all 42 blood tracks); add _dnase or _atac for one assay, e.g. cd4_t_dnase, cd4_t_atac, all_dnase, all_atac
link_scaling="none"             # none: borzoi log2FC x beta_link; a_p: additionally x mean ATAC counts per nucleus in the peak
link_significance="z_5"   # sig_count (finngen flag), sig_count_or_sig_zero (either finngen flag), or z_<thresh> e.g. z_5 (|beta_link/se_link| > 5)

expression_prediction_output_file=${personalized_expression_predictions_dir}"borzoi_gene_and_peak_link_expression_prediction_"${gtex_tissue_type}"_"${peak_track_set}"_"${link_scaling}"_"${link_significance}".txt"

sh predict_expression_from_borzoi_gene_and_peak_effects.sh $borzoi_gene_effects_file $borzoi_pred_dir $borzoi_peak_targets_file $peak_gene_reference_file $peak_re_scaling_file $gtex_expression_file $gtex_genotype_plink_stem $gtex_genotype_sample_mapping_file $peak_track_set $link_scaling $link_significance $expression_prediction_output_file

