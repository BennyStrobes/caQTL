import numpy as np
import os
import sys
import pdb
import gzip
import glob
import re
import h5py
from pandas_plink import read_plink


# Two genetically predicted expression scores per gene, each correlated with observed GTEx expression:
#   (1) borzoi_gene:      sum_v x_v * beta_gene(v)                       beta_gene = borzoi log2(alt/ref) of gene expression, per alt allele
#   (2) borzoi_peak_link: sum_v x_v * sum_k beta_peak_k(v) * beta_link_k  beta_peak = borzoi log2(alt/ref) of peak k accessibility, per alt allele
#                                                                         beta_link = significant peak-gene link effect (zero if not significant)
# Allele / scale conventions (checked, not assumed):
#   - Both borzoi effects are per copy of the alt allele of chr_pos_ref_alt. The genotype matrix is converted to alt-allele dosage
#     (0/1/2) using the plink a0/a1 alleles, whichever order plink stores them in. Variants whose alleles do not match are dropped.
#   - Effects are per allele, so they are applied to raw (centered) dosage, not standardized genotype. Standardizing genotype and
#     multiplying effects by the genotype sd (as in gdl_uncertainty_paper/expression_correlations) gives the identical prediction
#     up to centering, and correlations are invariant to that, so raw dosage is used here for clarity.
#   - Missing genotype calls are mean imputed (as in ca_qtls/fine_mapping/evaluate_expression_prediction_in_gtex.py).


# Description patterns (text after 'DNASE:' / 'ATAC:') that define the CD4 T-cell lineage within the blood accessibility targets file
cd4_t_dnase_patterns = [r'CD4-positive, alpha-beta T cell', r'^T-helper \d+ cell', r'^regulatory T cell']
cd4_t_atac_patterns = [r'^T lymphocyte 2 \(CD4\+\)$', r'^Naive T$']

# Max se of a peak-gene link (matches ca_qtls/fine_mapping/prepare_input_data_for_fine_mapping.py)
max_se_link = 1.0

true_strings = set(['TRUE', 'True', 'true', '1'])

# Rows of the borzoi peak h5 files read at a time
h5_chunk_size = 50000


def variant_key_from_id(variant_id):
	# chr21_31125413_G_T_b38 -> chr21_31125413_G_T (chr_pos_ref_alt; effect allele is alt)
	data = variant_id.split('_')
	if len(data) < 4:
		print('assumption error: unexpected variant id ' + variant_id)
		sys.exit(1)
	return '_'.join(data[:4])


def decode_if_bytes(arr):
	if len(arr) == 0:
		return arr
	if isinstance(arr[0], bytes):
		return np.asarray([x.decode('utf-8') for x in arr])
	return arr


def extract_chunk_num_from_borzoi_file(borzoi_input_file):
	file_name = os.path.basename(borzoi_input_file)
	return int(file_name.split('_chunk_')[1].split('_')[0])


def load_in_borzoi_gene_effects(borzoi_gene_effects_file):
	# Columns (tab, header): gene variant chr snp_pos a0 a1 borzoi_effect_size
	# borzoi_effect_size = logAlt - logRef for the whole blood track, per copy of a1 (= alt allele of the variant id)
	# Returns gene -> {variant_key: effect}, gene -> chrom ('chr21')
	gene_to_effects = {}
	gene_to_chrom = {}
	f = gzip.open(borzoi_gene_effects_file, 'rt')
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		gene_id = data[0].split('.')[0]
		variant_key = variant_key_from_id(data[1])
		chrom = data[2]
		if chrom.startswith('chr') == False:
			chrom = 'chr' + chrom
		a0 = data[4]
		a1 = data[5]
		ref = variant_key.split('_')[2]
		alt = variant_key.split('_')[3]
		if a0 != ref or a1 != alt:
			print('assumption error: borzoi gene effect alleles (' + a0 + ',' + a1 + ') do not match variant id ' + data[1])
			sys.exit(1)
		effect = float(data[6])
		if gene_id not in gene_to_effects:
			gene_to_effects[gene_id] = {}
			gene_to_chrom[gene_id] = chrom
		if gene_to_chrom[gene_id] != chrom:
			print('assumption error: gene ' + gene_id + ' on two chromosomes in ' + borzoi_gene_effects_file)
			sys.exit(1)
		if variant_key in gene_to_effects[gene_id]:
			print('assumption error: variant ' + variant_key + ' repeated for gene ' + gene_id)
			sys.exit(1)
		gene_to_effects[gene_id][variant_key] = effect
	f.close()
	return gene_to_effects, gene_to_chrom


def load_in_significant_peak_gene_links(peak_gene_links_file, link_significance):
	# hurdle_combined link effect (as in ca_qtls/fine_mapping/prepare_input_data_for_fine_mapping.py):
	#   beta_link = beta_count + (1 - p) * beta_zero, p = expr_cell_num / total_cell_num
	# A link is kept only if significant (beta_link = 0 otherwise, i.e. it contributes nothing):
	#   link_significance == 'sig_count':             finngen sig_count flag is true
	#   link_significance == 'sig_count_or_sig_zero': finngen sig_count or sig_zero flag is true
	#   link_significance == 'z_<thresh>' (e.g. z_5):  |beta_link / se_link| > thresh on the hurdle_combined link
	# Links with NA fields, non-finite or > max_se_link se, or se of zero are thrown out first (degenerate fits)
	# Returns peak_id -> list of (gene_id, beta_link); gene_id -> number of significant links
	peak_to_links = {}
	gene_to_n_links = {}
	n_sig = 0
	n_bad_se = 0
	f = gzip.open(peak_gene_links_file, 'rt')
	head_count = 0
	for line in f:
		data = line.rstrip('\n').split('\t')
		if head_count == 0:
			head_count = head_count + 1
			header = data
			beta_link_col = header.index('hurdle_count_beta')
			se_link_col = header.index('hurdle_count_se')
			beta_zero_col = header.index('hurdle_zero_beta')
			se_zero_col = header.index('hurdle_zero_se')
			total_cell_col = header.index('total_cell_num')
			expr_cell_col = header.index('expr_cell_num')
			sig_count_col = header.index('sig_count')
			sig_zero_col = header.index('sig_zero')
			continue
		peak_id = data[0]
		gene_id = data[1].split('.')[0]

		if data[beta_link_col] == 'NA' or data[se_link_col] == 'NA' or data[beta_zero_col] == 'NA' or data[se_zero_col] == 'NA':
			continue
		p_expr = float(data[expr_cell_col])/float(data[total_cell_col])
		beta_link = float(data[beta_link_col]) + (1.0 - p_expr)*float(data[beta_zero_col])
		se_link = np.sqrt(float(data[se_link_col])**2 + ((1.0 - p_expr)**2)*(float(data[se_zero_col])**2))
		if (np.isfinite(beta_link) and np.isfinite(se_link)) == False or se_link > max_se_link or se_link == 0.0:
			n_bad_se = n_bad_se + 1
			continue

		sig_count = data[sig_count_col] in true_strings
		sig_zero = data[sig_zero_col] in true_strings
		if link_significance == 'sig_count':
			significant = sig_count
		elif link_significance == 'sig_count_or_sig_zero':
			significant = sig_count or sig_zero
		elif link_significance.startswith('z_'):
			z_threshold = float(link_significance.split('_')[1])
			significant = np.abs(beta_link/se_link) > z_threshold
		else:
			print('assumption error: unknown link_significance ' + link_significance)
			sys.exit(1)
		if significant == False:
			continue

		if peak_id not in peak_to_links:
			peak_to_links[peak_id] = []
		peak_to_links[peak_id].append((gene_id, beta_link))
		if gene_id not in gene_to_n_links:
			gene_to_n_links[gene_id] = 0
		gene_to_n_links[gene_id] = gene_to_n_links[gene_id] + 1
		n_sig = n_sig + 1
	f.close()
	print(str(n_sig) + ' significant peak-gene links (' + link_significance + ') across ' + str(len(peak_to_links)) + ' peaks and ' + str(len(gene_to_n_links)) + ' genes; ' + str(n_bad_se) + ' links dropped for bad se before the significance test')
	return peak_to_links, gene_to_n_links


def load_in_peak_mean_counts(peak_re_scaling_file):
	# peak_id -> a_p, mean ATAC counts per nucleus in the peak (column positions as in ca_qtls/fine_mapping/prepare_input_data_for_fine_mapping.py)
	mapping = {}
	f = gzip.open(peak_re_scaling_file, 'rt')
	head_count = 0
	for line in f:
		data = line.rstrip('\n').split('\t')
		if head_count == 0:
			head_count = head_count + 1
			header = data
			continue
		if len(data) != len(header):
			print('assumption error: ragged line in ' + peak_re_scaling_file)
			sys.exit(1)
		mapping[data[2]] = float(data[15])
	f.close()
	return mapping


def apply_link_scaling(peak_to_links, link_scaling, peak_re_scaling_file):
	# link_scaling == 'none': predicted per-allele log2 expression change = borzoi_log2FC_peak * beta_link
	#                          (borzoi effects are already in log2 accessibility units; analog of 'robust_sd' in the caQTL pipeline)
	# link_scaling == 'a_p':  additionally multiply by a_p, the mean ATAC count per nucleus in the peak. First order,
	#                          delta count per allele = ln2 * a_p * log2FC, beta_link is per count on natural-log expression, and
	#                          the ln2 cancels when converting back to log2 expression (analog of 'a_p_robust_sd')
	if link_scaling == 'none':
		return peak_to_links
	if link_scaling != 'a_p':
		print('assumption error: unknown link_scaling ' + link_scaling)
		sys.exit(1)
	peak_to_a_p = load_in_peak_mean_counts(peak_re_scaling_file)
	scaled = {}
	n_missing = 0
	for peak_id in [*peak_to_links]:
		if peak_id not in peak_to_a_p:
			n_missing = n_missing + 1
			continue
		scaled[peak_id] = [(gene_id, beta_link*peak_to_a_p[peak_id]) for gene_id, beta_link in peak_to_links[peak_id]]
	print(str(n_missing) + ' linked peaks missing from re-scaling file (dropped)')
	return scaled


def extract_peak_track_columns(borzoi_peak_targets_file, peak_track_set):
	# The rows of the targets file used for the borzoi peak run are, in order, the columns of logRef/logAlt in the h5 files
	# peak_track_set is '<lineage>' or '<lineage>_<assay>':
	#   lineage 'all':   every track in the file (all blood mononuclear lineages)
	#   lineage 'cd4_t': CD4 T-cell lineage tracks only (the peaks and peak-gene links are from CD4 T cells)
	#   assay 'dnase' / 'atac': restrict to that assay; no assay suffix uses both
	# e.g. cd4_t, cd4_t_dnase, cd4_t_atac, all, all_dnase, all_atac
	if peak_track_set.endswith('_dnase'):
		lineage = peak_track_set[:-len('_dnase')]
		assays = ['DNASE:']
	elif peak_track_set.endswith('_atac'):
		lineage = peak_track_set[:-len('_atac')]
		assays = ['ATAC:']
	else:
		lineage = peak_track_set
		assays = ['DNASE:', 'ATAC:']
	if lineage not in ['all', 'cd4_t']:
		print('assumption error: unknown peak_track_set ' + peak_track_set)
		sys.exit(1)
	columns = []
	descriptions = []
	f = open(borzoi_peak_targets_file)
	head_count = 0
	row_index = 0
	for line in f:
		line = line.rstrip('\n')
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		description = data[8]
		if description.startswith('DNASE:') == False and description.startswith('ATAC:') == False:
			print('assumption error: track is neither DNASE nor ATAC: ' + description)
			sys.exit(1)
		keep = False
		if description.split(':', 1)[0] + ':' in assays:
			if lineage == 'all':
				keep = True
			else:
				if description.startswith('DNASE:'):
					patterns = cd4_t_dnase_patterns
				else:
					patterns = cd4_t_atac_patterns
				text = description.split(':', 1)[1]
				for pattern in patterns:
					if re.search(pattern, text) is not None:
						keep = True
		if keep:
			columns.append(row_index)
			descriptions.append(description)
		row_index = row_index + 1
	f.close()
	if len(columns) == 0:
		print('assumption error: no peak tracks selected')
		sys.exit(1)
	print(str(len(columns)) + ' of ' + str(row_index) + ' peak tracks used (' + peak_track_set + '):')
	for description in descriptions:
		print('\t' + description)
	return np.asarray(columns)


def load_in_borzoi_peak_link_gene_effects(borzoi_peak_results_dir, track_columns, peak_to_links, genes_to_keep):
	# Streams the peak borzoi h5 files (one row per variant-peak pair; logRef/logAlt are rows X tracks) and accumulates
	#   gene -> {variant_key: sum_k beta_peak_k(variant) * beta_link_k}
	# where beta_peak = mean over the selected tracks of logAlt - logRef (log2 fold change of peak accessibility per alt allele)
	# Also returns gene -> set of peaks that contributed
	gene_to_effects = {}
	gene_to_peaks = {}
	file_pattern = os.path.join(borzoi_peak_results_dir, 'model_0_chunk_*_borzoi_results.h5')
	borzoi_input_files = sorted(glob.glob(file_pattern), key=extract_chunk_num_from_borzoi_file)
	if len(borzoi_input_files) == 0:
		print('assumption error: no borzoi peak results files match ' + file_pattern)
		sys.exit(1)
	n_rows_total = 0
	n_rows_used = 0
	for borzoi_input_file in borzoi_input_files:
		if borzoi_input_file != '/lab-share/CHIP-Strober-e2/Public/ben/caQTL/borzoi/borzoi_predictions/model_0_chunk_0_borzoi_results.h5':
			continue
		print('need to include all borzoi input files')
		print(borzoi_input_file)
		with h5py.File(borzoi_input_file, 'r') as f:
			snp_ds = f['snp']
			peak_ds = f['gene']
			logRef_ds = f['logRef']
			logAlt_ds = f['logAlt']
			n_rows = peak_ds.shape[0]
			for start in range(0, n_rows, h5_chunk_size):
				end = min(start + h5_chunk_size, n_rows)
				snp_ids = decode_if_bytes(snp_ds[start:end])
				peak_ids = decode_if_bytes(peak_ds[start:end])
				logRef = logRef_ds[start:end, :][:, track_columns]
				logAlt = logAlt_ds[start:end, :][:, track_columns]
				peak_effects = np.mean(logAlt - logRef, axis=1)
				n_rows_total = n_rows_total + (end - start)
				for ii in range(end - start):
					peak_id = peak_ids[ii]
					if peak_id not in peak_to_links:
						continue
					variant_key = variant_key_from_id(snp_ids[ii])
					peak_effect = float(peak_effects[ii])
					for gene_id, beta_link in peak_to_links[peak_id]:
						if gene_id not in genes_to_keep:
							continue
						if gene_id not in gene_to_effects:
							gene_to_effects[gene_id] = {}
							gene_to_peaks[gene_id] = {}
						if variant_key not in gene_to_effects[gene_id]:
							gene_to_effects[gene_id][variant_key] = 0.0
						gene_to_effects[gene_id][variant_key] = gene_to_effects[gene_id][variant_key] + peak_effect*beta_link
						gene_to_peaks[gene_id][peak_id] = 1
						n_rows_used = n_rows_used + 1
	print(str(n_rows_total) + ' variant-peak rows read; ' + str(n_rows_used) + ' variant-peak-gene contributions to ' + str(len(gene_to_effects)) + ' genes')
	return gene_to_effects, gene_to_peaks


def load_in_gtex_expression(gtex_expression_file):
	# Bed file: chrom, start, tss, gene_id, then one column per sample (columns ordered as the genotype sample mapping)
	gene_to_expression = {}
	f = open(gtex_expression_file)
	head_count = 0
	for line in f:
		data = line.rstrip('\n').split('\t')
		if head_count == 0:
			head_count = head_count + 1
			n_samples = len(data[4:])
			continue
		gene_id = data[3].split('.')[0]
		chrom = data[0]
		if chrom.startswith('chr') == False:
			chrom = 'chr' + chrom
		if gene_id in gene_to_expression:
			print('assumption error: gene ' + gene_id + ' repeated in ' + gtex_expression_file)
			sys.exit(1)
		gene_to_expression[gene_id] = (chrom, np.asarray(data[4:]).astype(float))
	f.close()
	return gene_to_expression, n_samples


def create_mapping_from_variant_to_genotype_index(bim):
	# pandas_plink: a0 is the first allele in the bim file, a1 the second, and G counts copies of a1
	# Each gtex variant is entered under both allele orderings of chr_pos_ref_alt so that a borzoi variant with the same alleles is
	# matched whichever way plink stores them. Value: (genotype index, True if G counts the alt allele of that key / False if it counts ref)
	chroms = np.asarray(bim['chrom']).astype(str)
	positions = np.asarray(bim['pos']).astype(str)
	a0s = np.asarray(bim['a0']).astype(str)
	a1s = np.asarray(bim['a1']).astype(str)
	mapping = {}
	n_repeats = 0
	for index in range(len(chroms)):
		chrom = chroms[index]
		if chrom.startswith('chr') == False:
			chrom = 'chr' + chrom
		stem = chrom + '_' + positions[index] + '_'
		key_alt_is_a1 = stem + a0s[index] + '_' + a1s[index]
		key_alt_is_a0 = stem + a1s[index] + '_' + a0s[index]
		if key_alt_is_a1 in mapping or key_alt_is_a0 in mapping:
			n_repeats = n_repeats + 1
			continue
		mapping[key_alt_is_a1] = (index, True)
		mapping[key_alt_is_a0] = (index, False)
	return mapping, n_repeats


def extract_centered_alt_allele_dosage(variant_keys, variant_to_genotype_index, G, ordered_genotype_indices):
	# For variants found in gtex with matching alleles: centered dosage (0/1/2 minus its mean) of the alt allele, n_kept X n_samples
	# Returns the dosage matrix, boolean vector of kept variants, and number of kept variants whose alleles plink stores alt-first
	n_variants = len(variant_keys)
	found = np.zeros(n_variants, dtype=bool)
	genotype_indices = []
	g_counts_alt = []
	for variant_iter, variant_key in enumerate(variant_keys):
		if variant_key in variant_to_genotype_index:
			found[variant_iter] = True
			genotype_indices.append(variant_to_genotype_index[variant_key][0])
			g_counts_alt.append(variant_to_genotype_index[variant_key][1])
	if np.sum(found) == 0:
		return np.zeros((0, len(ordered_genotype_indices))), found, 0
	genotype_indices = np.asarray(genotype_indices)
	g_counts_alt = np.asarray(g_counts_alt)

	# Genotype matrix (dask needs sorted, unique row indices)
	unique_indices, inverse = np.unique(genotype_indices, return_inverse=True)
	geno_mat = (G[unique_indices, :].compute())[:, ordered_genotype_indices]
	geno_mat = np.asarray(geno_mat, dtype=float)[inverse, :]

	# G counts a1; where the alt allele is a0, flip to alt dosage
	geno_mat[g_counts_alt == False, :] = 2.0 - geno_mat[g_counts_alt == False, :]

	# Mean impute missing calls
	variant_means = np.nanmean(geno_mat, axis=1)
	missing_rows, missing_cols = np.where(np.isnan(geno_mat))
	geno_mat[missing_rows, missing_cols] = variant_means[missing_rows]

	# Drop variants that do not vary in these samples (they would contribute nothing after centering)
	variant_sdevs = np.std(geno_mat, axis=1)
	varying = (variant_sdevs > 0.0) & np.isfinite(variant_sdevs)
	found[np.where(found)[0][varying == False]] = False
	geno_mat = geno_mat[varying, :]

	centered_geno_mat = geno_mat - np.mean(geno_mat, axis=1)[:, None]
	n_alt_first = np.sum(g_counts_alt[varying] == False)
	return centered_geno_mat, found, n_alt_first


def predict_expression(variant_to_effect, variant_to_genotype_index, G, ordered_genotype_indices):
	# Genetically predicted expression = centered alt-allele dosage weighted by per-allele effects, summed over variants
	# Returns prediction vector (or None if no usable variant), number of variants, number found in gtex
	variant_keys = np.asarray([*variant_to_effect])
	if len(variant_keys) == 0:
		return None, 0, 0
	effects = np.asarray([variant_to_effect[variant_key] for variant_key in variant_keys])
	centered_geno_mat, found, n_alt_first = extract_centered_alt_allele_dosage(variant_keys, variant_to_genotype_index, G, ordered_genotype_indices)
	if np.sum(found) == 0:
		return None, len(variant_keys), 0
	pred_expr = np.dot(effects[found], centered_geno_mat)
	return pred_expr, len(variant_keys), int(np.sum(found))


def correlation(x, y):
	if x is None or y is None:
		return np.nan
	if np.std(x) == 0.0 or np.std(y) == 0.0:
		return np.nan
	return np.corrcoef(x, y)[0, 1]


def print_summary(name, corrs):
	corrs = np.asarray(corrs)
	valid = np.isfinite(corrs)
	if np.sum(valid) == 0:
		print(name + ': 0 genes with a prediction')
		return
	print(name + ': ' + str(int(np.sum(valid))) + ' genes; mean correlation=' + str(np.mean(corrs[valid])) + '; median correlation=' + str(np.median(corrs[valid])) + '; mean squared correlation=' + str(np.mean(np.square(corrs[valid]))) + '; fraction positive=' + str(np.mean(corrs[valid] > 0.0)))
	return


#####################
# Command line args
#####################
borzoi_gene_effects_file = sys.argv[1]
borzoi_peak_results_dir = sys.argv[2]
borzoi_peak_targets_file = sys.argv[3]
peak_gene_links_file = sys.argv[4]
peak_re_scaling_file = sys.argv[5]
gtex_expression_file = sys.argv[6]
gtex_genotype_plink_stem = sys.argv[7]
gtex_genotype_sample_mapping_file = sys.argv[8]
peak_track_set = sys.argv[9]
link_scaling = sys.argv[10]
link_significance = sys.argv[11]
output_file = sys.argv[12]


###########################
# Load in data
###########################
# Borzoi variant-to-gene effects (per alt allele)
gene_to_borzoi_gene_effects, gene_to_chrom = load_in_borzoi_gene_effects(borzoi_gene_effects_file)
print(str(len(gene_to_borzoi_gene_effects)) + ' genes with borzoi gene effects')

# GTEx expression and the genotype sample indices matching its columns
gene_to_expression, n_samples = load_in_gtex_expression(gtex_expression_file)
ordered_genotype_indices = (np.loadtxt(gtex_genotype_sample_mapping_file)).astype(int)
if len(ordered_genotype_indices) != n_samples:
	print('assumption error: number of expression samples does not match genotype sample mapping')
	sys.exit(1)

# Genes to analyze: borzoi gene effects and expression (peak-based prediction is NA for genes without significant links)
gene_ids = np.sort([gene_id for gene_id in gene_to_borzoi_gene_effects if gene_id in gene_to_expression])
genes_to_keep = {}
for gene_id in gene_ids:
	genes_to_keep[gene_id] = 1
print(str(len(gene_ids)) + ' genes with borzoi gene effects and gtex expression')

# Significant peak-gene links (beta_link is zero, i.e. dropped, for non-significant links), optionally rescaled
peak_to_links, gene_to_n_links = load_in_significant_peak_gene_links(peak_gene_links_file, link_significance)
peak_to_links = apply_link_scaling(peak_to_links, link_scaling, peak_re_scaling_file)

# Borzoi variant-to-peak effects (mean over selected tracks) times link effects, accumulated per gene
track_columns = extract_peak_track_columns(borzoi_peak_targets_file, peak_track_set)
gene_to_borzoi_peak_link_effects, gene_to_peaks_used = load_in_borzoi_peak_link_gene_effects(borzoi_peak_results_dir, track_columns, peak_to_links, genes_to_keep)


###########################
# Predict expression and correlate with observed, one chromosome of genotypes at a time
###########################
t = open(output_file, 'w')
t.write('gene_id\tchrom\tn_gtex_samples\tn_variants_borzoi_gene\tn_variants_borzoi_gene_in_gtex\tborzoi_gene_expression_correlation\tn_significant_peaks\tn_significant_peaks_with_borzoi\tn_variants_borzoi_peak_link\tn_variants_borzoi_peak_link_in_gtex\tborzoi_peak_link_expression_correlation\tgene_vs_peak_link_prediction_correlation\n')

gene_corrs = []
peak_link_corrs = []
n_genes_written = 0
n_genes_chrom_mismatch = 0
for chrom_num in range(1, 3):
	chrom_string = 'chr' + str(chrom_num)
	chrom_gene_ids = [gene_id for gene_id in gene_ids if gene_to_chrom[gene_id] == chrom_string]
	if len(chrom_gene_ids) == 0:
		continue
	print(chrom_string + ': ' + str(len(chrom_gene_ids)) + ' genes')
	sys.stdout.flush()

	(bim, fam, G) = read_plink(gtex_genotype_plink_stem + str(chrom_num), verbose=False)
	variant_to_genotype_index, n_repeats = create_mapping_from_variant_to_genotype_index(bim)

	for gene_id in chrom_gene_ids:
		expr_chrom, expr_vec = gene_to_expression[gene_id]
		if expr_chrom != chrom_string:
			n_genes_chrom_mismatch = n_genes_chrom_mismatch + 1
			continue

		# (1) Borzoi variant-to-gene prediction
		gene_pred, n_var_gene, n_var_gene_in_gtex = predict_expression(gene_to_borzoi_gene_effects[gene_id], variant_to_genotype_index, G, ordered_genotype_indices)
		gene_corr = correlation(gene_pred, expr_vec)

		# (2) Borzoi variant-to-peak times peak-gene link prediction
		n_sig_peaks = gene_to_n_links.get(gene_id, 0)
		if gene_id in gene_to_borzoi_peak_link_effects:
			peak_link_pred, n_var_peak, n_var_peak_in_gtex = predict_expression(gene_to_borzoi_peak_link_effects[gene_id], variant_to_genotype_index, G, ordered_genotype_indices)
			n_sig_peaks_with_borzoi = len(gene_to_peaks_used[gene_id])
		else:
			peak_link_pred, n_var_peak, n_var_peak_in_gtex = None, 0, 0
			n_sig_peaks_with_borzoi = 0
		peak_link_corr = correlation(peak_link_pred, expr_vec)
		pred_pred_corr = correlation(gene_pred, peak_link_pred)

		t.write(gene_id + '\t' + chrom_string + '\t' + str(n_samples) + '\t' + str(n_var_gene) + '\t' + str(n_var_gene_in_gtex) + '\t' + str(gene_corr) + '\t' + str(n_sig_peaks) + '\t' + str(n_sig_peaks_with_borzoi) + '\t' + str(n_var_peak) + '\t' + str(n_var_peak_in_gtex) + '\t' + str(peak_link_corr) + '\t' + str(pred_pred_corr) + '\n')
		gene_corrs.append(gene_corr)
		peak_link_corrs.append(peak_link_corr)
		n_genes_written = n_genes_written + 1
	t.flush()

t.close()
print(str(n_genes_written) + ' genes written; ' + str(n_genes_chrom_mismatch) + ' skipped for chromosome mismatch between borzoi and gtex')
print_summary('borzoi_gene', gene_corrs)
print_summary('borzoi_peak_link', peak_link_corrs)
both = np.isfinite(np.asarray(gene_corrs)) & np.isfinite(np.asarray(peak_link_corrs))
if np.sum(both) > 0:
	print_summary('borzoi_gene (genes with both predictions)', np.asarray(gene_corrs)[both])
	print_summary('borzoi_peak_link (genes with both predictions)', np.asarray(peak_link_corrs)[both])
