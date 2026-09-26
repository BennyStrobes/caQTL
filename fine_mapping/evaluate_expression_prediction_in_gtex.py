import numpy as np
import pdb
import sys
import os
from pandas_plink import read_plink
from scipy import stats
from scipy.optimize import minimize_scalar


def load_in_prediction_file(prediction_file):
    # Results file from run_eqtl_only_expression_prediction.py / run_joint_eqtl_caqtl_expression_prediction.py
    # Returns dictionary: gene_id (no version) -> (variant_ids_file, array of prediction weights)
    # Weights are posterior mean eQTL effects of the alt allele (last allele in finngen variant id chr_pos_ref_alt), per sd of genotype
    gene_to_weights = {}
    f = open(prediction_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            continue
        gene_id = data[0].split('.')[0]
        variant_ids_file = data[1]
        weights = np.asarray(data[2].split(',')).astype(float)
        if gene_id in gene_to_weights:
            print('assumption error: gene ' + gene_id + ' repeated in ' + prediction_file)
            pdb.set_trace()
        gene_to_weights[gene_id] = (variant_ids_file, weights)
    f.close()
    return gene_to_weights


def load_in_chromatin_path_probs(chromatin_only_prediction_file):
    # Results file from run_chromatin_only_expression_prediction.py: eqtl_alpha column is the probability each variant has a chromatin path to the gene
    # Returns dictionary: gene_id (no version) -> array of chromatin path probabilities
    gene_to_path_probs = {}
    f = open(chromatin_only_prediction_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            path_prob_col = data.index('eqtl_alpha')
            continue
        gene_to_path_probs[data[0].split('.')[0]] = np.asarray(data[path_prob_col].split(',')).astype(float)
    f.close()
    return gene_to_path_probs


def load_in_gtex_expression(gtex_expression_file):
    # Bed file: chrom, start, tss, gene_id, then one column per sample
    # Returns dictionary: gene_id (no version) -> (chrom, expression vector), and number of samples
    gene_to_expression = {}
    f = open(gtex_expression_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            n_samples = len(data[4:])
            continue
        gene_id = data[3].split('.')[0]
        chrom = data[0]
        if chrom.startswith('chr') == False:
            chrom = 'chr' + chrom
        gene_to_expression[gene_id] = (chrom, np.asarray(data[4:]).astype(float))
    f.close()
    return gene_to_expression, n_samples


def create_mapping_from_variant_to_genotype_index(bim):
    # pandas_plink: a0 is first allele in bim file, a1 is second allele in bim file, and genotype matrix G counts the number of a1 alleles
    # Finngen variant id is chr_pos_ref_alt and finngen effect allele is alt
    # Returns dictionary: 'chr_pos_ref_alt' -> (genotype index, True if G counts the alt allele / False if G counts the ref allele)
    # Each gtex variant is entered under both allele orderings so that finngen variants with swapped ref/alt are matched (with correct sign)
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
            # Same position and alleles listed twice in gtex genotype. Keep first
            n_repeats += 1
            continue
        mapping[key_alt_is_a1] = (index, True)
        mapping[key_alt_is_a0] = (index, False)
    return mapping, n_repeats


def extract_standardized_effect_allele_genotype(variant_ids, variant_to_genotype_index, G, ordered_genotype_indices):
    # For variants found in gtex: standardized (mean 0, variance 1 in gtex samples) dosage of the finngen effect (alt) allele
    # Returns standardized genotype matrix (n_kept_variants X n_samples), boolean vector of kept variants, number of kept variants with alleles swapped relative to gtex
    n_variants = len(variant_ids)
    found = np.zeros(n_variants, dtype=bool)
    genotype_indices = []
    g_counts_alt = []
    for variant_iter, variant_id in enumerate(variant_ids):
        if variant_id in variant_to_genotype_index:
            found[variant_iter] = True
            genotype_indices.append(variant_to_genotype_index[variant_id][0])
            g_counts_alt.append(variant_to_genotype_index[variant_id][1])
    if np.sum(found) == 0:
        return np.zeros((0, len(ordered_genotype_indices))), found, 0
    genotype_indices = np.asarray(genotype_indices)
    g_counts_alt = np.asarray(g_counts_alt)

    # Genotype matrix (dask needs sorted, unique row indices)
    unique_indices, inverse = np.unique(genotype_indices, return_inverse=True)
    geno_mat = (G[unique_indices, :].compute())[:, ordered_genotype_indices]
    geno_mat = geno_mat[inverse, :]

    # Convert to dosage of finngen effect (alt) allele
    geno_mat[g_counts_alt == False, :] = 2.0 - geno_mat[g_counts_alt == False, :]

    # Mean impute missing genotype calls
    variant_means = np.nanmean(geno_mat, axis=1)
    missing_rows, missing_cols = np.where(np.isnan(geno_mat))
    geno_mat[missing_rows, missing_cols] = variant_means[missing_rows]

    # Drop variants that do not vary in gtex samples
    variant_sdevs = np.std(geno_mat, axis=1)
    varying = (variant_sdevs > 0.0) & np.isfinite(variant_sdevs)
    found[np.where(found)[0][varying == False]] = False
    geno_mat = geno_mat[varying, :]

    # Standardize (prediction weights are per sd of genotype)
    standardized_geno_mat = (geno_mat - np.mean(geno_mat, axis=1)[:, None])/np.std(geno_mat, axis=1)[:, None]

    # Finngen ref/alt agrees with gtex ref/alt when gtex ref is a1 (G counts ref)
    n_swapped = np.sum(g_counts_alt[varying] == True)
    return standardized_geno_mat, found, n_swapped


def estimate_cis_snp_heritability_with_lrt(genotype_mat, expr_vec):
    # Same function as in gdl_uncertainty_paper/expression_correlations/personalized_expression_correlations_per_tissue.py
    # genotype_mat is n_samples X n_variants
    # Cis-SNP heritability via maximum likelihood on a single variance component model
    # (y ~ N(0, h2*GRM + (1-h2)*I) up to a scale), with a likelihood ratio test against h2 = 0.
    # The p-value uses the 50:50 mixture of chi2(0) and chi2(1) since h2 = 0 is on the boundary.
    X = np.asarray(genotype_mat, dtype=float)
    y = np.asarray(expr_vec, dtype=float)
    y = y - np.mean(y)

    X = X - np.mean(X, axis=0)
    snp_sdevs = np.std(X, axis=0)
    valid_snps = np.isfinite(snp_sdevs) & (snp_sdevs > 0.0)
    X = X[:, valid_snps]
    snp_sdevs = snp_sdevs[valid_snps]
    if X.shape[1] == 0:
        return np.nan, np.nan

    X = X/snp_sdevs[None, :]
    n_samples = X.shape[0]
    grm = np.dot(X, np.transpose(X))/X.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(grm)
    transformed_y = np.dot(np.transpose(eigenvectors), y)

    def log_likelihood(h2):
        variance_scale = h2*eigenvalues + (1.0 - h2)
        if np.any(variance_scale <= 0.0):
            return -np.inf
        residual_var = np.mean(np.square(transformed_y)/variance_scale)
        if residual_var <= 0.0:
            return -np.inf
        return -0.5*(n_samples*np.log(2.0*np.pi) + n_samples*np.log(residual_var) + np.sum(np.log(variance_scale)) + n_samples)

    null_log_likelihood = log_likelihood(0.0)
    opt = minimize_scalar(lambda h2: -log_likelihood(h2), bounds=(0.0, 0.999999), method='bounded')
    h2 = opt.x
    alt_log_likelihood = -opt.fun
    lrt_stat = np.maximum(2.0*(alt_log_likelihood - null_log_likelihood), 0.0)
    lrt_pvalue = 0.5*stats.chi2.sf(lrt_stat, df=1)
    return h2, lrt_pvalue


def print_summary(method_names, method_to_corrs, gene_subset, subset_name):
    # gene_subset: boolean vector of genes to summarize
    print('##########################')
    print(subset_name + ' (' + str(np.sum(gene_subset)) + ' genes)')
    if np.sum(gene_subset) == 0:
        return
    for method_name in method_names:
        corrs = np.asarray(method_to_corrs[method_name])[gene_subset]
        valid = np.isfinite(corrs)
        if np.sum(valid) == 0:
            print(method_name + ': 0 genes with a prediction')
            continue
        print(method_name + ': ' + str(np.sum(valid)) + ' genes; mean correlation=' + str(np.mean(corrs[valid])) + '; median correlation=' + str(np.median(corrs[valid])) + '; mean squared correlation=' + str(np.mean(np.square(corrs[valid]))) + '; fraction positive=' + str(np.mean(corrs[valid] > 0.0)))
    # Paired comparisons to first method
    base_corrs = np.asarray(method_to_corrs[method_names[0]])[gene_subset]
    for method_name in method_names[1:]:
        corrs = np.asarray(method_to_corrs[method_name])[gene_subset]
        valid = np.isfinite(corrs) & np.isfinite(base_corrs)
        if np.sum(valid) == 0:
            continue
        diffs = corrs[valid] - base_corrs[valid]
        print(method_name + ' - ' + method_names[0] + ' (' + str(np.sum(valid)) + ' genes): mean difference in correlation=' + str(np.mean(diffs)) + ' (se=' + str(np.std(diffs)/np.sqrt(len(diffs))) + '); fraction of genes with larger correlation=' + str(np.mean(diffs > 0.0)))
    sys.stdout.flush()
    return


######################
# Command line args
######################
fingen_cell_type = sys.argv[1]
gtex_tissue_type = sys.argv[2]
gtex_expression_file = sys.argv[3]
gtex_genotype_plink_stem = sys.argv[4]
gtex_genotype_sample_mapping_file = sys.argv[5]
gtex_expression_prediction_evaluation_output_file = sys.argv[6]
eqtl_only_prediction_file = sys.argv[7]
joint_prediction_file = sys.argv[8]
chromatin_only_prediction_file = sys.argv[9]
chromatin_informed_prediction_file = sys.argv[10]


# Methods to evaluate (paired comparisons are made to the first method)
method_names = ['eqtl_only', 'joint_eqtl_caqtl', 'chromatin_only', 'chromatin_informed']
method_to_weights = {}
method_to_weights['eqtl_only'] = load_in_prediction_file(eqtl_only_prediction_file)
method_to_weights['joint_eqtl_caqtl'] = load_in_prediction_file(joint_prediction_file)
# Chromatin only: weights are chromatin-predicted eQTL effects (caQTL effect x peak-gene effect; no eQTL data). All zero for genes with no peaks (correlation is then nan)
method_to_weights['chromatin_only'] = load_in_prediction_file(chromatin_only_prediction_file)
# Chromatin informed: eQTL spike and slab where prior depends on whether variant has a chromatin path (stage 2; uses chromatin only results as prior)
method_to_weights['chromatin_informed'] = load_in_prediction_file(chromatin_informed_prediction_file)
# Probability each variant has a chromatin path to the gene (used to stratify genes)
gene_to_path_probs = load_in_chromatin_path_probs(chromatin_only_prediction_file)

# Gtex expression (columns ordered the same as genotype sample mapping)
gene_to_expression, n_samples = load_in_gtex_expression(gtex_expression_file)

# Genotype-sample indices corresponding to the expression samples of this tissue
ordered_genotype_indices = (np.loadtxt(gtex_genotype_sample_mapping_file)).astype(int)
if len(ordered_genotype_indices) != n_samples:
    print('assumption error: number of expression samples does not match genotype sample mapping')
    pdb.set_trace()

# Genes to evaluate: in every prediction file and in gtex expression
gene_ids = np.sort([gene_id for gene_id in method_to_weights[method_names[0]] if gene_id in gene_to_expression and np.all([gene_id in method_to_weights[method_name] for method_name in method_names])])
print(str(len(gene_ids)) + ' genes in all prediction files and in gtex ' + gtex_tissue_type + ' expression (' + str(len(method_to_weights[method_names[0]])) + ' genes in ' + method_names[0] + ' prediction file)')
sys.stdout.flush()

# Open output file handle
t = open(gtex_expression_prediction_evaluation_output_file, 'w')
header = ['gene_id', 'chrom', 'n_gtex_samples', 'n_variants', 'n_variants_in_gtex', 'n_variants_in_gtex_with_swapped_alleles', 'max_chromatin_path_prob_in_gtex']
for method_name in method_names:
    header.append(method_name + '_correlation')
    header.append(method_name + '_fraction_abs_weight_in_gtex')
header.append('gtex_cis_snp_h2')
header.append('gtex_cis_snp_h2_pvalue')
t.write('\t'.join(header) + '\n')

method_to_corrs = {}
for method_name in method_names:
    method_to_corrs[method_name] = []

max_path_probs = []
n_genes_skipped = 0
for chrom_num in range(1, 23):
    chrom_string = 'chr' + str(chrom_num)
    chrom_gene_ids = [gene_id for gene_id in gene_ids if gene_to_expression[gene_id][0] == chrom_string]
    if len(chrom_gene_ids) == 0:
        continue
    print(chrom_string + ': ' + str(len(chrom_gene_ids)) + ' genes')
    sys.stdout.flush()

    # Load in chromosome plink data
    (bim, fam, G) = read_plink(gtex_genotype_plink_stem + str(chrom_num), verbose=False)
    variant_to_genotype_index, n_repeats = create_mapping_from_variant_to_genotype_index(bim)

    for gene_id in chrom_gene_ids:
        expr_vec = gene_to_expression[gene_id][1]

        # Variant ids (shared across methods)
        variant_ids_file = method_to_weights[method_names[0]][gene_id][0]
        variant_ids = np.loadtxt(variant_ids_file, dtype=str, ndmin=1)
        passed = True
        for method_name in method_names:
            if method_to_weights[method_name][gene_id][0] != variant_ids_file or len(method_to_weights[method_name][gene_id][1]) != len(variant_ids):
                passed = False
        if passed == False or variant_ids[0].split('_')[0] != chrom_string:
            # Methods do not share variants, or gene is on a different chromosome in finngen and gtex
            n_genes_skipped += 1
            continue

        # Standardized dosage of finngen effect allele for variants found in gtex (missing variants are dropped)
        standardized_geno_mat, found, n_swapped = extract_standardized_effect_allele_genotype(variant_ids, variant_to_genotype_index, G, ordered_genotype_indices)

        # Largest probability of a chromatin path among the gene's variants found in gtex
        if np.sum(found) > 0 and len(gene_to_path_probs[gene_id]) == len(variant_ids):
            max_path_prob = np.max(gene_to_path_probs[gene_id][found])
        else:
            max_path_prob = 0.0
        max_path_probs.append(max_path_prob)

        line = [gene_id, chrom_string, str(n_samples), str(len(variant_ids)), str(np.sum(found)), str(n_swapped), str(max_path_prob)]
        for method_name in method_names:
            weights = method_to_weights[method_name][gene_id][1]
            # Genetically predicted expression in gtex samples
            pred_expr = np.dot(weights[found], standardized_geno_mat)
            if np.sum(found) == 0 or np.std(pred_expr) == 0.0 or np.std(expr_vec) == 0.0:
                corr = np.nan
            else:
                corr = np.corrcoef(pred_expr, expr_vec)[0, 1]
            if np.sum(np.abs(weights)) > 0.0:
                frac_weight = np.sum(np.abs(weights[found]))/np.sum(np.abs(weights))
            else:
                frac_weight = np.nan
            method_to_corrs[method_name].append(corr)
            line.append(str(corr))
            line.append(str(frac_weight))
        # Cis-SNP heritability of gtex expression (and LRT p-value) using the gene's variants found in gtex. Does not depend on any of the prediction methods
        if np.sum(found) > 0:
            cis_snp_h2, cis_snp_h2_pvalue = estimate_cis_snp_heritability_with_lrt(np.transpose(standardized_geno_mat), expr_vec)
        else:
            cis_snp_h2, cis_snp_h2_pvalue = np.nan, np.nan
        line.append(str(cis_snp_h2))
        line.append(str(cis_snp_h2_pvalue))
        t.write('\t'.join(line) + '\n')
    t.flush()

t.close()
print(str(n_genes_skipped) + ' genes skipped (methods do not share variants or chromosome mismatch)')

# Print summary of prediction accuracy (all genes, and stratified by whether the gene has a variant with a chromatin path)
max_path_probs = np.asarray(max_path_probs)
print_summary(method_names, method_to_corrs, np.ones(len(max_path_probs), dtype=bool), 'All genes')
# Bins of max_chromatin_path_prob_in_gtex (largest probability of a chromatin path among the gene's variants found in gtex)
bin_edges = [0.0, 0.1, 0.5, 0.9, 1.0]
for bin_iter in range(len(bin_edges) - 1):
    if bin_iter == len(bin_edges) - 2:
        gene_subset = (max_path_probs >= bin_edges[bin_iter]) & (max_path_probs <= bin_edges[bin_iter + 1])
    else:
        gene_subset = (max_path_probs >= bin_edges[bin_iter]) & (max_path_probs < bin_edges[bin_iter + 1])
    print_summary(method_names, method_to_corrs, gene_subset, 'Genes with max_chromatin_path_prob_in_gtex in [' + str(bin_edges[bin_iter]) + ', ' + str(bin_edges[bin_iter + 1]) + ']')
