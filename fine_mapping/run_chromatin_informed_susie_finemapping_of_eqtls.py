import sys
import os
import argparse
import pdb
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from susie_rss import SUSIE_RSS

# Chromatin-informed SuSiE fine-mapping of eQTLs (simple version)
# Standard SuSiE on the eQTL data, where the prior weight of variant j is proportional to 1 + (enrichment - 1)*path_prob_j
# path_prob_j: probability variant j has a chromatin path to the gene (caQTL variant for a peak with a peak-gene link; from run_chromatin_only_expression_prediction.py)
# Sign gate: a variant's path is only counted if the chromatin-predicted direction of effect (peak-gene effect x caQTL effect) agrees with the sign of the variant's marginal eQTL z-score
# enrichment is shared across genes. It is either fixed (--enrichment) or learned across genes by EM
# With enrichment = 1 this is exactly run_standard_susie_finemapping_of_eqtls.py


def load_in_chromatin_only_results(chromatin_only_results_file):
    # Results file from run_chromatin_only_expression_prediction.py
    # Returns dictionary: gene_id -> (variant_ids_file, probability of chromatin path, chromatin-predicted eQTL effect)
    gene_to_chromatin = {}
    f = open(chromatin_only_results_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            path_prob_col = data.index('eqtl_alpha')
            pred_eqtl_col = data.index('eqtl_posterior_mean')
            continue
        path_probs = np.asarray(data[path_prob_col].split(',')).astype(float)
        pred_eqtl = np.asarray(data[pred_eqtl_col].split(',')).astype(float)
        gene_to_chromatin[data[0]] = (data[1], path_probs, pred_eqtl)
    f.close()
    return gene_to_chromatin


def load_in_input_data(sumstat_summary_file, gene_to_chromatin, min_path_prob, use_sign_gate):
    # Returns ordered gene ids and dictionary: gene_id -> dict with keys variant_ids_file, ld_file, eqtl_effects, eqtl_se, path_probs
    gene_ids = []
    gene_to_data = {}
    f = open(sumstat_summary_file, 'r')
    head_count = 0
    for line in f:
        data = line.rstrip().split('\t')
        if head_count == 0:
            head_count += 1
            continue
        gene_id = data[0]
        variant_ids_file = data[1]
        eqtl_effects = np.load(data[3])
        eqtl_se = np.load(data[4])

        # Probability of chromatin path for each variant (zero if gene has no chromatin results)
        path_probs = np.zeros(len(eqtl_effects))
        if gene_id in gene_to_chromatin and gene_to_chromatin[gene_id][0] == variant_ids_file and len(gene_to_chromatin[gene_id][1]) == len(eqtl_effects):
            path_probs = np.copy(gene_to_chromatin[gene_id][1])
            pred_eqtl = gene_to_chromatin[gene_id][2]
            # Variants with small probability of chromatin path are treated as having no chromatin path
            path_probs[path_probs < min_path_prob] = 0.0
            # Sign gate: path only counts if chromatin-predicted direction agrees with direction of marginal eQTL effect
            if use_sign_gate:
                path_probs[np.sign(pred_eqtl) != np.sign(eqtl_effects)] = 0.0
            path_probs = np.minimum(path_probs, 1.0)

        gene_ids.append(gene_id)
        gene_to_data[gene_id] = {'variant_ids_file': variant_ids_file, 'ld_file': data[2], 'eqtl_effects': eqtl_effects, 'eqtl_se': eqtl_se, 'path_probs': path_probs}
    f.close()
    return np.asarray(gene_ids), gene_to_data


def fit_gene(gene_data, enrichment, L, eqtl_sample_size, estimate_residual_variance, prior_tol=1e-9):
    ld_matrix = np.load(gene_data['ld_file'])
    path_probs = gene_data['path_probs']
    prior_weights = 1.0 + (enrichment - 1.0)*path_probs
    susie_fit = SUSIE_RSS(L=L, zscore_residual_variance=estimate_residual_variance).fit(gene_data['eqtl_effects'], gene_data['eqtl_se'], ld_matrix, eqtl_sample_size, prior_weights=prior_weights)
    # Keep only what is needed (the susie object holds the LD matrix, which is too large to keep for every gene)
    in_use = susie_fit.V > prior_tol  # single effects with prior variance of zero are not in use
    has_path = path_probs > 0.0
    fit = {}
    fit['pip'] = susie_fit.pip
    fit['credible_sets'] = susie_fit.credible_sets
    fit['converged'] = susie_fit.converged
    fit['sigma2'] = susie_fit.sigma2
    fit['posterior_mean'] = susie_fit.posterior_mean
    fit['n_effects_in_use'] = np.sum(in_use)
    fit['alpha_at_path_variants'] = np.sum(susie_fit.alpha[in_use, :][:, has_path], axis=0)  # summed over single effects in use
    return fit


def update_enrichment(gene_ids_with_path, gene_to_data, gene_to_fit, max_enrichment):
    # Maximize expected log prior of the location of each single effect (effects with prior variance of zero are not in use) over enrichment
    # sum_genes [sum_effects sum_j alpha_lj*log(1 + (e-1)*path_prob_j) - n_effects*log(sum_i (1 + (e-1)*path_prob_i))]
    alphas = []
    rhos = []
    n_effects = []
    n_variants = []
    rho_sums = []
    for gene_id in gene_ids_with_path:
        fit = gene_to_fit[gene_id]
        path_probs = gene_to_data[gene_id]['path_probs']
        if fit['n_effects_in_use'] == 0:
            continue
        alphas.append(fit['alpha_at_path_variants'])
        rhos.append(path_probs[path_probs > 0.0])
        n_effects.append(fit['n_effects_in_use'])
        n_variants.append(len(path_probs))
        rho_sums.append(np.sum(path_probs))
    if len(alphas) == 0:
        return 1.0
    alphas = np.hstack(alphas)
    rhos = np.hstack(rhos)
    n_effects = np.asarray(n_effects)
    n_variants = np.asarray(n_variants)
    rho_sums = np.asarray(rho_sums)

    def objective(log_e):
        e = np.exp(log_e)
        return np.sum(alphas*np.log(1.0 + (e - 1.0)*rhos)) - np.sum(n_effects*np.log(n_variants + (e - 1.0)*rho_sums))

    # Golden section search over log(enrichment)
    lo = np.log(1e-2)
    hi = np.log(max_enrichment)
    golden = (np.sqrt(5.0) - 1.0)/2.0
    x1 = hi - golden*(hi - lo)
    x2 = lo + golden*(hi - lo)
    f1 = objective(x1)
    f2 = objective(x2)
    for step in range(60):
        if f1 < f2:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + golden*(hi - lo)
            f2 = objective(x2)
        else:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - golden*(hi - lo)
            f1 = objective(x1)
    return np.exp(0.5*(lo + hi))


##########################
# Command line arguments
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--sumstat_summary_file', type=str)  # one line per gene (fine-mapping input summary from prepare_input_data_for_fine_mapping.py)
parser.add_argument('--chromatin_only_results_file', type=str)  # output of run_chromatin_only_expression_prediction.py
parser.add_argument('--fine_mapping_output_file', type=str)
parser.add_argument('--eqtl_sample_size', type=int, default=None)  # None: z-score model (X'X = R, X'y = z, residual variance 1)
parser.add_argument('--L', type=int, default=10)
parser.add_argument('--estimate_residual_variance', action='store_true', default=False)  # same meaning as in run_standard_susie_finemapping_of_eqtls.py
parser.add_argument('--enrichment', type=float, default=None)  # None: learn enrichment across genes. Otherwise fix it at this value
parser.add_argument('--min_path_prob', type=float, default=0.01)  # Variants with probability of chromatin path less than this are treated as having no chromatin path
parser.add_argument('--no_sign_gate', action='store_true', default=False)  # If set, a variant's path counts whether or not the chromatin-predicted direction agrees with the marginal eQTL direction
parser.add_argument('--max_enrichment', type=float, default=30.0)  # Upper bound on learned enrichment. The EM is self-reinforcing (a larger enrichment concentrates posteriors on path variants, which raises the enrichment), and PIPs become over-confident when enrichment is large and path probabilities are over-stated
parser.add_argument('--max_em_rounds', type=int, default=10)
args = parser.parse_args()

sumstat_summary_file = args.sumstat_summary_file
chromatin_only_results_file = args.chromatin_only_results_file
fine_mapping_output_file = args.fine_mapping_output_file
eqtl_sample_size = args.eqtl_sample_size
L = args.L
use_sign_gate = args.no_sign_gate == False
print('settings: zscore_mode ' + str(eqtl_sample_size is None) + ', zscore_residual_variance ' + str(args.estimate_residual_variance) + ', L ' + str(L) + ', sign_gate ' + str(use_sign_gate) + ', min_path_prob ' + str(args.min_path_prob), flush=True)

# Load in data
gene_to_chromatin = load_in_chromatin_only_results(chromatin_only_results_file)
gene_ids, gene_to_data = load_in_input_data(sumstat_summary_file, gene_to_chromatin, args.min_path_prob, use_sign_gate)
gene_ids_with_path = np.asarray([gene_id for gene_id in gene_ids if np.sum(gene_to_data[gene_id]['path_probs']) > 0.0])
print(str(len(gene_ids_with_path)) + ' of ' + str(len(gene_ids)) + ' genes have a variant with a chromatin path', flush=True)

# Fit genes
gene_to_fit = {}
if args.enrichment is not None:
    # Enrichment is fixed
    enrichment = args.enrichment
    for gene_id in gene_ids:
        gene_to_fit[gene_id] = fit_gene(gene_to_data[gene_id], enrichment, L, eqtl_sample_size, args.estimate_residual_variance)
else:
    # Learn enrichment across genes by EM. Start from standard SuSiE (enrichment of 1)
    # Genes with no chromatin path have uniform prior weights whatever the enrichment is. So they are fit only once
    enrichment = 1.0
    for gene_id in gene_ids:
        gene_to_fit[gene_id] = fit_gene(gene_to_data[gene_id], enrichment, L, eqtl_sample_size, args.estimate_residual_variance)
    for em_round in range(args.max_em_rounds):
        new_enrichment = update_enrichment(gene_ids_with_path, gene_to_data, gene_to_fit, args.max_enrichment)
        print('EM round ' + str(em_round) + ': enrichment=' + str(new_enrichment), flush=True)
        converged = np.abs(np.log(new_enrichment) - np.log(enrichment)) < 0.01
        enrichment = new_enrichment
        for gene_id in gene_ids_with_path:
            gene_to_fit[gene_id] = fit_gene(gene_to_data[gene_id], enrichment, L, eqtl_sample_size, args.estimate_residual_variance)
        if converged:
            break
print('enrichment=' + str(enrichment), flush=True)
if args.enrichment is None and enrichment > 0.99*args.max_enrichment:
    print('note: learned enrichment is at its upper bound (--max_enrichment)', flush=True)

# Write results (same columns as run_standard_susie_finemapping_of_eqtls.py, plus two columns at the end)
t = open(fine_mapping_output_file, 'w')
t.write('gene_id\tvariant_ids_file\tpips\tcredible_sets\tconverged\tresidual_variance\teqtl_posterior_mean_file\tenrichment\tgated_path_probs\n')
output_stem = os.path.splitext(fine_mapping_output_file)[0]
for gene_id in gene_ids:
    fit = gene_to_fit[gene_id]
    # Credible sets: variant indices (0-based) per set, sets separated by ';'
    credible_sets_string = ';'.join([','.join([str(x) for x in cs['snp_indices']]) for cs in fit['credible_sets']])
    if credible_sets_string == '':
        credible_sets_string = 'NA'
    eqtl_posterior_mean_file = output_stem + '_gene_' + gene_id + '_eqtl_posterior_mean.npy'
    np.save(eqtl_posterior_mean_file, fit['posterior_mean'])
    t.write(gene_id + '\t' + gene_to_data[gene_id]['variant_ids_file'] + '\t' + ','.join([str(x) for x in fit['pip']]) + '\t' + credible_sets_string + '\t' + str(fit['converged']) + '\t' + str(fit['sigma2']) + '\t' + eqtl_posterior_mean_file + '\t' + str(enrichment) + '\t' + ','.join(['%.6g' % x for x in gene_to_data[gene_id]['path_probs']]) + '\n')
t.close()
print(str(len(gene_ids)) + ' genes written to ' + fine_mapping_output_file)
