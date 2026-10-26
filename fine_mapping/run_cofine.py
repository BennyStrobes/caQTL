import sys
import os
import argparse
import pdb
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cofine import COFINE, prepare_gene_data

# CoFine (grant Aim 3B): eQTL fine-mapping with chromatin-mediated directional priors, fit genome-wide by tempered Gibbs (see cofine.py)
# Input: the fine-mapping input summary of prepare_input_data_for_fine_mapping.py (one line per gene: LD, eQTL, caQTL and peak-gene files)
# --no_chromatin drops the caQTL and peak-gene data (every gene gets no peaks): spike-and-slab eQTL fine-mapping with the same sampler,
# the grant's "CoFine without caQTL and peak-gene link data" comparison
# --learn_link_se_inflation learns one shared inflation of the squared peak-gene link standard errors (for anti-conservative links)


def load_in_input_data(sumstat_summary_file, no_chromatin, max_genes, variance_explained_threshold):
    # Returns ordered gene ids, their variant ids files, the per-gene inputs of COFINE, and per gene the number of input peaks and the
    # input indices of the peaks passed to COFINE
    # Peaks whose link has a zero or non-finite standard error (degenerate hurdle fits) are dropped: N(gamma_k, 0) is not a likelihood
    gene_ids = []
    variant_ids_files = []
    genes = []
    n_input_peaks = []
    kept_peak_indices = []
    n_links_dropped = 0
    f = open(sumstat_summary_file, 'r')
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            continue
        gene_id = data[0]
        ld_reg = np.load(data[2])
        eqtl_effects = np.load(data[3])
        eqtl_se = np.load(data[4])
        p = len(eqtl_effects)
        caqtl_effects = np.load(data[5]).reshape(-1, p)
        caqtl_se = np.load(data[6]).reshape(-1, p)
        peak_gene_effects = np.atleast_1d(np.load(data[7]))
        peak_gene_se = np.atleast_1d(np.load(data[8]))
        keep = np.isfinite(peak_gene_effects) & np.isfinite(peak_gene_se) & (peak_gene_se > 0.0)
        n_links_dropped += int(np.sum(~keep))
        n_input_peaks.append(len(keep) if not no_chromatin else 0)
        kept_peak_indices.append(np.flatnonzero(keep) if not no_chromatin else np.zeros(0, dtype=int))
        caqtl_effects = caqtl_effects[keep]
        caqtl_se = caqtl_se[keep]
        peak_gene_effects = peak_gene_effects[keep]
        peak_gene_se = peak_gene_se[keep]
        if no_chromatin:
            caqtl_effects = caqtl_effects[:0]
            caqtl_se = caqtl_se[:0]
            peak_gene_effects = peak_gene_effects[:0]
            peak_gene_se = peak_gene_se[:0]
        # Undo the LD regularization of the preparation step, R_reg = (1 - s) R + s I (the low-rank transform replaces it)
        ld_s = float(data[12])
        ld = (ld_reg - ld_s * np.eye(p)) / (1.0 - ld_s) if ld_s > 0.0 else ld_reg
        np.fill_diagonal(ld, 1.0)
        genes.append(prepare_gene_data(ld, eqtl_effects, eqtl_se, caqtl_effects, caqtl_se, peak_gene_effects, peak_gene_se, variance_explained_threshold))
        gene_ids.append(gene_id)
        variant_ids_files.append(data[1])
        if max_genes is not None and len(gene_ids) >= max_genes:
            break
        if len(gene_ids) % 500 == 0:
            print('loaded gene ' + str(len(gene_ids)), flush=True)
    f.close()
    print(str(n_links_dropped) + ' peak-gene links dropped for a zero or non-finite link standard error', flush=True)
    return gene_ids, variant_ids_files, genes, n_input_peaks, kept_peak_indices


def array_to_string(arr):
    if len(arr) == 0:
        return 'NA'
    return ','.join(['%.6g' % x for x in arr])


##########################
# Command line arguments
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--sumstat_summary_file', type=str)  # one line per gene (fine-mapping input summary from prepare_input_data_for_fine_mapping.py)
parser.add_argument('--output_stem', type=str)
parser.add_argument('--n_iters', type=int, default=1000)
parser.add_argument('--burn_in', type=int, default=400)
parser.add_argument('--no_tgs', action='store_true', default=False)  # random-scan Gibbs instead of tempered Gibbs
parser.add_argument('--tgs_epsilon', type=float, default=0.5)
parser.add_argument('--tgs_sub_steps_per_snp', type=float, default=1.0)
parser.add_argument('--tgs_r2_update_threshold', type=float, default=0.01)
parser.add_argument('--swap_r2', type=float, default=0.5)  # path swaps between variants with r2 >= swap_r2
parser.add_argument('--swaps_per_path', type=float, default=2.0)  # 0: no path swaps
parser.add_argument('--variance_explained_threshold', type=float, default=0.995)  # fraction of the LD trace kept by the low-rank transform
parser.add_argument('--no_chromatin', action='store_true', default=False)  # drop caQTL and peak-gene data (eQTL-only comparison)
parser.add_argument('--learn_link_se_inflation', action='store_true', default=False)  # learn a shared inflation c2 of the squared peak-gene link standard errors (anti-conservative links)
parser.add_argument('--link_se_inflation_prior_df', type=float, default=4.0)  # degrees of freedom of the scaled-inverse-chi-square(df, 1) prior on c2
parser.add_argument('--max_genes', type=int, default=None)  # only the first max_genes genes (for testing)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

print('settings: ' + ' '.join(key + '=' + str(value) for key, value in vars(args).items()), flush=True)

# Load in data
gene_ids, variant_ids_files, genes, n_input_peaks, kept_peak_indices = load_in_input_data(args.sumstat_summary_file, args.no_chromatin, args.max_genes, args.variance_explained_threshold)

# Fit
model = COFINE(n_iters=args.n_iters, burn_in=args.burn_in, tgs=not args.no_tgs, tgs_epsilon=args.tgs_epsilon, tgs_sub_steps_per_snp=args.tgs_sub_steps_per_snp, tgs_r2_update_threshold=args.tgs_r2_update_threshold, swap_r2=args.swap_r2, swaps_per_path=args.swaps_per_path, learn_link_se_inflation=args.learn_link_se_inflation, link_se_inflation_prior_df=args.link_se_inflation_prior_df, seed=args.seed)
model.fit(genes)

# Per-gene results: values per variant (in variant ids file order) or per peak (in peak order of the input), comma separated
#   eqtl_pips: P(beta_j != 0); eqtl_posterior_mean: E[beta_j] (standardized genotype, per gene standard error scale of the input)
#   path_probs: P(H_j = 1), probability of a chromatin-mediated path; chromatin_effect_posterior_mean: E[m_j]
#   link_probs: P(gamma_k != 0) for each input peak (nan for peaks dropped for a zero or non-finite link standard error)
t = open(args.output_stem + '_results.txt', 'w')
t.write('gene_id\tvariant_ids_file\tn_peaks\teqtl_pips\teqtl_posterior_mean\tpath_probs\tchromatin_effect_posterior_mean\tlink_probs\n')
for gene_id, variant_ids_file, st, n_peaks, kept in zip(gene_ids, variant_ids_files, model.state, n_input_peaks, kept_peak_indices):
    link_prob = np.full(n_peaks, np.nan)
    link_prob[kept] = st['link_prob']
    t.write(gene_id + '\t' + variant_ids_file + '\t' + str(n_peaks) + '\t' + array_to_string(st['eqtl_pip']) + '\t' + array_to_string(st['eqtl_posterior_mean']) + '\t' + array_to_string(st['path_prob']) + '\t' + array_to_string(st['chromatin_effect_posterior_mean']) + '\t' + array_to_string(link_prob) + '\n')
t.close()

# Hyper-parameters: posterior mean and 95% interval over the retained sweeps, and the retained samples themselves
samples = model.sampled_hyperparameters
t = open(args.output_stem + '_hyperparameter_summary.txt', 'w')
t.write('hyperparameter\tposterior_mean\tposterior_95_lb\tposterior_95_ub\n')
for index, name in enumerate(model.hyperparameter_names):
    t.write(name + '\t' + str(np.mean(samples[:, index])) + '\t' + str(np.quantile(samples[:, index], 0.025)) + '\t' + str(np.quantile(samples[:, index], 0.975)) + '\n')
t.close()
np.save(args.output_stem + '_hyperparameter_samples.npy', samples)
print(str(len(gene_ids)) + ' genes written to ' + args.output_stem + '_results.txt')
