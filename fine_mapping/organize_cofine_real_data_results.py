import argparse
import numpy as np

# Organize the real-data CoFine fits (run_cofine.sh: CoFine and CoFine without caQTL and peak-gene link data) into small tables for
# visualize_cofine_real_data_results.R:
#   <output_stem>_genes.txt: per gene, the expected number of causal eQTLs (sum of PIPs) and the number of variants with PIP > 0.5
#       and > 0.9 under each fit, and CoFine's expected number of variants with a chromatin path (sum of P(H = 1))
#   <output_stem>_variants.txt: variants with PIP >= min_pip in either fit: both PIPs, CoFine's P(H = 1), and the marginal eQTL z-score
#   <output_stem>_hyperparameter_samples.txt: CoFine's retained (post burn-in) hyper-parameter samples, one row per sweep and hyper-parameter
hyperparameter_names = ['pi0', 'sigma2_0', 'pi1', 'lambda', 'tau2', 'pi_delta', 'sigma2_delta', 'pi_gamma', 'sigma2_gamma']


def load_results(results_file):
    # gene id -> (variant ids file, eQTL PIPs, P(H = 1))
    gene_to_results = {}
    f = open(results_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            header = data
            continue
        pips = np.asarray(data[header.index('eqtl_pips')].split(',')).astype(float)
        path_probs = np.asarray(data[header.index('path_probs')].split(',')).astype(float)
        gene_to_results[data[0]] = (data[header.index('variant_ids_file')], pips, path_probs)
    f.close()
    return gene_to_results


def load_eqtl_z_scores(sumstat_summary_file):
    # gene id -> marginal eQTL z-scores (fine-mapping input summary of prepare_input_data_for_fine_mapping.py)
    gene_to_z = {}
    f = open(sumstat_summary_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            continue
        gene_to_z[data[0]] = np.load(data[3]) / np.load(data[4])
    f.close()
    return gene_to_z


parser = argparse.ArgumentParser()
parser.add_argument('--sumstat_summary_file', type=str)  # the input of run_cofine.py
parser.add_argument('--cofine_stem', type=str)  # output stem of run_cofine.py
parser.add_argument('--no_chromatin_stem', type=str)  # output stem of run_cofine.py --no_chromatin
parser.add_argument('--output_stem', type=str)
parser.add_argument('--min_pip', type=float, default=0.1)
args = parser.parse_args()

cofine = load_results(args.cofine_stem + '_results.txt')
no_chromatin = load_results(args.no_chromatin_stem + '_results.txt')
gene_to_z = load_eqtl_z_scores(args.sumstat_summary_file)
gene_ids = [gene_id for gene_id in cofine if gene_id in no_chromatin and gene_id in gene_to_z]

t_gene = open(args.output_stem + '_genes.txt', 'w')
t_gene.write('gene_id\tn_variants\tsum_pip_cofine\tsum_pip_no_chromatin\tn_pip_0.5_cofine\tn_pip_0.5_no_chromatin\tn_pip_0.9_cofine\tn_pip_0.9_no_chromatin\tsum_path_prob\n')
t_var = open(args.output_stem + '_variants.txt', 'w')
t_var.write('gene_id\tvariant_id\tpip_cofine\tpip_no_chromatin\tpath_prob\teqtl_z\n')
for gene_id in gene_ids:
    variant_ids_file, pip_c, path_prob = cofine[gene_id]
    pip_n = no_chromatin[gene_id][1]
    z = gene_to_z[gene_id]
    if not (len(pip_c) == len(pip_n) == len(z)):
        print('gene ' + gene_id + ': fits and input have different numbers of variants, skipped')
        continue
    variant_ids = np.loadtxt(variant_ids_file, dtype=str, ndmin=1)
    if len(variant_ids) != len(z):
        variant_ids = np.asarray([gene_id + '_variant_' + str(j) for j in range(len(z))])
    t_gene.write('\t'.join([gene_id, str(len(z)), str(np.sum(pip_c)), str(np.sum(pip_n)), str(int(np.sum(pip_c > 0.5))), str(int(np.sum(pip_n > 0.5))), str(int(np.sum(pip_c > 0.9))), str(int(np.sum(pip_n > 0.9))), str(np.sum(path_prob))]) + '\n')
    for j in np.flatnonzero(np.maximum(pip_c, pip_n) >= args.min_pip):
        t_var.write('\t'.join([gene_id, variant_ids[j], str(pip_c[j]), str(pip_n[j]), str(path_prob[j]), str(z[j])]) + '\n')
t_gene.close()
t_var.close()

samples = np.load(args.cofine_stem + '_hyperparameter_samples.npy')
t_hyp = open(args.output_stem + '_hyperparameter_samples.txt', 'w')
t_hyp.write('sweep\thyperparameter\tvalue\n')
for sweep in range(samples.shape[0]):
    for index, name in enumerate(hyperparameter_names):
        t_hyp.write(str(sweep) + '\t' + name + '\t' + str(samples[sweep, index]) + '\n')
t_hyp.close()
print(str(len(gene_ids)) + ' genes organized into ' + args.output_stem + '_*.txt')
