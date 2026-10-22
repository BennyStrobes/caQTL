import sys
import argparse
import numpy as np

# Evaluate CoFine fits (run_cofine.py) on data from simulate_cofine_data.py: calibration and power of eQTL PIPs (all variants, and
# split by whether the variant truly has a chromatin path), calibration of P(H_j = 1), and recovery of the hyper-parameters.
# Several results stems can be given (e.g. CoFine and CoFine with --no_chromatin) to compare them on the same simulated data.


def load_truth(truth_summary_file):
    gene_to_truth = {}
    f = open(truth_summary_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            continue
        truth = np.load(data[1])
        gene_to_truth[data[0]] = (truth['beta'] != 0.0, truth['H'])
    f.close()
    return gene_to_truth


def load_results(results_file):
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
        gene_to_results[data[0]] = (pips, path_probs)
    f.close()
    return gene_to_results


def load_two_column_file(file_name):
    out = {}
    f = open(file_name)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            continue
        out[data[0]] = data[1:]
    f.close()
    return out


def print_calibration(probs, truth, label, bin_edges=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)):
    print('    ' + label + ' calibration (bin: n, mean predicted, fraction true)')
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (probs >= lo) & (probs < hi)
        if np.sum(in_bin) == 0:
            continue
        print('      [%.1f, %.1f): %d, %.3f, %.3f' % (lo, min(hi, 1.0), np.sum(in_bin), np.mean(probs[in_bin]), np.mean(truth[in_bin])))


def print_power(pips, causal, label):
    for threshold in [0.5, 0.9]:
        called = pips > threshold
        n_true = np.sum(called & causal)
        fdp = 1.0 - n_true / np.sum(called) if np.sum(called) > 0 else np.nan
        print('    ' + label + ' PIP > %.1f: %d called, %d true, power %.3f, false discovery proportion %.3f' % (threshold, np.sum(called), n_true, n_true / max(np.sum(causal), 1), fdp))


parser = argparse.ArgumentParser()
parser.add_argument('--simulation_stem', type=str)  # output_dir/output_stem of simulate_cofine_data.py
parser.add_argument('--results_stems', type=str, nargs='+')  # output stems of run_cofine.py
parser.add_argument('--labels', type=str, nargs='+')
args = parser.parse_args()

gene_to_truth = load_truth(args.simulation_stem + '_truth_summary.txt')
true_hyperparameters = load_two_column_file(args.simulation_stem + '_true_hyperparameters.txt')
for results_stem, label in zip(args.results_stems, args.labels):
    gene_to_results = load_results(results_stem + '_results.txt')
    gene_ids = [gene_id for gene_id in gene_to_results if gene_id in gene_to_truth]
    pips = np.concatenate([gene_to_results[gene_id][0] for gene_id in gene_ids])
    path_probs = np.concatenate([gene_to_results[gene_id][1] for gene_id in gene_ids])
    causal = np.concatenate([gene_to_truth[gene_id][0] for gene_id in gene_ids])
    on_path = np.concatenate([gene_to_truth[gene_id][1] for gene_id in gene_ids])
    print(label + ' (' + str(len(gene_ids)) + ' genes, ' + str(np.sum(causal)) + ' causal eQTLs, ' + str(np.sum(causal & on_path)) + ' on a chromatin path)')
    print_calibration(pips, causal, 'eQTL PIP')
    print_power(pips, causal, 'all variants:')
    print_power(pips[on_path], causal[on_path], 'variants on a true path:')
    print_power(pips[~on_path], causal[~on_path], 'variants off a true path:')
    print('    expected vs true number of causal eQTLs: %.1f vs %d' % (np.sum(pips), np.sum(causal)))
    if np.sum(path_probs) > 0:
        print_calibration(path_probs, on_path, 'P(H = 1)')
    hyperparameters = load_two_column_file(results_stem + '_hyperparameter_summary.txt')
    print('    hyperparameters (true; posterior mean [95% interval])')
    for name in hyperparameters:
        print('      %s: %s; %.4g [%.4g, %.4g]' % (name, true_hyperparameters.get(name, ['NA'])[0], float(hyperparameters[name][0]), float(hyperparameters[name][1]), float(hyperparameters[name][2])))
