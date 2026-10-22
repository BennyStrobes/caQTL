import os
import glob
import argparse
import numpy as np

# Organize CoFine simulation results (run_cofine_simulation.sh) across seeds into three small tables for
# visualize_cofine_simulation_results.R. Only seeds with a truth file and results of both fits are used.
#   <output_stem>_calibration.txt: per seed, method, quantity (eqtl_pip, or path_prob for CoFine) and bin:
#       number of variants, sum of predicted probabilities, number truly causal (eqtl_pip) or truly on a path (path_prob)
#   <output_stem>_power.txt: per seed, method, stratum (all variants / on a true path / off a true path) and PIP threshold:
#       number of causal variants, number called, number of true calls
#   <output_stem>_hyperparameters.txt: per seed, CoFine hyper-parameter posterior mean and 95% interval, and the true value

methods = [('cofine', 'CoFine'), ('cofine_no_chromatin', 'CoFine without chromatin')]
bin_edges = [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001]
pip_thresholds = [0.5, 0.9]


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


parser = argparse.ArgumentParser()
parser.add_argument('--simulation_dir', type=str)  # cofine_simulation_dir of run_cofine_simulation.sh
parser.add_argument('--simulation_stem_prefix', type=str)  # e.g. CD4_T_cofine_sim_n_genes_500 (followed by _seed_<seed>)
parser.add_argument('--output_stem', type=str)
args = parser.parse_args()

t_cal = open(args.output_stem + '_calibration.txt', 'w')
t_cal.write('seed\tmethod\tquantity\tbin_lower\tbin_upper\tn_variants\tsum_predicted\tn_true\n')
t_pow = open(args.output_stem + '_power.txt', 'w')
t_pow.write('seed\tmethod\tstratum\tpip_threshold\tn_causal\tn_called\tn_true_called\n')
t_hyp = open(args.output_stem + '_hyperparameters.txt', 'w')
t_hyp.write('seed\thyperparameter\ttrue_value\tposterior_mean\tposterior_95_lb\tposterior_95_ub\n')

truth_files = sorted(glob.glob(os.path.join(args.simulation_dir, args.simulation_stem_prefix + '_seed_*_truth_summary.txt')))
n_seeds_used = 0
for truth_summary_file in truth_files:
    simulation_stem = truth_summary_file[:-len('_truth_summary.txt')]
    seed = simulation_stem.split('_seed_')[-1]
    if not all(os.path.exists(simulation_stem + '_' + method + '_results.txt') and os.path.exists(simulation_stem + '_' + method + '_hyperparameter_summary.txt') for method, label in methods):
        print('seed ' + seed + ': results missing, skipped')
        continue
    n_seeds_used += 1

    # Truth: causal eQTL and true chromatin path of every variant
    gene_to_truth = {}
    for gene_id, values in load_two_column_file(truth_summary_file).items():
        truth = np.load(values[0])
        gene_to_truth[gene_id] = (truth['beta'] != 0.0, truth['H'])

    for method, label in methods:
        gene_to_results = load_results(simulation_stem + '_' + method + '_results.txt')
        gene_ids = [gene_id for gene_id in gene_to_results if gene_id in gene_to_truth]
        pips = np.concatenate([gene_to_results[gene_id][0] for gene_id in gene_ids])
        path_probs = np.concatenate([gene_to_results[gene_id][1] for gene_id in gene_ids])
        causal = np.concatenate([gene_to_truth[gene_id][0] for gene_id in gene_ids])
        on_path = np.concatenate([gene_to_truth[gene_id][1] for gene_id in gene_ids])

        quantities = [('eqtl_pip', pips, causal)]
        if method == 'cofine':
            quantities.append(('path_prob', path_probs, on_path))
        for quantity, probs, truth in quantities:
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                in_bin = (probs >= lo) & (probs < hi)
                t_cal.write('\t'.join([seed, label, quantity, str(lo), str(min(hi, 1.0)), str(int(np.sum(in_bin))), str(np.sum(probs[in_bin])), str(int(np.sum(truth[in_bin])))]) + '\n')

        for stratum, subset in [('All variants', np.ones(len(pips), dtype=bool)), ('On a true chromatin path', on_path), ('Off a true chromatin path', ~on_path)]:
            for threshold in pip_thresholds:
                called = (pips > threshold) & subset
                t_pow.write('\t'.join([seed, label, stratum, str(threshold), str(int(np.sum(causal & subset))), str(int(np.sum(called))), str(int(np.sum(called & causal)))]) + '\n')

    true_hyperparameters = load_two_column_file(simulation_stem + '_true_hyperparameters.txt')
    for name, values in load_two_column_file(simulation_stem + '_cofine_hyperparameter_summary.txt').items():
        t_hyp.write('\t'.join([seed, name, true_hyperparameters[name][0]] + values) + '\n')

t_cal.close()
t_pow.close()
t_hyp.close()
print(str(n_seeds_used) + ' of ' + str(len(truth_files)) + ' seeds organized into ' + args.output_stem + '_*.txt')
