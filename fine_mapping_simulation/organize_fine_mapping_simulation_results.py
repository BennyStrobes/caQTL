import numpy as np
import os
import sys
import pdb
import argparse


def load_pips(fine_mapping_results_file):
    # gene_id -> vector of PIPs, from a fine-mapping results file (gene_id, variant_ids_file, comma-separated pips)
    gene_to_pips = {}
    f = open(fine_mapping_results_file, 'r')
    head_count = 0
    for line in f:
        line = line.rstrip()
        data = line.split('\t')
        if head_count == 0:
            head_count += 1
            continue
        gene_to_pips[data[0]] = np.asarray(data[2].split(','), dtype=float)
    f.close()
    return gene_to_pips


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation_fine_mapping_results_dir', type=str, required=True)
    parser.add_argument('--sumstats_output_dir', type=str, required=True)
    parser.add_argument('--n_simulations', type=int, required=True)
    parser.add_argument('--eqtl_sample_size', type=int, required=True)
    parser.add_argument('--output_file', type=str, required=True, help='one line per simulation-gene-variant with truth and the PIPs of each method')
    return parser.parse_args()


########################
# Command line args
########################
args = parse_args()
simulation_fine_mapping_results_dir = args.simulation_fine_mapping_results_dir
sumstats_output_dir = args.sumstats_output_dir
n_simulations = args.n_simulations
eqtl_sample_size = args.eqtl_sample_size
output_file = args.output_file


t = open(output_file, 'w')
t.write('simulation_number\tgene_id\tvariant_index\tis_constrained\tcausal_class\teqtl_only_pip\tca_qtl_mediated_pip\n')

n_simulations_found = 0
for simulation_number in range(1, n_simulations + 1):
    simulation_stem = 'simulation_' + str(simulation_number) + '_eqtl_sample_size_' + str(eqtl_sample_size)
    eqtl_only_results_file = simulation_fine_mapping_results_dir + simulation_stem + '_eqtl_only_finemapping_results.txt'
    ca_qtl_mediated_results_file = simulation_fine_mapping_results_dir + simulation_stem + '_ca_qtl_mediated_eqtl_finemapping_results.txt'
    causal_effect_summary_file = sumstats_output_dir + simulation_stem + '_simulated_causal_effect_summary.txt'
    if not (os.path.exists(eqtl_only_results_file) and os.path.exists(ca_qtl_mediated_results_file) and os.path.exists(causal_effect_summary_file)):
        print('skipping simulation ' + str(simulation_number) + ': results not found')
        continue
    n_simulations_found += 1

    eqtl_only_pips = load_pips(eqtl_only_results_file)
    ca_qtl_mediated_pips = load_pips(ca_qtl_mediated_results_file)

    # Loop through genes in the simulated causal effect summary
    f = open(causal_effect_summary_file, 'r')
    head_count = 0
    for line in f:
        line = line.rstrip()
        data = line.split('\t')
        if head_count == 0:
            head_count += 1
            continue
        gene_id = data[0]
        is_constrained = data[1]
        causal_eqtl_effects = np.load(data[2])          # p
        causal_caqtl_effects = np.load(data[3])         # K x p
        causal_peak_gene_effects = np.load(data[4])     # K

        # Causal class of each variant: mediated if it is a caQTL causal variant of a peak with a non-zero link, else direct if causal
        causal = causal_eqtl_effects != 0.0
        mediated = np.any((causal_caqtl_effects != 0.0) & (causal_peak_gene_effects[:, None] != 0.0), axis=0)
        causal_class = np.array(['none'] * len(causal), dtype=object)
        causal_class[causal] = 'direct'
        causal_class[causal & mediated] = 'mediated'

        pips1 = eqtl_only_pips[gene_id]
        pips2 = ca_qtl_mediated_pips[gene_id]
        if len(pips1) != len(causal) or len(pips2) != len(causal):
            print('assumption error: PIP vectors and causal effect vectors have different lengths for ' + gene_id)
            pdb.set_trace()

        for variant_index in range(len(causal)):
            t.write('\t'.join([str(simulation_number), gene_id, str(variant_index), is_constrained, causal_class[variant_index], str(pips1[variant_index]), str(pips2[variant_index])]) + '\n')
    f.close()
t.close()
print('organized results for ' + str(n_simulations_found) + ' simulations', flush=True)
