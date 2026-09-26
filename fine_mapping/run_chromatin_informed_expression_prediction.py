import sys
import os
import argparse
import pdb
import numpy as np
from chromatin_informed_expression_prediction import CHROMATIN_INFORMED_EXPRESSION_PREDICTION


def load_in_chromatin_only_results(chromatin_only_results_file):
    # Results file from run_chromatin_only_expression_prediction.py (stage 1)
    # Returns dictionary: gene_id -> (variant_ids_file, probability of chromatin path, mean of chromatin-predicted eQTL effect given path, variance of chromatin-predicted eQTL effect given path)
    gene_to_chromatin = {}
    f = open(chromatin_only_results_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            path_prob_col = data.index('eqtl_alpha')
            pred_mean_col = data.index('pred_mean_given_path')
            pred_var_col = data.index('pred_var_given_path')
            continue
        gene_id = data[0]
        path_probs = np.asarray(data[path_prob_col].split(',')).astype(float)
        pred_mean_given_path = np.asarray(data[pred_mean_col].split(',')).astype(float)
        pred_var_given_path = np.asarray(data[pred_var_col].split(',')).astype(float)
        gene_to_chromatin[gene_id] = (data[1], path_probs, pred_mean_given_path, pred_var_given_path)
    f.close()
    return gene_to_chromatin


def load_in_input_data(sumstat_summary_file, chromatin_only_results_file, min_variants_per_gene=10):
    # Chromatin info from stage 1
    gene_to_chromatin = load_in_chromatin_only_results(chromatin_only_results_file)

    # Initialize data storage object
    gene_to_data = {}
    n_genes_filtered = 0
    n_genes_without_chromatin_results = 0

    f = open(sumstat_summary_file, 'r')
    head_count = 0
    for line in f:
        data = line.rstrip().split('\t')
        if head_count == 0:
            head_count += 1
            continue
        # Extract info in line (caQTL and peak-gene files in the summary file are not used here. They enter through the stage 1 results)
        gene_id = data[0]
        variant_ids_file = data[1]
        ld_file = data[2]
        eqtl_effects_file = data[3]
        eqtl_se_file = data[4]

        # Load in data
        eqtl_effects = np.load(eqtl_effects_file)
        eqtl_se = np.load(eqtl_se_file)

        # Filter out genes with too few variants
        if len(eqtl_effects) < min_variants_per_gene:
            n_genes_filtered += 1
            continue

        if gene_id in gene_to_data:
            print("Error: gene_id %s already in gene_to_data" % (gene_id))
            sys.exit(1)
        if gene_id not in gene_to_data:
            gene_to_data[gene_id] = {}

        gene_to_data[gene_id]['variant_ids_file'] = variant_ids_file
        gene_to_data[gene_id]['ld_file'] = ld_file
        gene_to_data[gene_id]['eqtl_effects'] = eqtl_effects
        gene_to_data[gene_id]['eqtl_se'] = eqtl_se

        # Chromatin info for the gene
        if gene_id in gene_to_chromatin:
            if gene_to_chromatin[gene_id][0] != variant_ids_file or len(gene_to_chromatin[gene_id][1]) != len(eqtl_effects):
                print("Error: stage 1 results for gene_id %s do not use the same variants" % (gene_id))
                sys.exit(1)
            gene_to_data[gene_id]['path_probs'] = gene_to_chromatin[gene_id][1]
            gene_to_data[gene_id]['pred_mean_given_path'] = gene_to_chromatin[gene_id][2]
            gene_to_data[gene_id]['pred_var_given_path'] = gene_to_chromatin[gene_id][3]
        else:
            # Gene not in stage 1 results: no variants have a chromatin path
            n_genes_without_chromatin_results += 1
            gene_to_data[gene_id]['path_probs'] = np.zeros(len(eqtl_effects))
            gene_to_data[gene_id]['pred_mean_given_path'] = np.zeros(len(eqtl_effects))
            gene_to_data[gene_id]['pred_var_given_path'] = np.zeros(len(eqtl_effects))

    f.close()
    print('Filtered out ' + str(n_genes_filtered) + ' genes with fewer than ' + str(min_variants_per_gene) + ' variants (' + str(len(gene_to_data)) + ' genes remain)')
    print(str(n_genes_without_chromatin_results) + ' genes not found in stage 1 (chromatin only) results')
    return gene_to_data


def array_to_string(arr):
    # Comma-separated string (NA if array is empty)
    if len(arr) == 0:
        return 'NA'
    return ','.join(['%.6g' % x for x in arr])


def write_results_to_summary_file(model, gene_to_data, output_file):
    # One line per gene (same first four columns as the other expression prediction results files)
    # eqtl_prob_with_chromatin_path: posterior probability variant has an eQTL effect that follows the chromatin-predicted effect
    # chromatin_path_prob: prior probability variant has a chromatin path to the gene (from stage 1, after thresholding)
    t = open(output_file, 'w')
    t.write('\t'.join(['gene_id', 'variant_ids_file', 'eqtl_posterior_mean', 'eqtl_alpha', 'eqtl_prob_with_chromatin_path', 'chromatin_path_prob']) + '\n')
    for gene_id in model.gene_ids:
        # eQTL effects (posterior mean. These are the expression prediction weights, on scale of standardized genotype)
        eqtl_posterior_mean = model.get_posterior_mean_eqtl_effects(gene_id)
        eqtl_alpha = model.q0[gene_id] + model.q1[gene_id]
        t.write('\t'.join([gene_id, gene_to_data[gene_id]['variant_ids_file'], array_to_string(eqtl_posterior_mean), array_to_string(eqtl_alpha), array_to_string(model.q1[gene_id]), array_to_string(model.path_probs[gene_id])]) + '\n')
    t.close()
    print(str(len(model.gene_ids)) + ' genes written to ' + output_file)
    return


##########################
# Command line arguments
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--sumstat_summary_file', type=str)  # one line per gene (fine-mapping input summary from prepare_input_data_for_fine_mapping.py)
parser.add_argument('--chromatin_only_results_file', type=str)  # stage 1 results (output of run_chromatin_only_expression_prediction.py)
parser.add_argument('--expression_prediction_output_file', type=str)


args = parser.parse_args()

sumstat_summary_file = args.sumstat_summary_file
chromatin_only_results_file = args.chromatin_only_results_file
expression_prediction_output_file = args.expression_prediction_output_file



# Load in data
gene_input_data = load_in_input_data(sumstat_summary_file, chromatin_only_results_file)


# Fit chromatin-informed expression prediction model (once, across all genes)
expression_prediction_model = CHROMATIN_INFORMED_EXPRESSION_PREDICTION(max_iter=200).fit(gene_input_data)

# Write results to summary file (one line per gene)
write_results_to_summary_file(expression_prediction_model, gene_input_data, expression_prediction_output_file)
