import sys
import os
import argparse
import pdb
import numpy as np
from eqtl_only_expression_prediction import EQTL_ONLY_EXPRESSION_PREDICTION

def load_in_input_data(sumstat_summary_file, min_variants_per_gene=10):
    # Initialize data storage object
    gene_to_data = {}
    n_genes_filtered = 0

    f = open(sumstat_summary_file, 'r')
    head_count = 0
    for line in f:
        data = line.rstrip().split('\t')
        if head_count == 0:
            head_count += 1
            continue
        # Extract info in line (caQTL and peak-gene files in the summary file are not used by the eQTL-only model)
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

    f.close()
    print('Filtered out ' + str(n_genes_filtered) + ' genes with fewer than ' + str(min_variants_per_gene) + ' variants (' + str(len(gene_to_data)) + ' genes remain)')
    return gene_to_data


def array_to_string(arr):
    # Comma-separated string (NA if array is empty)
    if len(arr) == 0:
        return 'NA'
    return ','.join(['%.6g' % x for x in arr])


def write_results_to_summary_file(model, gene_to_data, output_file):
    # One line per gene (same first four columns as the joint eQTL/caQTL expression prediction results file)
    t = open(output_file, 'w')
    t.write('\t'.join(['gene_id', 'variant_ids_file', 'eqtl_posterior_mean', 'eqtl_alpha']) + '\n')
    for gene_id in model.gene_ids:
        # eQTL effects (posterior mean = alpha*mu. These are the expression prediction weights, on scale of standardized genotype)
        eqtl_alpha = model.eqtl_alphas[gene_id]
        eqtl_posterior_mean = eqtl_alpha*model.eqtl_mus[gene_id]
        t.write('\t'.join([gene_id, gene_to_data[gene_id]['variant_ids_file'], array_to_string(eqtl_posterior_mean), array_to_string(eqtl_alpha)]) + '\n')
    t.close()
    print(str(len(model.gene_ids)) + ' genes written to ' + output_file)
    return


##########################
# Command line arguments
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--sumstat_summary_file', type=str)  # one line per gene (fine-mapping input summary from prepare_input_data_for_fine_mapping.py)
parser.add_argument('--expression_prediction_output_file', type=str)


args = parser.parse_args()

sumstat_summary_file = args.sumstat_summary_file
expression_prediction_output_file = args.expression_prediction_output_file



# Load in data
gene_input_data = load_in_input_data(sumstat_summary_file)


# Fit eQTL-only expression prediction model (once, across all genes)
expression_prediction_model = EQTL_ONLY_EXPRESSION_PREDICTION(max_iter=200).fit(gene_input_data)

# Write results to summary file (one line per gene)
write_results_to_summary_file(expression_prediction_model, gene_input_data, expression_prediction_output_file)


