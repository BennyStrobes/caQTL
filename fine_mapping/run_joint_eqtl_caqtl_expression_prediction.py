import sys
import os
import argparse
import pdb
import numpy as np
from joint_eqtl_caqtl_expression_prediction import JOINT_EQTL_CAQTL_EXPRESSION_PREDICTION

def load_in_input_data(sumstat_summary_file):
    # Initialize data storage object
    gene_to_data = {}
    n_peaks_filtered = 0
    n_peaks_total = 0

    f = open(sumstat_summary_file, 'r')
    head_count = 0
    for line in f:
        data = line.rstrip().split('\t')
        if head_count == 0:
            head_count += 1
            continue
        # Extract info in line
        gene_id = data[0]
        variant_ids_file = data[1]
        ld_file = data[2]
        eqtl_effects_file = data[3]
        eqtl_se_file = data[4]
        caqtl_effects_file = data[5]
        caqtl_se_file = data[6]
        peak_effects_file = data[7]
        peak_se_file = data[8]

        # Load in data
        caqtl_effects = np.load(caqtl_effects_file)
        caqtl_se = np.load(caqtl_se_file)
        peak_effects = np.load(peak_effects_file)
        peak_se = np.load(peak_se_file)
        eqtl_effects = np.load(eqtl_effects_file)
        eqtl_se = np.load(eqtl_se_file)

        # Load in peak ids if they exist (file sits next to the variant ids file; not listed in the summary file)
        peak_ids_file = variant_ids_file.split('_variant_ids.txt')[0] + '_peak_ids.txt'
        if os.path.isfile(peak_ids_file) and os.path.getsize(peak_ids_file) > 0:
            peak_ids = np.loadtxt(peak_ids_file, dtype=str, ndmin=1)
        else:
            peak_ids = np.asarray([], dtype=str)
        if len(peak_ids) != len(peak_effects):
            # Peak ids not available (or do not match the peaks): use NA
            peak_ids = np.asarray(['NA']*len(peak_effects))

        # Filter out peaks with invalid peak-gene effects (standard error of zero or non-finite values)
        valid_peaks = np.isfinite(peak_effects) & np.isfinite(peak_se) & (peak_se > 0.0)
        n_peaks_filtered += np.sum(~valid_peaks)
        n_peaks_total += len(valid_peaks)
        caqtl_effects = caqtl_effects[valid_peaks, :]
        caqtl_se = caqtl_se[valid_peaks, :]
        peak_effects = peak_effects[valid_peaks]
        peak_se = peak_se[valid_peaks]
        peak_ids = peak_ids[valid_peaks]

        if gene_id in gene_to_data:
            print("Error: gene_id %s already in gene_to_data" % (gene_id))
            sys.exit(1)
        if gene_id not in gene_to_data:
            gene_to_data[gene_id] = {}

        gene_to_data[gene_id]['variant_ids_file'] = variant_ids_file
        gene_to_data[gene_id]['ld_file'] = ld_file
        gene_to_data[gene_id]['eqtl_effects'] = eqtl_effects
        gene_to_data[gene_id]['eqtl_se'] = eqtl_se
        gene_to_data[gene_id]['caqtl_effects'] = caqtl_effects
        gene_to_data[gene_id]['caqtl_se'] = caqtl_se
        gene_to_data[gene_id]['peak_effects'] = peak_effects
        gene_to_data[gene_id]['peak_se'] = peak_se
        gene_to_data[gene_id]['peak_ids'] = peak_ids

    f.close()
    print('Filtered out ' + str(n_peaks_filtered) + ' of ' + str(n_peaks_total) + ' peak-gene pairs with invalid peak-gene effects')
    return gene_to_data


def array_to_string(arr):
    # Comma-separated string (NA if array is empty)
    if len(arr) == 0:
        return 'NA'
    return ','.join(['%.6g' % x for x in arr])


def write_results_to_summary_file(model, gene_to_data, mediation_indicator, output_file):
    # One line per gene
    t = open(output_file, 'w')
    t.write('\t'.join(['gene_id', 'variant_ids_file', 'eqtl_posterior_mean', 'eqtl_alpha', 'peak_ids', 'peak_gene_posterior_mean', 'peak_gene_alpha', 'peak_mediation_prob']) + '\n')
    for gene_id in model.gene_ids:
        # eQTL effects (posterior mean = alpha*mu. These are the expression prediction weights, on scale of standardized genotype)
        eqtl_alpha = model.eqtl_alphas[gene_id]
        eqtl_posterior_mean = eqtl_alpha*model.eqtl_mus[gene_id]
        # Peak-gene effects
        peak_ids = gene_to_data[gene_id]['peak_ids']
        peak_gene_alpha = model.peak_gene_alphas[gene_id]
        peak_gene_posterior_mean = peak_gene_alpha*model.peak_gene_mus[gene_id]
        if len(peak_ids) == 0:
            peak_ids_string = 'NA'
        else:
            peak_ids_string = ','.join(peak_ids)
        # Probability peak mediates gene (only estimated if mediation_indicator is True)
        if mediation_indicator:
            peak_mediation_prob_string = array_to_string(model.mediation_probs[gene_id])
        else:
            peak_mediation_prob_string = 'NA'
        t.write('\t'.join([gene_id, gene_to_data[gene_id]['variant_ids_file'], array_to_string(eqtl_posterior_mean), array_to_string(eqtl_alpha), peak_ids_string, array_to_string(peak_gene_posterior_mean), array_to_string(peak_gene_alpha), peak_mediation_prob_string]) + '\n')
    t.close()
    print(str(len(model.gene_ids)) + ' genes written to ' + output_file)
    return


##########################
# Command line arguments
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--sumstat_summary_file', type=str)  # one line per gene (fine-mapping input summary from prepare_input_data_for_fine_mapping.py)
parser.add_argument('--expression_prediction_output_file', type=str)
parser.add_argument('--mediation_indicator', default=False, action='store_true')  # If set, each peak-gene pair gets an additional binary variable indicating whether the peak mediates eQTL effects on the gene


args = parser.parse_args()

sumstat_summary_file = args.sumstat_summary_file
expression_prediction_output_file = args.expression_prediction_output_file
mediation_indicator = args.mediation_indicator



# Load in data
gene_input_data = load_in_input_data(sumstat_summary_file)


# Fit joint eQTL/caQTL expression prediction model (once, across all genes)
expression_prediction_model = JOINT_EQTL_CAQTL_EXPRESSION_PREDICTION(mediation_indicator=mediation_indicator, max_iter=200).fit(gene_input_data)

# Write results to summary file (one line per gene)
write_results_to_summary_file(expression_prediction_model, gene_input_data, mediation_indicator, expression_prediction_output_file)


