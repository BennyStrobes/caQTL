import sys
import os
import argparse
import numpy as np
# Model L: causal eQTL effects as a spike plus a Gaussian centred on the chromatin-predicted effect
# (see caqtl_mediated_fine_mapping_model_proposal_predicted_mean.md, Section 2). Same inputs and output formats as the other runners.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_qtl_predicted_mean_susie import CAQTL_PREDICTED_MEAN_SUSIE


parser = argparse.ArgumentParser()
parser.add_argument('--sumstat_summary_file', type=str)
parser.add_argument('--fine_mapping_output_file', type=str)
parser.add_argument('--shared_hyperparameter_output_file', type=str)
parser.add_argument('--L_eqtl', type=int, default=10)
parser.add_argument('--L_caqtl', type=int, default=5)
parser.add_argument('--t_bin_edges', type=str, default='2,4')  # hurdle |z| bin edges: one lambda per bin; 'none' for a single lambda
parser.add_argument('--lambda_prior_sd', type=float, default=10.0)  # Gaussian prior N(0, sd^2) on each lambda
parser.add_argument('--lambda_init', type=float, default=0.0)
parser.add_argument('--sigma2_init', type=float, default=25.0)  # initial slab variance (z units) for both variance classes
parser.add_argument('--single_variance', action='store_true', default=False)  # one slab variance for linked and unlinked variants
parser.add_argument('--sigma2_floor', type=float, default=0.1)
parser.add_argument('--min_caqtl_pip', type=float, default=1e-3)  # (peak, variant) pairs with caQTL PIP x P(link non-zero) below this do not enter the variant's mixture
parser.add_argument('--max_peaks_per_variant', type=int, default=3)  # cap on the peaks in a variant's mixture (2^cap components)
parser.add_argument('--max_rounds', type=int, default=30)
parser.add_argument('--tol', type=float, default=1e-3)  # convergence: max relative change of lambda and the slab variances between rounds
parser.add_argument('--all_effects_on', action='store_true', default=False)  # do not switch single effects off (default: SuSiE's rule)
parser.add_argument('--no_component_output', action='store_true', default=False)
args = parser.parse_args()

sumstat_summary_file = args.sumstat_summary_file
fine_mapping_output_file = args.fine_mapping_output_file
shared_hyperparameter_output_file = args.shared_hyperparameter_output_file


#######################
# 1. Load in input data
#######################
input_data_obj = {'eqtl_effects': [], 'eqtl_ses': [], 'caqtl_effects': [], 'caqtl_ses': [], 'peak_gene_effects': [], 'peak_gene_ses': [], 'ld_file_names': [], 'caqtl_imputed_masks': []}
gene_ids = []
variant_ids_files = []
f = open(sumstat_summary_file, 'r')
head_count = 0
for line in f:
    data = line.rstrip().split('\t')
    if head_count == 0:
        head_count += 1
        continue
    gene_ids.append(data[0])
    variant_ids_files.append(data[1])
    input_data_obj['ld_file_names'].append(data[2])
    input_data_obj['eqtl_effects'].append(np.load(data[3]))
    input_data_obj['eqtl_ses'].append(np.load(data[4]))
    input_data_obj['caqtl_effects'].append(np.load(data[5]))
    input_data_obj['caqtl_ses'].append(np.load(data[6]))
    input_data_obj['peak_gene_effects'].append(np.load(data[7]))
    input_data_obj['peak_gene_ses'].append(np.load(data[8]))
    input_data_obj['caqtl_imputed_masks'].append(np.load(data[9]) if (len(data) > 9 and os.path.exists(data[9])) else None)
f.close()
n_genes = len(gene_ids)

# Drop links with a non-finite hurdle estimate or a non-positive / non-finite SE (failed hurdle fits), with their caQTL rows
kept_link_indices = []
n_bad_links = 0
for g in range(n_genes):
    ghat = input_data_obj['peak_gene_effects'][g]
    gse = input_data_obj['peak_gene_ses'][g]
    bad = (~np.isfinite(ghat)) | (~np.isfinite(gse)) | (gse <= 0.0)
    kept_link_indices.append(np.where(~bad)[0])
    if np.any(bad):
        n_bad_links += int(np.sum(bad))
        keep = ~bad
        input_data_obj['peak_gene_effects'][g] = ghat[keep]
        input_data_obj['peak_gene_ses'][g] = gse[keep]
        input_data_obj['caqtl_effects'][g] = np.atleast_2d(input_data_obj['caqtl_effects'][g])[keep]
        input_data_obj['caqtl_ses'][g] = np.atleast_2d(input_data_obj['caqtl_ses'][g])[keep]
        if input_data_obj['caqtl_imputed_masks'][g] is not None:
            input_data_obj['caqtl_imputed_masks'][g] = np.atleast_2d(input_data_obj['caqtl_imputed_masks'][g])[keep]
print(str(n_bad_links) + ' links with non-finite effect or non-positive/non-finite se dropped', flush=True)
n_genes_no_peaks = int(np.sum([len(x) == 0 for x in input_data_obj['peak_gene_effects']]))
print(str(n_genes) + ' genes loaded (' + str(n_genes_no_peaks) + ' with no linked peaks, fit as eQTL-only)', flush=True)


#######################
# 2. Fit
#######################
t_bin_edges = [] if args.t_bin_edges.strip().lower() in ('none', '') else [float(x) for x in args.t_bin_edges.split(',')]
model = CAQTL_PREDICTED_MEAN_SUSIE(L_eqtl=args.L_eqtl, L_caqtl=args.L_caqtl, t_bin_edges=t_bin_edges, lambda_prior_sd=args.lambda_prior_sd, lambda_init=args.lambda_init, sigma2_init=args.sigma2_init, single_variance=args.single_variance, sigma2_floor=args.sigma2_floor, min_caqtl_pip=args.min_caqtl_pip, max_peaks_per_variant=args.max_peaks_per_variant, max_rounds=args.max_rounds, tol=args.tol, switch_off_effects=(not args.all_effects_on))
print('settings (Model L, predicted-mean slab): lambda bins ' + ', '.join(model.bin_names) + ', lambda prior sd ' + str(args.lambda_prior_sd) + ', lambda init ' + str(args.lambda_init) + ', sigma2 init ' + str(args.sigma2_init) + ', single_variance ' + str(args.single_variance) + ', sigma2 floor ' + str(args.sigma2_floor) + ', min_caqtl_pip ' + str(args.min_caqtl_pip) + ', max_peaks_per_variant ' + str(args.max_peaks_per_variant) + ', switch_off_effects ' + str(not args.all_effects_on) + ', L_eqtl ' + str(model.L_eqtl) + ', L_caqtl ' + str(model.L_caqtl), flush=True)
model.fit(input_data_obj, None, None)


#######################
# 3. Save results
#######################
t = open(fine_mapping_output_file, 'w')
t.write('gene_id\tvariant_ids_file\tpips\n')
for g in range(n_genes):
    t.write(gene_ids[g] + '\t' + variant_ids_files[g] + '\t' + ','.join([str(x) for x in model.gene_results[g]['eqtl_pip']]) + '\n')
t.close()

# Shared hyperparameters (one row). link_scale = lambda of the strongest hurdle bin, for the link-support table.
t = open(shared_hyperparameter_output_file, 'w')
lam_cols = ['lambda_' + b for b in model.bin_names]
t.write('\t'.join(['link_scale'] + lam_cols + ['sigma2_unlinked', 'sigma2_linked', 'link_pi', 'link_tau2', 'lambda_prior_sd', 'expected_causal_unlinked', 'expected_causal_linked', 'n_links', 'n_rounds', 'converged', 'final_elbo', 'zscore_mode']) + '\n')
t.write('\t'.join(map(str, [model.lam[-1]] + list(model.lam) + [model.sigma2_unlinked, model.sigma2_linked, model.link_pi, model.link_tau2, model.lambda_prior_sd, model.expected_causal_unlinked, model.expected_causal_linked, model.n_links, model.n_rounds, model.converged, model.elbo[-1], True])) + '\n')
t.close()

t = open(os.path.splitext(fine_mapping_output_file)[0] + '_gene_convergence.txt', 'w')
t.write('gene_id\tp\tK\tinit\tgene_elbo\tlast_elbo_increment\tsigma2_eqtl\tsigma2_caqtl_mean\n')
for g in range(n_genes):
    t.write('\t'.join(map(str, [gene_ids[g], model.data[g]['p'], model.data[g]['K'], 'predicted_mean', model.gene_results[g]['elbo'], 0.0, 1.0, 1.0 if model.data[g]['K'] > 0 else float('nan')])) + '\n')
t.close()

if not args.no_component_output:
    output_stem = os.path.splitext(fine_mapping_output_file)[0]
    t = open(output_stem + '_components.txt', 'w')
    t.write('gene_id\tinit\tkept_link_indices\tlink_probs\tlink_means\tlink_inclusion_probs\tpeak_selection_probs\tpeak_selection_mass\tpeak_groups\tdirect_pip_file\tmediated_pip_file\tcaqtl_pip_file\teqtl_posterior_mean_file\teqtl_predicted_mean_file\tprediction_weight_file\n')
    for g in range(n_genes):
        res = model.gene_results[g]
        gene_stem = output_stem + '_gene_' + gene_ids[g]
        direct_pip_file = gene_stem + '_direct_pip.npy'
        mediated_pip_file = gene_stem + '_mediated_pip.npy'
        caqtl_pip_file = gene_stem + '_caqtl_pip.npy'
        eqtl_posterior_mean_file = gene_stem + '_eqtl_posterior_mean.npy'
        eqtl_predicted_mean_file = gene_stem + '_eqtl_predicted_mean.npy'
        prediction_weight_file = gene_stem + '_prediction_weight.npy'
        np.save(direct_pip_file, res['direct_pip'])
        np.save(mediated_pip_file, res['mediated_pip'])
        np.save(caqtl_pip_file, res['caqtl_pip'])
        np.save(eqtl_posterior_mean_file, res['eqtl_posterior_mean'])
        np.save(eqtl_predicted_mean_file, res['eqtl_predicted_mean'])
        np.save(prediction_weight_file, res['prediction_weight'])
        t.write('\t'.join([gene_ids[g], res['init'], ','.join(map(str, kept_link_indices[g])), ','.join(map(str, res['link_prob'])), ','.join(map(str, res['link_mean'])), ','.join(map(str, res['link_inclusion_prob'])), ','.join(map(str, res['peak_selection_prob'])), ','.join(map(str, res['peak_selection_mass'])), ','.join(res['peak_group']), direct_pip_file, mediated_pip_file, caqtl_pip_file, eqtl_posterior_mean_file, eqtl_predicted_mean_file, prediction_weight_file]) + '\n')
    t.close()
print('done', flush=True)
