import sys
import os
import argparse
import pdb
import numpy as np
# Model code (susie_rss.py, ca_qtl_mediated_susie.py) is copied into this directory from ../fine_mapping_simulation; import from here
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_qtl_elements_susie import CAQTL_ELEMENTS_SUSIE


parser = argparse.ArgumentParser()
parser.add_argument('--sumstat_summary_file', type=str)  # one line per gene (fine-mapping input summary from prepare_input_data_for_fine_mapping.py)
parser.add_argument('--fine_mapping_output_file', type=str)  # eQTL PIPs, same format as the eQTL-only fine-mapping output
parser.add_argument('--shared_hyperparameter_output_file', type=str)
parser.add_argument('--eqtl_sample_size', type=int, default=None)  # None (with caqtl_sample_size None): z-score model, sample sizes not needed
parser.add_argument('--caqtl_sample_size', type=int, default=None)
parser.add_argument('--L_eqtl', type=int, default=10)
parser.add_argument('--L_caqtl', type=int, default=5)
parser.add_argument('--link_scale_init', type=float, default=1.0)  # lambda: fixed value, or the prior mean / starting value with --estimate_link_scale
parser.add_argument('--estimate_link_scale', action='store_true', default=False)  # treat lambda as a random variable (Gaussian variational factor) updated from the selected peaks
parser.add_argument('--link_scale_prior_sd', type=float, default=None)  # prior sd of lambda (default: 0.25 x link_scale_init)
parser.add_argument('--no_estimate_link_bias_variance', action='store_true', default=False)  # fix tau_u^2 at --link_bias_variance
parser.add_argument('--link_bias_variance', type=float, default=0.0)
parser.add_argument('--peak_weight_ratio_init', type=float, default=1.0)  # rho_p = omega_p / omega_v: prior weight of a real-link peak element relative to a variant
parser.add_argument('--no_estimate_peak_weight_ratio', action='store_true', default=False)
parser.add_argument('--free_peak_effects', action='store_true', default=False)  # E-free: selected peaks get a free (symmetric) effect instead of the link value g_k
parser.add_argument('--pool_gate', type=str, default='running', choices=['running', 'hurdle'])  # peak pool weight rho_p x q: q refreshed each outer iteration from the full link posterior ('running') or fixed at the hurdle-only value ('hurdle')
parser.add_argument('--max_outer_iter', type=int, default=100)
parser.add_argument('--hyper_tol', type=float, default=1e-3)
parser.add_argument('--elbo_tol', type=float, default=1e-3)
parser.add_argument('--estimate_residual_variance', action='store_true', default=False)  # z-score mode: estimate a residual variance per trait per gene (z ~ N(Rb, sigma2 R), y'y = z'R^-1 z, n = p) instead of fixing it at 1
parser.add_argument('--no_component_output', action='store_true', default=False)  # skip per-gene direct/mediated/caQTL PIPs and link posteriors
args = parser.parse_args()

sumstat_summary_file = args.sumstat_summary_file
fine_mapping_output_file = args.fine_mapping_output_file
shared_hyperparameter_output_file = args.shared_hyperparameter_output_file
eqtl_sample_size = args.eqtl_sample_size
caqtl_sample_size = args.caqtl_sample_size


#######################
# 1. Load in input data
#######################
input_data_obj = {'eqtl_effects': [], 'eqtl_ses': [], 'caqtl_effects': [], 'caqtl_ses': [], 'peak_gene_effects': [], 'peak_gene_ses': [], 'ld_file_names': []}
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
f.close()
n_genes = len(gene_ids)

# Drop links with a non-finite hurdle estimate or a non-positive / non-finite SE (failed hurdle fits carry no information;
# a placeholder with a huge SE would wreck the moment-based tau_g^2 initialization). The peak's caQTL rows go with it, so K shrinks.
# kept_link_indices[g] maps the remaining links back to the rows of the gene's input link files.
kept_link_indices = []
n_bad_links = 0
n_bad_sumstat_genes = 0
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
    for key in ['eqtl_effects', 'eqtl_ses', 'caqtl_effects', 'caqtl_ses']:
        if not np.all(np.isfinite(input_data_obj[key][g])) or np.any(input_data_obj[key][g] <= 0.0) and key.endswith('ses'):
            n_bad_sumstat_genes += 1
            print('assumption error: non-finite or non-positive values in ' + key + ' for gene ' + gene_ids[g])
            pdb.set_trace()
print(str(n_bad_links) + ' links with non-finite effect or non-positive/non-finite se dropped', flush=True)
n_genes_no_peaks = int(np.sum([len(x) == 0 for x in input_data_obj['peak_gene_effects']]))
print(str(n_genes) + ' genes loaded (' + str(n_genes_no_peaks) + ' with no linked peaks, fit as eQTL-only)', flush=True)


#######################
# 2. Run joint caQTL-mediated eQTL fine-mapping across genes
#######################
if args.estimate_link_scale and not args.no_estimate_link_bias_variance:
    print('assumption error: --estimate_link_scale requires --no_estimate_link_bias_variance (lambda and tau_u^2 are not jointly identified); fix tau_u^2 with --link_bias_variance')
    sys.exit(1)
model = CAQTL_ELEMENTS_SUSIE(L_eqtl=args.L_eqtl, L_caqtl=args.L_caqtl, link_scale_init=args.link_scale_init, estimate_link_scale=args.estimate_link_scale, link_scale_prior_sd=args.link_scale_prior_sd, estimate_link_bias_variance=(not args.no_estimate_link_bias_variance), link_bias_variance_init=args.link_bias_variance, peak_weight_ratio_init=args.peak_weight_ratio_init, estimate_peak_weight_ratio=(not args.no_estimate_peak_weight_ratio), free_peak_effects=args.free_peak_effects, pool_gate=args.pool_gate, zscore_residual_variance=args.estimate_residual_variance, max_outer_iter=args.max_outer_iter, tol=args.elbo_tol, hyper_tol=args.hyper_tol)
print('settings (Model E): zscore_mode ' + str(eqtl_sample_size is None) + ', zscore_residual_variance ' + str(model.zscore_residual_variance) + ', lambda ' + str(model.link_scale_init) + (' (random variable, prior sd ' + str(model.link_scale_prior_sd) + ')' if model.estimate_link_scale else ' (fixed)') + ', estimate_link_bias_variance ' + str(model.estimate_link_bias_variance) + ', peak_weight_ratio_init ' + str(model.peak_weight_ratio_init) + ', estimate_peak_weight_ratio ' + str(model.estimate_peak_weight_ratio) + ', free_peak_effects ' + str(model.free_peak_effects) + ', pool_gate ' + model.pool_gate + ', L_eqtl ' + str(model.L_eqtl) + ', L_caqtl ' + str(model.L_caqtl), flush=True)


def save_outputs(model, n_iter_done=None):
    # Writes the results file, shared hyperparameters, per-gene convergence and (unless suppressed) the component files
    #######################
    # 3. Save results
    #######################
    # eQTL PIPs, same format as the eQTL-only fine-mapping output
    t = open(fine_mapping_output_file, 'w')
    t.write('gene_id\tvariant_ids_file\tpips\n')
    for g in range(n_genes):
        t.write(gene_ids[g] + '\t' + variant_ids_files[g] + '\t' + ','.join([str(x) for x in model.gene_results[g]['eqtl_pip']]) + '\n')
    t.close()

    # Shared hyperparameters and convergence
    t = open(shared_hyperparameter_output_file, 'w')
    t.write('link_prior_prob\tlink_prior_variance\tlink_scale\tlink_scale_sd\tlink_bias_variance\tpeak_weight_ratio\tfree_peak_effects\tpool_gate\tn_outer_iter\tconverged\tfinal_elbo\tzscore_mode\n')
    t.write('\t'.join(map(str, [model.pi, model.tau2_g, model.link_scale, np.sqrt(model.link_scale_var), model.link_bias_variance, model.rho_p, model.free_peak_effects, model.pool_gate, model.n_iter, model.converged, model.elbo[-1], model.zscore_mode])) + '\n')
    t.close()

    # Per-gene convergence diagnostics: final gene ELBO and its change in the last outer iteration (large values = still drifting)
    t = open(os.path.splitext(fine_mapping_output_file)[0] + '_gene_convergence.txt', 'w')
    t.write('gene_id\tp\tK\tinit\tgene_elbo\tlast_elbo_increment\tsigma2_eqtl\tsigma2_caqtl_mean\n')
    for g in range(n_genes):
        sigma2_caqtl_mean = np.mean(model.gene_results[g]['sigma2_A']) if model.data[g]['K'] > 0 else float('nan')
        t.write('\t'.join(map(str, [gene_ids[g], model.data[g]['p'], model.data[g]['K'], model.gene_results[g]['init'], model.gene_results[g]['elbo'], model.gene_elbo_increment[g], model.gene_results[g]['sigma2_E'], sigma2_caqtl_mean])) + '\n')
    t.close()

    # Component-level results per gene (npy) with a cross-gene summary of their file names and link posteriors
    if not args.no_component_output:
        output_stem = os.path.splitext(fine_mapping_output_file)[0]
        t = open(output_stem + '_components.txt', 'w')
        t.write('gene_id\tinit\tkept_link_indices\tlink_probs\tlink_means\tlink_inclusion_probs\tpeak_selection_probs\tpeak_selection_mass\tdirect_pip_file\tmediated_pip_file\tcaqtl_pip_file\teqtl_posterior_mean_file\n')
        for g in range(n_genes):
            res = model.gene_results[g]
            gene_stem = output_stem + '_gene_' + gene_ids[g]
            direct_pip_file = gene_stem + '_direct_pip.npy'
            mediated_pip_file = gene_stem + '_mediated_pip.npy'
            caqtl_pip_file = gene_stem + '_caqtl_pip.npy'
            eqtl_posterior_mean_file = gene_stem + '_eqtl_posterior_mean.npy'
            np.save(direct_pip_file, res['direct_pip'])
            np.save(mediated_pip_file, res['mediated_pip'])
            np.save(caqtl_pip_file, res['caqtl_pip'])
            np.save(eqtl_posterior_mean_file, res['eqtl_posterior_mean'])
            t.write('\t'.join([gene_ids[g], res['init'], ','.join(map(str, kept_link_indices[g])), ','.join(map(str, res['link_prob'])), ','.join(map(str, res['link_mean'])), ','.join(map(str, res['link_inclusion_prob'])), ','.join(map(str, res['peak_selection_prob'])), ','.join(map(str, res['peak_selection_mass'])), direct_pip_file, mediated_pip_file, caqtl_pip_file, eqtl_posterior_mean_file]) + '\n')
        t.close()



model.fit(input_data_obj, eqtl_sample_size, caqtl_sample_size)
save_outputs(model)
print('done', flush=True)
