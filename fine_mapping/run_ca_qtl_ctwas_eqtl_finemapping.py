import sys
import os
import argparse
import numpy as np
# Model J: cTWAS on peaks (see caqtl_mediated_fine_mapping_model_proposal_ctwas_groups.md). Same inputs and output formats as the other runners.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_qtl_ctwas_susie import CAQTL_CTWAS_SUSIE


parser = argparse.ArgumentParser()
parser.add_argument('--sumstat_summary_file', type=str)
parser.add_argument('--fine_mapping_output_file', type=str)
parser.add_argument('--shared_hyperparameter_output_file', type=str)
parser.add_argument('--L_eqtl', type=int, default=10)
parser.add_argument('--L_caqtl', type=int, default=5)
parser.add_argument('--t_bin_edges', type=str, default='2,4')  # hurdle |z| bin edges for the peak groups
parser.add_argument('--t_bin_quantiles', type=int, default=0)  # > 0: use this many equal-count bins of |t| instead of fixed edges (edges are printed)
parser.add_argument('--no_sign_groups', action='store_true', default=False)  # significance bins only, no concordant / discordant split
parser.add_argument('--sign_min_z', type=float, default=2.0)  # |eQTL z| at the caQTL lead needed for the marginal direction to count as clear
parser.add_argument('--sign_gate', action='store_true', default=False)  # impose the sign: discordant peaks (both directions clear) get prior weight 0 instead of a learned group weight
parser.add_argument('--group_pseudo_elements', type=float, default=10.0)  # EM shrinkage of each peak group's weight toward the variant rate (pseudo-elements)
parser.add_argument('--max_rounds', type=int, default=20)
parser.add_argument('--group_tol', type=float, default=1e-3)  # convergence: max relative change of any group weight between rounds
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
model = CAQTL_CTWAS_SUSIE(L_eqtl=args.L_eqtl, L_caqtl=args.L_caqtl, t_bin_edges=[float(x) for x in args.t_bin_edges.split(',')], t_bin_quantiles=args.t_bin_quantiles, sign_groups=(not args.no_sign_groups), sign_min_z=args.sign_min_z, sign_gate=args.sign_gate, group_pseudo_elements=args.group_pseudo_elements, max_rounds=args.max_rounds, group_tol=args.group_tol)
print('settings (Model J, cTWAS on peaks): sign_groups ' + str(not args.no_sign_groups) + ', t_bin_quantiles ' + str(args.t_bin_quantiles) + ', sign_min_z ' + str(args.sign_min_z) + ', sign_gate ' + str(args.sign_gate) + ', group_pseudo_elements ' + str(args.group_pseudo_elements) + ', L_eqtl ' + str(model.L_eqtl) + ', L_caqtl ' + str(model.L_caqtl), flush=True)
model.fit(input_data_obj, None, None)


#######################
# 3. Save results
#######################
t = open(fine_mapping_output_file, 'w')
t.write('gene_id\tvariant_ids_file\tpips\n')
for g in range(n_genes):
    t.write(gene_ids[g] + '\t' + variant_ids_files[g] + '\t' + ','.join([str(x) for x in model.gene_results[g]['eqtl_pip']]) + '\n')
t.close()

# Group weights (one row per group) plus a fixed link_scale of 1 so the link-support table's calibration reports the empirical hurdle-to-eQTL scale
t = open(shared_hyperparameter_output_file, 'w')
t.write('group\tprior_weight\tenrichment_over_variants\tn_elements\texpected_causal\tlink_scale\tn_rounds\tconverged\tfinal_elbo\tzscore_mode\n')
for i, gname in enumerate(model.groups):
    t.write('\t'.join(map(str, [gname, model.pi_group[i], model.pi_group[i] / model.pi_group[0], int(model.group_sizes[i]), model.group_counts[i], 1.0, model.n_rounds, model.converged, model.elbo[-1], True])) + '\n')
t.close()

t = open(os.path.splitext(fine_mapping_output_file)[0] + '_gene_convergence.txt', 'w')
t.write('gene_id\tp\tK\tinit\tgene_elbo\tlast_elbo_increment\tsigma2_eqtl\tsigma2_caqtl_mean\n')
for g in range(n_genes):
    t.write('\t'.join(map(str, [gene_ids[g], model.data[g]['p'], model.data[g]['K'], 'ctwas', model.gene_results[g]['elbo'], 0.0, 1.0, 1.0 if model.data[g]['K'] > 0 else float('nan')])) + '\n')
t.close()

if not args.no_component_output:
    output_stem = os.path.splitext(fine_mapping_output_file)[0]
    t = open(output_stem + '_components.txt', 'w')
    t.write('gene_id\tinit\tkept_link_indices\tlink_probs\tlink_means\tlink_inclusion_probs\tpeak_selection_probs\tpeak_selection_mass\tpeak_groups\tpeak_marginal_direction\tpeak_sign_gate\tdirect_pip_file\tmediated_pip_file\tcaqtl_pip_file\teqtl_posterior_mean_file\n')
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
        t.write('\t'.join([gene_ids[g], res['init'], ','.join(map(str, kept_link_indices[g])), ','.join(map(str, res['link_prob'])), ','.join(map(str, res['link_mean'])), ','.join(map(str, res['link_inclusion_prob'])), ','.join(map(str, res['peak_selection_prob'])), ','.join(map(str, res['peak_selection_mass'])), ','.join(res['peak_group']), ','.join(map(str, res['direction'])), ','.join(map(str, res['gate'])), direct_pip_file, mediated_pip_file, caqtl_pip_file, eqtl_posterior_mean_file]) + '\n')
    t.close()
print('done', flush=True)
