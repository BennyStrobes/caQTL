import sys
import os
import argparse
import pdb
import numpy as np
# Model code (susie_rss.py, ca_qtl_mediated_susie.py) is copied into this directory from ../fine_mapping_simulation; import from here
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from susie_rss import SUSIE_RSS


parser = argparse.ArgumentParser()
parser.add_argument('--sumstat_summary_file', type=str)  # one line per gene (fine-mapping input summary from prepare_input_data_for_fine_mapping.py)
parser.add_argument('--fine_mapping_output_file', type=str)
parser.add_argument('--eqtl_sample_size', type=int, default=None)  # None: z-score model (X'X = R, X'y = z, residual variance 1)
parser.add_argument('--L', type=int, default=10)
parser.add_argument('--estimate_residual_variance', action='store_true', default=False)  # z-score mode: estimate a residual variance per gene (z ~ N(Rb, sigma2 R), y'y = z'R^-1 z, n = p) instead of fixing it at 1
args = parser.parse_args()

sumstat_summary_file = args.sumstat_summary_file
fine_mapping_output_file = args.fine_mapping_output_file
eqtl_sample_size = args.eqtl_sample_size
L = args.L

t = open(fine_mapping_output_file, 'w')
t.write('gene_id\tvariant_ids_file\tpips\tcredible_sets\tconverged\tresidual_variance\teqtl_posterior_mean_file\n')
output_stem = os.path.splitext(fine_mapping_output_file)[0]
print('settings: zscore_mode ' + str(eqtl_sample_size is None) + ', zscore_residual_variance ' + str(args.estimate_residual_variance) + ', L ' + str(L), flush=True)

f = open(sumstat_summary_file, 'r')
head_count = 0
gene_counter = 0
for line in f:
    data = line.rstrip().split('\t')
    if head_count == 0:
        head_count += 1
        continue
    gene_id = data[0]
    variant_ids_file = data[1]
    ld_file = data[2]
    eqtl_effects_file = data[3]
    eqtl_se_file = data[4]

    ld_matrix = np.load(ld_file)
    eqtl_effects = np.load(eqtl_effects_file)
    eqtl_se = np.load(eqtl_se_file)

    susie_fit = SUSIE_RSS(L=L, zscore_residual_variance=args.estimate_residual_variance).fit(eqtl_effects, eqtl_se, ld_matrix, eqtl_sample_size)

    # Credible sets: variant indices (0-based) per set, sets separated by ';'
    credible_sets_string = ';'.join([','.join([str(x) for x in cs['snp_indices']]) for cs in susie_fit.credible_sets])
    if credible_sets_string == '':
        credible_sets_string = 'NA'
    # Posterior mean effect per variant (sum over single effects of alpha * mu; z scale in z-score mode), same per-gene file convention as the mediated runner
    eqtl_posterior_mean_file = output_stem + '_gene_' + gene_id + '_eqtl_posterior_mean.npy'
    np.save(eqtl_posterior_mean_file, susie_fit.posterior_mean)
    t.write(gene_id + '\t' + variant_ids_file + '\t' + ','.join([str(x) for x in susie_fit.pip]) + '\t' + credible_sets_string + '\t' + str(susie_fit.converged) + '\t' + str(susie_fit.sigma2) + '\t' + eqtl_posterior_mean_file + '\n')
    t.flush()
    gene_counter += 1
    if gene_counter % 500 == 0:
        print('gene ' + str(gene_counter), flush=True)
f.close()
t.close()
print(str(gene_counter) + ' genes written to ' + fine_mapping_output_file)
