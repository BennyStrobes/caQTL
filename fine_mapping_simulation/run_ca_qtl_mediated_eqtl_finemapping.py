import numpy as np
import os
import sys
import pdb
import argparse
from susie_rss import SUSIE_RSS
from ca_qtl_mediated_susie import CAQTL_MEDIATED_SUSIE














def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sumstat_summary_file', type=str, required=True, help='one line per gene')
    parser.add_argument('--fine_mapping_output_file', type=str, required=True, help='eQTL PIPs, same format as the eQTL-only fine-mapping output')
    parser.add_argument('--eqtl_sample_size', type=int, required=True)
    parser.add_argument('--shared_hyperparameter_output_file', type=str, required=True)
    parser.add_argument('--write_component_output', action='store_true', default=False, help='also write per-gene direct/mediated/caQTL PIPs and link posteriors')
    return parser.parse_args()


########################
# Command line args
########################
args = parse_args()
sumstat_summary_file = args.sumstat_summary_file
fine_mapping_output_file = args.fine_mapping_output_file
eqtl_sample_size = args.eqtl_sample_size
shared_hyperparameter_output_file = args.shared_hyperparameter_output_file
write_component_output = args.write_component_output



#######################
# 1. Load in input data
#######################
eqtl_effects = []
eqtl_ses = []
caqtl_effects = []
caqtl_ses = []
peak_gene_effects = []
peak_gene_ses = []
ld_file_names = []
gene_ids = []
variant_ids_files = []

# Loop through genes and run SuSiE fine-mapping for each gene
f = open(sumstat_summary_file, 'r')
head_count = 0
for line in f:
    line = line.strip()
    data = line.split('\t')
    # Skip the header line
    if head_count == 0:
        head_count += 1
        header = np.copy(data)
        continue

    # Extract relevant information for the gene
    gene_id = data[0]
    variant_ids_file = data[1]
    ld_file = data[2]
    eqtl_effects_file = data[3]
    eqtl_se_file = data[4]
    caqtl_effects_file = data[5]
    caqtl_se_file = data[6]
    peak_gene_effects_file = data[7]
    peak_gene_se_file = data[8]

    # Load in data for the gene
    ld_matrix = np.load(ld_file)
    eqtl_effect = np.load(eqtl_effects_file)
    eqtl_se = np.load(eqtl_se_file)
    caqtl_effect = np.load(caqtl_effects_file)
    caqtl_se = np.load(caqtl_se_file)
    peak_gene_effect = np.load(peak_gene_effects_file)
    peak_gene_se = np.load(peak_gene_se_file)

    # Add the gene's data to the lists
    eqtl_effects.append(eqtl_effect)
    eqtl_ses.append(eqtl_se)
    caqtl_effects.append(caqtl_effect)
    caqtl_ses.append(caqtl_se)
    peak_gene_effects.append(peak_gene_effect)
    peak_gene_ses.append(peak_gene_se)
    ld_file_names.append(ld_file)
    gene_ids.append(gene_id)
    variant_ids_files.append(variant_ids_file)
f.close()

input_data_obj = {
    'eqtl_effects': eqtl_effects,
    'eqtl_ses': eqtl_ses,
    'caqtl_effects': caqtl_effects,
    'caqtl_ses': caqtl_ses,
    'peak_gene_effects': peak_gene_effects,
    'peak_gene_ses': peak_gene_ses,
    'ld_file_names': ld_file_names
}


#######################
# 2. Run new joint eqtl-caQTL fine-mapping jointly across genes
#######################
# caQTL and eQTL sumstats come from the same simulated individuals
caqtl_sample_size = eqtl_sample_size

model = CAQTL_MEDIATED_SUSIE(L_eqtl=10, L_caqtl=5).fit(input_data_obj, eqtl_sample_size, caqtl_sample_size)


#######################
# 3. Save results
#######################
# eQTL PIPs, same format as the eQTL-only fine-mapping output
t = open(fine_mapping_output_file, 'w')
t.write('gene_id\tvariant_ids_file\tpips\n')
for g in range(len(gene_ids)):
    t.write(gene_ids[g] + '\t' + variant_ids_files[g] + '\t' + ','.join([str(x) for x in model.gene_results[g]['eqtl_pip']]) + '\n')
t.close()

# Shared hyperparameters and convergence
t = open(shared_hyperparameter_output_file, 'w')
t.write('link_prior_prob\tlink_prior_variance\tlink_scale\tlink_bias_variance\tn_outer_iter\tconverged\tfinal_elbo\n')
t.write('\t'.join(map(str, [model.pi, model.tau2_g, model.link_scale, model.link_bias_variance, model.n_iter, model.converged, model.elbo[-1]])) + '\n')
t.close()

# Component-level results per gene (npy) with a cross-gene summary of their file names and link posteriors (optional)
if write_component_output:
    output_stem = os.path.splitext(fine_mapping_output_file)[0]
    t = open(output_stem + '_components.txt', 'w')
    t.write('gene_id\tinit\tlink_probs\tlink_means\tdirect_pip_file\tmediated_pip_file\tcaqtl_pip_file\teqtl_posterior_mean_file\n')
    for g in range(len(gene_ids)):
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
        t.write('\t'.join([gene_ids[g], res['init'], ','.join(map(str, res['link_prob'])), ','.join(map(str, res['link_mean'])), direct_pip_file, mediated_pip_file, caqtl_pip_file, eqtl_posterior_mean_file]) + '\n')
    t.close()
