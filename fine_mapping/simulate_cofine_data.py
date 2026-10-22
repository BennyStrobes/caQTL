import sys
import os
import argparse
import pdb
import numpy as np

# Simulate eQTL, caQTL and peak-gene link summary statistics from the CoFine model (see cofine.py) and write them in the format of
# the fine-mapping input summary of prepare_input_data_for_fine_mapping.py, so that run_cofine.py runs on them unchanged.
# LD: the real LD of each gene (and its number of peaks and peak-gene standard errors) from a fine-mapping input summary file
# (--fm_input_summary_file), or synthetic AR(1) LD (--n_variants_synthetic) for tests without the real data.
# Effects are on the standardized-genotype scale. Marginal effects are drawn as b ~ N(R b_causal, s^2 R), with s = 1/sqrt(N).
# Truth (causal effects, paths) is written per gene, and the true hyper-parameters to <output_stem>_true_hyperparameters.txt.


def ld_square_root(ld):
    # L with L L' = R (R made positive semidefinite by clipping negative eigenvalues)
    eigenvalues, eigenvectors = np.linalg.eigh(ld)
    return eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))


def spike_and_slab(size, pi, sd):
    on = np.random.random(size) < pi
    return np.where(on, np.random.normal(0.0, sd, size), 0.0)


def load_real_gene_structure(fm_input_summary_file, n_genes):
    # Per gene: (variant ids file, LD file, LD regularization s, peak-gene standard errors) for the first n_genes genes
    genes = []
    f = open(fm_input_summary_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            continue
        genes.append((data[0], data[1], data[2], float(data[12]), np.atleast_1d(np.load(data[8]))))
        if len(genes) >= n_genes:
            break
    f.close()
    return genes


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str)
parser.add_argument('--output_stem', type=str, default='cofine_sim')
parser.add_argument('--fm_input_summary_file', type=str, default=None)  # real LD and peak structure; None: synthetic LD
parser.add_argument('--n_genes', type=int, default=300)
parser.add_argument('--n_variants_synthetic', type=int, default=200)  # synthetic LD only: variants per gene
parser.add_argument('--ld_rho_synthetic', type=float, default=0.95)  # synthetic LD only: R_ij = rho^|i - j|
parser.add_argument('--mean_peaks_synthetic', type=float, default=5.0)  # synthetic LD only: peaks per gene ~ 1 + Poisson(mean - 1)
parser.add_argument('--link_se_synthetic', type=float, default=0.05)  # synthetic LD only: peak-gene standard error
parser.add_argument('--eqtl_sample_size', type=int, default=1108)
parser.add_argument('--caqtl_sample_size', type=int, default=1108)
parser.add_argument('--pi0', type=float, default=0.003)
parser.add_argument('--sigma_0', type=float, default=0.1)
parser.add_argument('--pi1', type=float, default=0.5)
parser.add_argument('--lam', type=float, default=1.0)
parser.add_argument('--tau', type=float, default=0.02)
parser.add_argument('--pi_delta', type=float, default=0.01)
parser.add_argument('--sigma_delta', type=float, default=0.2)
parser.add_argument('--pi_gamma', type=float, default=0.3)
parser.add_argument('--sigma_gamma', type=float, default=0.3)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()
np.random.seed(args.seed)

os.makedirs(args.output_dir, exist_ok=True)
stem = os.path.join(args.output_dir, args.output_stem)
s_E = 1.0 / np.sqrt(args.eqtl_sample_size)
s_C = 1.0 / np.sqrt(args.caqtl_sample_size)

if args.fm_input_summary_file is not None:
    gene_structure = load_real_gene_structure(args.fm_input_summary_file, args.n_genes)
else:
    gene_structure = []
    p = args.n_variants_synthetic
    synthetic_ld = args.ld_rho_synthetic ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    synthetic_ld_file = stem + '_synthetic_ld.npy'
    np.save(synthetic_ld_file, synthetic_ld)
    synthetic_variant_ids_file = stem + '_synthetic_variant_ids.txt'
    np.savetxt(synthetic_variant_ids_file, np.asarray(['chr1_' + str(1000 + 10 * j) + '_A_G' for j in range(p)]), fmt='%s')
    for g in range(args.n_genes):
        K = 1 + np.random.poisson(args.mean_peaks_synthetic - 1.0)
        gene_structure.append(('SIMGENE' + str(g), synthetic_variant_ids_file, synthetic_ld_file, 0.0, np.full(K, args.link_se_synthetic)))

summary = open(stem + '_fine_mapping_input_summary.txt', 'w')
summary.write('Gene_Name\tVariant_IDs_File\tLD_File\teQTL_Effects_File\teQTL_SE_File\tcaQTL_Effects_File\tcaQTL_SE_File\tEstimated_Peak_Gene_Effects_File\tEstimated_Peak_Gene_SE_File\tcaQTL_Imputed_Mask_File\tcaQTL_R2pred_File\tLD_Min_Eigenvalue_Before_Repair\tLD_Regularization_s\n')
truth = open(stem + '_truth_summary.txt', 'w')
truth.write('gene_id\ttruth_file\n')
n_causal_off = 0
n_causal_on = 0
n_path = 0
for gene_id, variant_ids_file, ld_file, ld_s, link_se in gene_structure:
    ld_reg = np.load(ld_file)
    p = ld_reg.shape[0]
    ld = (ld_reg - ld_s * np.eye(p)) / (1.0 - ld_s) if ld_s > 0.0 else ld_reg
    np.fill_diagonal(ld, 1.0)
    L = ld_square_root(ld)
    K = len(link_se)

    # Causal effects from the CoFine prior
    delta = spike_and_slab((K, p), args.pi_delta, args.sigma_delta)
    gamma = spike_and_slab(K, args.pi_gamma, args.sigma_gamma)
    m = gamma @ delta
    H = np.any((delta != 0.0) & (gamma[:, None] != 0.0), axis=0)
    on = np.random.random(p) < np.where(H, args.pi1, args.pi0)
    beta = np.where(on, np.where(H, np.random.normal(args.lam * m, args.tau), np.random.normal(0.0, args.sigma_0, p)), 0.0)
    n_causal_off += int(np.sum(on & ~H))
    n_causal_on += int(np.sum(on & H))
    n_path += int(np.sum(H))

    # Marginal summary statistics
    eqtl_effects = ld @ beta + s_E * (L @ np.random.normal(size=p))
    caqtl_effects = delta @ ld + s_C * (np.random.normal(size=(K, p)) @ L.T)
    peak_gene_effects = gamma + link_se * np.random.normal(size=K)

    gene_stem = stem + '_gene_' + gene_id
    files = {}
    for name, arr in [('eqtl_effects', eqtl_effects), ('eqtl_se', np.full(p, s_E)), ('caqtl_effects', caqtl_effects), ('caqtl_se', np.full((K, p), s_C)),
                      ('peak_gene_effects', peak_gene_effects), ('peak_gene_se', link_se), ('caqtl_imputed_mask', np.zeros((K, p), dtype=bool)), ('caqtl_r2pred', np.ones((K, p)))]:
        files[name] = gene_stem + '_' + name + '.npy'
        np.save(files[name], arr)
    summary.write('\t'.join([gene_id, variant_ids_file, ld_file, files['eqtl_effects'], files['eqtl_se'], files['caqtl_effects'], files['caqtl_se'], files['peak_gene_effects'], files['peak_gene_se'], files['caqtl_imputed_mask'], files['caqtl_r2pred'], 'NA', str(ld_s)]) + '\n')
    truth_file = gene_stem + '_truth.npz'
    np.savez(truth_file, beta=beta, H=H, m=m, delta=delta, gamma=gamma)
    truth.write(gene_id + '\t' + truth_file + '\n')
summary.close()
truth.close()

t = open(stem + '_true_hyperparameters.txt', 'w')
t.write('hyperparameter\tvalue\n')
for name, value in [('pi0', args.pi0), ('sigma2_0', args.sigma_0 ** 2), ('pi1', args.pi1), ('lambda', args.lam), ('tau2', args.tau ** 2), ('pi_delta', args.pi_delta), ('sigma2_delta', args.sigma_delta ** 2), ('pi_gamma', args.pi_gamma), ('sigma2_gamma', args.sigma_gamma ** 2)]:
    t.write(name + '\t' + str(value) + '\n')
t.close()
print(str(len(gene_structure)) + ' genes simulated: ' + str(n_causal_off) + ' causal eQTLs off a chromatin path, ' + str(n_causal_on) + ' on a path (' + str(n_path) + ' variants on a path)')
