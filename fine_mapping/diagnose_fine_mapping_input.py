import sys
import os
import argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from susie_rss import SUSIE_RSS
# Scan fine-mapping input genes for problems: non-finite values, non-positive SEs, bad LD (nan / asymmetric / indefinite),
# and per-gene z-score SuSiE behaviour (non-convergence, nan). Writes one row per gene and prints a summary.

parser = argparse.ArgumentParser()
parser.add_argument('--sumstat_summary_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--max_genes', type=int, default=300)  # first max_genes genes in the summary file
parser.add_argument('--susie_max_iter', type=int, default=100)
args = parser.parse_args()

t = open(args.output_file, 'w')
t.write('gene_id\tp\tK\tn_nonfinite_eqtl\tn_eqtl_se_le0\tn_nonfinite_caqtl\tn_caqtl_se_le0\tn_nonfinite_links\tn_link_se_le0\tld_n_nonfinite\tld_symmetric\tld_diag_ok\tld_min_eig\tmax_abs_z_eqtl\tmax_abs_z_caqtl\teqtl_susie_converged\teqtl_susie_n_iter\teqtl_susie_nan\tn_peaks_susie_nonconverged\tn_peaks_susie_nan\n')
f = open(args.sumstat_summary_file)
head_count = 0
n_genes = 0
counts = {}
def bump(key, value):
    counts[key] = counts.get(key, 0) + int(value)
for line in f:
    data = line.rstrip().split('\t')
    if head_count == 0:
        head_count += 1
        continue
    if n_genes >= args.max_genes:
        break
    n_genes += 1
    gene_id = data[0]
    ld = np.load(data[2])
    bE = np.load(data[3]); sE = np.load(data[4]); bA = np.load(data[5]); sA = np.load(data[6]); gh = np.load(data[7]); gs = np.load(data[8])
    p = len(bE); K = bA.shape[0]
    n_nonfinite_eqtl = int(np.sum(~np.isfinite(bE)) + np.sum(~np.isfinite(sE)))
    n_eqtl_se_le0 = int(np.sum(sE <= 0))
    n_nonfinite_caqtl = int(np.sum(~np.isfinite(bA)) + np.sum(~np.isfinite(sA)))
    n_caqtl_se_le0 = int(np.sum(sA <= 0))
    n_nonfinite_links = int(np.sum(~np.isfinite(gh)) + np.sum(~np.isfinite(gs)))
    n_link_se_le0 = int(np.sum(gs <= 0))
    ld_n_nonfinite = int(np.sum(~np.isfinite(ld)))
    ld_symmetric = bool(np.allclose(ld, ld.T, equal_nan=True))
    ld_diag_ok = bool(np.allclose(np.diag(ld), 1.0))
    ld_min_eig = float(np.min(np.linalg.eigvalsh(ld))) if ld_n_nonfinite == 0 else np.nan
    with np.errstate(all='ignore'):
        max_abs_z_eqtl = float(np.nanmax(np.abs(bE / sE)))
        max_abs_z_caqtl = float(np.nanmax(np.abs(bA / sA))) if K > 0 else np.nan
    # z-score SuSiE on the eQTL and on each peak
    eqtl_conv = eqtl_iter = eqtl_nan = 'NA'
    n_peaks_nonconv = n_peaks_nan = 'NA'
    if n_nonfinite_eqtl == 0 and n_eqtl_se_le0 == 0 and ld_n_nonfinite == 0:
        fit = SUSIE_RSS(L=10, max_iter=args.susie_max_iter).fit(bE, sE, ld, None)
        eqtl_conv = fit.converged; eqtl_iter = fit.n_iter; eqtl_nan = bool(np.any(~np.isfinite(fit.pip)))
        n_peaks_nonconv = 0; n_peaks_nan = 0
        if n_nonfinite_caqtl == 0 and n_caqtl_se_le0 == 0:
            for k in range(K):
                fit = SUSIE_RSS(L=5, max_iter=args.susie_max_iter).fit(bA[k], sA[k], ld, None)
                n_peaks_nonconv += int(not fit.converged); n_peaks_nan += int(np.any(~np.isfinite(fit.pip)))
    row = [gene_id, p, K, n_nonfinite_eqtl, n_eqtl_se_le0, n_nonfinite_caqtl, n_caqtl_se_le0, n_nonfinite_links, n_link_se_le0, ld_n_nonfinite, ld_symmetric, ld_diag_ok, ld_min_eig, max_abs_z_eqtl, max_abs_z_caqtl, eqtl_conv, eqtl_iter, eqtl_nan, n_peaks_nonconv, n_peaks_nan]
    t.write('\t'.join(map(str, row)) + '\n'); t.flush()
    bump('genes_with_nonfinite_eqtl', n_nonfinite_eqtl > 0); bump('genes_with_eqtl_se_le0', n_eqtl_se_le0 > 0)
    bump('genes_with_nonfinite_caqtl', n_nonfinite_caqtl > 0); bump('genes_with_caqtl_se_le0', n_caqtl_se_le0 > 0)
    bump('genes_with_nonfinite_links', n_nonfinite_links > 0); bump('genes_with_link_se_le0', n_link_se_le0 > 0)
    bump('genes_with_nonfinite_ld', ld_n_nonfinite > 0); bump('genes_with_asymmetric_ld', not ld_symmetric); bump('genes_with_bad_ld_diag', not ld_diag_ok)
    bump('genes_with_indefinite_ld', (ld_min_eig < -1e-8) if np.isfinite(ld_min_eig) else 0)
    bump('genes_eqtl_susie_nonconverged', eqtl_conv is False); bump('genes_eqtl_susie_nan', eqtl_nan is True)
    bump('genes_with_peak_susie_nonconverged', n_peaks_nonconv not in ('NA', 0)); bump('genes_with_peak_susie_nan', n_peaks_nan not in ('NA', 0))
    if n_genes % 50 == 0:
        print('gene ' + str(n_genes), flush=True)
f.close()
t.close()
print('Scanned ' + str(n_genes) + ' genes:')
for key in counts:
    print('  ' + key + ': ' + str(counts[key]))
