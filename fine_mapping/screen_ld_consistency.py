import sys
import os
import argparse
import numpy as np
from scipy.stats import chi2
# LD-consistency screen of the prepared fine-mapping inputs, run before both fine-mappers so they see identical inputs.
# For each gene and each trait (the eQTL and every linked peak's caQTL):
#   DENTIST-S / SLALOM (Chen et al. 2021; Kanai et al. 2022): for every variant j with r^2 >= min_r2 to the lead variant l,
#     T_j = (z_j - r_jl z_l)^2 / (1 - r_jl^2) ~ chi2_1 if the summary statistics and the LD panel agree; an outlier is p < p_threshold.
#     Uses the LD before the s-regularization (undone from the recorded s), since shrinking r biases T at strong signals.
#   Kriging residual (optional, Zou et al. 2022): t_j = (Omega z)_j / sqrt(Omega_jj) with Omega = R^-1, the standardized
#     residual of z_j against its prediction from all other variants. Reported only; it is also large at isolated strong causal variants.
# A trait fails if it has more than max_outliers outliers. Failing peaks are dropped from the gene (arrays rewritten with an
# _ld_screened suffix into screened_arrays_dir); genes with a failing eQTL are dropped only with --drop_failed_eqtl_genes. Imputed caQTL cells are not tested.
# Output: the screened summary (same columns as the input plus Kept_Peak_Indices) and a per-gene, per-trait diagnostics table.

parser = argparse.ArgumentParser()
parser.add_argument('--fm_input_summary_file', type=str)
parser.add_argument('--screened_summary_file', type=str)
parser.add_argument('--diagnostics_file', type=str)
parser.add_argument('--screened_arrays_dir', type=str, default=None)  # where the rewritten per-peak arrays go; default: the directory of the screened summary file (so reruns never touch the prepared inputs)
parser.add_argument('--min_r2', type=float, default=0.6)  # variants tested against the lead: r^2 to the lead at least this
parser.add_argument('--dentist_p_threshold', type=float, default=1e-4)
parser.add_argument('--max_outliers', type=int, default=0)  # a trait fails if it has more outliers than this
parser.add_argument('--drop_failed_eqtl_genes', action='store_true', default=False)
parser.add_argument('--no_drop_failed_peaks', action='store_true', default=False)  # annotate only
parser.add_argument('--no_kriging', action='store_true', default=False)  # skip the (p x p inverse) kriging residuals
parser.add_argument('--kriging_threshold', type=float, default=4.0)  # |t| above this is counted in the diagnostics
args = parser.parse_args()
dentist_T_threshold = chi2.isf(args.dentist_p_threshold, 1)


def dentist_s(z, R, tested, min_r2):
    # Outliers against the lead variant among `tested` variants. Returns (lead index, n tested partners, n outliers, max T, outlier indices)
    if np.sum(tested) == 0:
        return -1, 0, 0, 0.0, np.zeros(0, dtype=int)
    lead = int(np.argmax(np.where(tested, np.abs(z), -np.inf)))
    r = np.clip(R[:, lead], -1.0, 1.0)
    partners = tested & (r**2 >= min_r2)
    partners[lead] = False
    if np.sum(partners) == 0:
        return lead, 0, 0, 0.0, np.zeros(0, dtype=int)
    T = (z[partners] - r[partners] * z[lead])**2 / np.maximum(1.0 - r[partners]**2, 0.01)
    idx = np.where(partners)[0]
    outliers = idx[T > dentist_T_threshold]
    return lead, int(np.sum(partners)), len(outliers), float(np.max(T)), outliers


def kriging_residuals(Omega, Z):
    # Z: p x m columns of z-scores. t = (Omega Z) / sqrt(diag(Omega)) per column
    return np.dot(Omega, Z) / np.sqrt(np.diag(Omega))[:, None]


screened_arrays_dir = args.screened_arrays_dir if args.screened_arrays_dir is not None else os.path.dirname(os.path.abspath(args.screened_summary_file))
if not os.path.exists(screened_arrays_dir):
    os.makedirs(screened_arrays_dir)
f = open(args.fm_input_summary_file)
t_sum = open(args.screened_summary_file, 'w')
t_diag = open(args.diagnostics_file, 'w')
t_diag.write('gene_id\ttrait\tlead_variant\tlead_z\tn_partners_tested\tn_dentist_outliers\tmax_dentist_T\tkriging_max_abs_t\tkriging_n_above_threshold\tfail\taction\n')
head_count = 0
n_genes = 0
n_genes_written = 0
n_genes_eqtl_fail = 0
n_peaks = 0
n_peaks_fail = 0
for line in f:
    data = line.rstrip().split('\t')
    if head_count == 0:
        head_count += 1
        header = data
        t_sum.write('\t'.join(header + ['Kept_Peak_Indices']) + '\n')
        continue
    n_genes += 1
    gene_id = data[0]
    variant_ids = np.loadtxt(data[1], dtype=str, ndmin=1)
    R_reg = np.load(data[2]).astype(float)
    p = R_reg.shape[0]
    zE = np.load(data[3]) / np.load(data[4])
    bA = np.atleast_2d(np.load(data[5]))
    sA = np.atleast_2d(np.load(data[6]))
    K = bA.shape[0]
    zA = bA / sA if K > 0 else np.zeros((0, p))
    has_extra = len(data) >= 13
    imputed = np.atleast_2d(np.load(data[9])).astype(bool) if (has_extra and K > 0) else np.zeros((K, p), dtype=bool)
    s = float(data[12]) if has_extra else 0.0
    # LD before regularization (R_reg = (1 - s) R + s I); the PSD projection is kept
    R = (R_reg - s * np.eye(p)) / (1.0 - s) if s > 0.0 else R_reg
    np.fill_diagonal(R, 1.0)

    # Kriging residuals against all other variants (uses the regularized LD the models use)
    kr_max = np.full(K + 1, np.nan)
    kr_n = np.full(K + 1, -1)
    if not args.no_kriging:
        Omega = np.linalg.inv(R_reg)
        Z = np.column_stack([zE[:, None], zA.T]) if K > 0 else zE[:, None]
        tk = kriging_residuals(Omega, Z)
        for c in range(K + 1):
            typed = np.ones(p, dtype=bool) if c == 0 else ~imputed[c - 1]
            vals = np.abs(tk[typed, c])
            kr_max[c] = np.max(vals) if len(vals) > 0 else np.nan
            kr_n[c] = int(np.sum(vals > args.kriging_threshold))
        del Omega

    # DENTIST-S per trait
    lead, n_part, n_out, maxT, _ = dentist_s(zE, R, np.ones(p, dtype=bool), args.min_r2)
    eqtl_fail = n_out > args.max_outliers
    n_genes_eqtl_fail += int(eqtl_fail)
    eqtl_action = 'gene dropped' if (eqtl_fail and args.drop_failed_eqtl_genes) else ('flagged' if eqtl_fail else 'kept')
    t_diag.write('\t'.join(map(str, [gene_id, 'eQTL', variant_ids[lead] if lead >= 0 else 'NA', zE[lead] if lead >= 0 else 'NA', n_part, n_out, maxT, kr_max[0], kr_n[0], eqtl_fail, eqtl_action])) + '\n')
    keep_peaks = np.ones(K, dtype=bool)
    for k in range(K):
        n_peaks += 1
        lead, n_part, n_out, maxT, _ = dentist_s(zA[k], R, ~imputed[k], args.min_r2)
        peak_fail = n_out > args.max_outliers
        n_peaks_fail += int(peak_fail)
        if peak_fail and not args.no_drop_failed_peaks:
            keep_peaks[k] = False
        t_diag.write('\t'.join(map(str, [gene_id, 'caQTL_peak' + str(k), variant_ids[lead] if lead >= 0 else 'NA', zA[k, lead] if lead >= 0 else 'NA', n_part, n_out, maxT, kr_max[k + 1], kr_n[k + 1], peak_fail, 'peak dropped' if (peak_fail and not args.no_drop_failed_peaks) else ('flagged' if peak_fail else 'kept')])) + '\n')
    t_diag.flush()

    if eqtl_fail and args.drop_failed_eqtl_genes:
        continue
    row = list(data)
    if np.sum(keep_peaks) < K:
        # Rewrite the per-peak arrays without the failing peaks (files with an _ld_screened suffix, in screened_arrays_dir)
        for col in [5, 6, 7, 8] + ([9, 10] if has_extra else []):
            arr = np.load(data[col])
            arr = np.atleast_2d(arr)[keep_peaks] if col in [5, 6, 9, 10] else arr[keep_peaks]
            new_file = os.path.join(screened_arrays_dir, os.path.splitext(os.path.basename(data[col]))[0] + '_ld_screened.npy')
            np.save(new_file, arr)
            row[col] = new_file
    row.append(','.join(map(str, np.where(keep_peaks)[0])) if K > 0 else '')
    t_sum.write('\t'.join(row) + '\n')
    n_genes_written += 1
    if n_genes % 500 == 0:
        print('gene ' + str(n_genes), flush=True)
f.close()
t_sum.close()
t_diag.close()
print(str(n_genes) + ' genes screened: ' + str(n_genes_eqtl_fail) + ' with an LD-inconsistent eQTL (' + ('dropped' if args.drop_failed_eqtl_genes else 'flagged, kept') + '); ' + str(n_peaks_fail) + ' of ' + str(n_peaks) + ' peaks with an LD-inconsistent caQTL (' + ('flagged, kept' if args.no_drop_failed_peaks else 'dropped') + '); ' + str(n_genes_written) + ' genes written to ' + args.screened_summary_file, flush=True)
