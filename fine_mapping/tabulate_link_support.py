import sys
import os
import argparse
import numpy as np
# For every link the mediated model kept: the hurdle estimate, the peak's caQTL lead variant, the caQTL and eQTL z-scores there,
# the implied eQTL z (hurdle estimate / lambda x caQTL z at the lead), and the model's link and inclusion probabilities.
# Then a summary of how many links are testable (|implied z| above a threshold) and, among those, how many have an observed
# eQTL z of the same sign and of comparable size, or of the opposite sign. Answers "are there strong links the eQTL contradicts?"

parser = argparse.ArgumentParser()
parser.add_argument('--mediated_results_file', type=str)  # its _components.txt and shared hyperparameter file are located from this
parser.add_argument('--shared_hyperparameter_file', type=str)
parser.add_argument('--fm_input_summary_file', type=str)  # the (screened) summary the mediated runner was given
parser.add_argument('--output_file', type=str)
parser.add_argument('--testable_z', type=float, default=2.0)  # |implied eQTL z| above this counts as testable
parser.add_argument('--calibration_min_mediates_prob', type=float, default=0.5)  # links with P(mediates) above this enter the lambda calibration (weighted by it)
parser.add_argument('--eqtl_noise_variance', type=float, default=1.0)  # noise variance of an eQTL z-score (1 in z-score mode; the estimated residual variance if that option was used)
args = parser.parse_args()

hyper = dict(zip(*[l.rstrip().split('\t') for l in open(args.shared_hyperparameter_file).read().rstrip().split('\n')[:2]]))
lam = float(hyper['link_scale'])

fm_rows = {}
f = open(args.fm_input_summary_file)
header = None
for line in f:
    data = line.rstrip().split('\t')
    if header is None:
        header = data
        continue
    fm_rows[data[0]] = dict(zip(header, data))
f.close()

components_file = os.path.splitext(args.mediated_results_file)[0] + '_components.txt'
t = open(args.output_file, 'w')
t.write('gene_id\tlink_index\tpeak_index_in_inputs\tlink_estimate\tlink_se\tlink_z\tcaqtl_lead_variant\tcaqtl_lead_z\teqtl_z_at_lead\timplied_eqtl_z\tlink_nonzero_prob\tlink_mediates_prob\tcaqtl_pip_at_lead\ttestable\tsame_sign\n')
n_links = 0
n_testable = 0
n_same = 0
n_opposite = 0
n_same_comparable = 0
n_testable_included = 0
n_opposite_included = 0
cal_y = []   # hurdle estimate x caQTL z at the lead (noise-free side of z_E = (ghat / lambda) z_caQTL + noise)
cal_x = []   # observed eQTL z at the lead
cal_w = []   # P(mediates)
f = open(components_file)
header = None
for line in f:
    data = line.rstrip().split('\t')
    if header is None:
        header = data
        continue
    row = dict(zip(header, data))
    g = row['gene_id']
    if row['link_probs'] == '' or g not in fm_rows:
        continue
    link_probs = np.array(row['link_probs'].split(','), dtype=float)
    incl = np.array(row['link_inclusion_probs'].split(','), dtype=float) if row.get('link_inclusion_probs', '') != '' else link_probs
    kept = np.array(row['kept_link_indices'].split(','), dtype=int)
    caqtl_pip = np.load(row['caqtl_pip_file'])
    fm = fm_rows[g]
    variant_ids = np.loadtxt(fm['Variant_IDs_File'], dtype=str, ndmin=1)
    zE = np.load(fm['eQTL_Effects_File']) / np.load(fm['eQTL_SE_File'])
    bA = np.atleast_2d(np.load(fm['caQTL_Effects_File']))
    sA = np.atleast_2d(np.load(fm['caQTL_SE_File']))
    zA = bA / sA
    ghat = np.load(fm['Estimated_Peak_Gene_Effects_File'])
    gse = np.load(fm['Estimated_Peak_Gene_SE_File'])
    imputed = np.atleast_2d(np.load(fm['caQTL_Imputed_Mask_File'])).astype(bool) if 'caQTL_Imputed_Mask_File' in fm and os.path.exists(fm['caQTL_Imputed_Mask_File']) else np.zeros(bA.shape, dtype=bool)
    for k_fit in range(len(link_probs)):
        k = int(kept[k_fit])
        z = zA[k]
        typed = ~imputed[k]
        lead = int(np.argmax(np.where(typed, np.abs(z), -np.inf)))
        implied = ghat[k] / lam * z[lead]
        testable = abs(implied) > args.testable_z
        same_sign = np.sign(implied) == np.sign(zE[lead])
        n_links += 1
        if incl[k_fit] > args.calibration_min_mediates_prob:
            cal_y.append(ghat[k] * z[lead])
            cal_x.append(zE[lead])
            cal_w.append(incl[k_fit])
        if testable:
            n_testable += 1
            n_testable_included += incl[k_fit] > 0.5
            if same_sign:
                n_same += 1
                n_same_comparable += abs(zE[lead]) > 0.5 * abs(implied)
            else:
                n_opposite += 1
                n_opposite_included += incl[k_fit] > 0.5
        t.write('\t'.join(map(str, [g, k_fit, k, ghat[k], gse[k], ghat[k] / gse[k], variant_ids[lead], z[lead], zE[lead], implied, link_probs[k_fit], incl[k_fit], caqtl_pip[k_fit, lead], testable, same_sign])) + '\n')
f.close()
t.close()
print('lambda ' + str(lam) + '; ' + str(n_links) + ' links; ' + str(n_testable) + ' testable (|implied eQTL z| > ' + str(args.testable_z) + ')')
print('  testable with observed eQTL z of the same sign: ' + str(n_same) + ' (of which ' + str(n_same_comparable) + ' with |observed| > half |implied|)')
print('  testable with observed eQTL z of the opposite sign: ' + str(n_opposite) + ' (' + str(n_opposite_included) + ' of these have inclusion prob > 0.5)')
print('  testable with inclusion prob > 0.5: ' + str(n_testable_included))
# External calibration of lambda from the mediating peaks: regress y = ghat x z_caQTL(lead) on x = observed eQTL z(lead) through
# the origin, weighted by P(mediates), and correct the slope for the noise in x (variance eqtl_noise_variance per z-score):
#   lambda = sum w y x / (sum w x^2 - sum w sigma2). Outside the variational loop and on marginal statistics, so neither peak stacking
# nor the residual bookkeeping can hide the signal. Biased downward where a direct eQTL effect sits on the caQTL lead (collinearity).
cal_y = np.array(cal_y); cal_x = np.array(cal_x); cal_w = np.array(cal_w)
if len(cal_y) >= 3:
    denom = np.sum(cal_w * cal_x**2) - np.sum(cal_w) * args.eqtl_noise_variance
    lam_naive = np.sum(cal_w * cal_y * cal_x) / np.sum(cal_w * cal_x**2)
    lam_cal = np.sum(cal_w * cal_y * cal_x) / denom if denom > 0 else float('nan')
    # leave-one-out spread as a rough standard error
    loo = []
    for i in range(len(cal_y)):
        m = np.ones(len(cal_y), dtype=bool); m[i] = False
        d_i = np.sum(cal_w[m] * cal_x[m]**2) - np.sum(cal_w[m]) * args.eqtl_noise_variance
        if d_i > 0:
            loo.append(np.sum(cal_w[m] * cal_y[m] * cal_x[m]) / d_i)
    loo = np.array(loo)
    se = np.sqrt((len(loo) - 1) / len(loo) * np.sum((loo - np.mean(loo))**2)) if len(loo) > 2 else float('nan')
    print('lambda calibration on ' + str(len(cal_y)) + ' links with P(mediates) > ' + str(args.calibration_min_mediates_prob) + ' (run used lambda ' + str(round(lam, 4)) + '): naive slope ' + '%.3f' % lam_naive + ', attenuation-corrected lambda ' + '%.3f' % lam_cal + ' (jackknife se ' + '%.3f' % se + ')')
else:
    print('lambda calibration: fewer than 3 links with P(mediates) > ' + str(args.calibration_min_mediates_prob) + '; no estimate')
print('per-link table: ' + args.output_file)
