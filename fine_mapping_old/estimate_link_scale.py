import sys
import argparse
import numpy as np
# Estimate the link scale lambda on the z-score scale from the prepared fine-mapping inputs:
#   predicted eQTL z-effect at variant j = sum_k ghat_k * z_caqtl[k, j];  observed = z_eqtl[j]
#   slope of observed on predicted = 1 / lambda  (noise-corrected: var(pred) minus its noise, sum_k ghat_k^2, since var(z noise) = 1)
# Also reports OLS slopes in bins of |predicted| (top bins are least attenuated), as in the exploratory analysis.

parser = argparse.ArgumentParser()
parser.add_argument('--sumstat_summary_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--lambda_only', action='store_true', default=False)  # print only the chosen lambda to stdout (for shell capture)
args = parser.parse_args()

pred = []
obs = []
noise = []
f = open(args.sumstat_summary_file)
head_count = 0
for line in f:
    data = line.rstrip().split('\t')
    if head_count == 0:
        head_count += 1
        continue
    zE = np.load(data[3]) / np.load(data[4])
    bA = np.load(data[5]); sA = np.load(data[6]); ghat = np.load(data[7])
    if bA.shape[0] == 0:
        continue
    zA = bA / sA
    pred.append(ghat @ zA)
    obs.append(zE)
    noise.append(np.ones(len(zE)) * np.sum(ghat**2))
f.close()
pred = np.concatenate(pred); obs = np.concatenate(obs); noise = np.concatenate(noise)
n_pairs = len(pred)

# Noise-corrected slope over all pairs
cov = np.mean(pred * obs) - np.mean(pred) * np.mean(obs)
var_pred = np.var(pred)
slope_naive = cov / var_pred
slope_corrected = cov / (var_pred - np.mean(noise)) if var_pred > np.mean(noise) else np.nan

# OLS slope within bins of |pred| (top bins least attenuated by noise in pred)
lines = ['n_pairs\t' + str(n_pairs), 'slope_naive\t' + str(slope_naive), 'slope_noise_corrected\t' + str(slope_corrected)]
bin_slopes = {}
for lo in [0.0, 2.0, 4.0, 6.0]:
    keep = np.abs(pred) > lo
    if np.sum(keep) > 100:
        p = pred[keep]; o = obs[keep]
        bin_slopes[lo] = (np.mean(p * o) - np.mean(p) * np.mean(o)) / np.var(p)
        lines.append('slope_ols_abs_pred_gt_' + str(lo) + '\t' + str(bin_slopes[lo]) + '\t(n = ' + str(int(np.sum(keep))) + ')')

# Choose lambda: noise-corrected slope if usable, else the top |pred| bin with a positive slope, else the naive slope
if np.isfinite(slope_corrected) and slope_corrected > 0:
    slope_chosen = slope_corrected; source = 'noise_corrected'
else:
    positive_bins = [lo for lo in bin_slopes if np.isfinite(bin_slopes[lo]) and bin_slopes[lo] > 0]
    if len(positive_bins) > 0:
        top = max(positive_bins); slope_chosen = bin_slopes[top]; source = 'ols_abs_pred_gt_' + str(top)
    else:
        slope_chosen = slope_naive; source = 'naive'
if not (np.isfinite(slope_chosen) and slope_chosen > 0):
    print('assumption error: no positive slope of eQTL z on predicted z; cannot estimate lambda')
    sys.exit(1)
lambda_chosen = 1.0 / slope_chosen
lines.append('lambda_source\t' + source)
lines.append('lambda\t' + str(lambda_chosen))
if args.output_file is not None:
    open(args.output_file, 'w').write('\n'.join(lines) + '\n')
if args.lambda_only:
    print(lambda_chosen)
else:
    print('\n'.join(lines))
