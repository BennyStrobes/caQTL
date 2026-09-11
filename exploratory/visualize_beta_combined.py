import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker

# numpy renamed trapz to trapezoid in 2.0
trapezoid = getattr(np, 'trapezoid', None) or np.trapz

SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
GRID_GRAY = '#d9d8d4'

# |z_pred| bins used for the confidence-stratified figures
Z_EDGES = [0, 1, 2, 3, 4, 6, 8, np.inf]
# |z_pred| thresholds used for the small-multiple bin scatter
Z_THRESHOLDS = [0, 2, 4, 6]
# |z_pred| bins used to stratify the lead-variant localization figures
LEAD_Z_EDGES = [0, 1, 2, 3, 5, np.inf]
# n_peaks bins
N_PEAK_EDGES = [1, 2, 3, 6, 11, np.inf]
# Fraction trimmed from each tail of the axes in the effect-size density scatter
SCATTER_TAIL_QUANTILE = 0.0005
# |z| thresholds applied to both the prediction and the observed eQTL in the confident effect-size scatters
SCATTER_Z_THRESHOLDS = [3, 5]


########################
# Data loading
########################
def load_pairs_with_prediction(beta_combined_file):
    # Stream the file in chunks, keeping only variant-gene pairs that have a beta_combined prediction
    cols = ['gene_id', 'beta_eqtl_hat', 'beta_eqtl_se', 'beta_combined', 'se_combined', 'af', 'n_peaks']
    # Unbiased prediction variance (may be negative) is written by newer versions of generate_beta_combined.py
    header = pd.read_csv(beta_combined_file, sep='\t', nrows=0).columns
    has_unbiased = 'var_combined_unbiased' in header
    if has_unbiased:
        cols = cols + ['var_combined_unbiased']
    else:
        print("WARNING: no var_combined_unbiased column; the noise-corrected correlation will use the conservative "
              "se_combined^2, which overstates the noise and can give a negative corrected variance", flush=True)
    chunks = []
    gene_max_chunks = []
    n_total = 0
    for chunk in pd.read_csv(beta_combined_file, sep='\t', usecols=cols, chunksize=5000000, na_values=['NA']):
        n_total += len(chunk)
        # Per-gene max |z_eQTL| over every tested pair, so genes with no predicted variant are still counted as eGenes
        ok = chunk['beta_eqtl_se'] > 0
        gene_max_chunks.append((chunk.loc[ok, 'beta_eqtl_hat'] / chunk.loc[ok, 'beta_eqtl_se']).abs()
                               .groupby(chunk.loc[ok, 'gene_id'].to_numpy()).max())
        chunk = chunk.dropna(subset=['beta_eqtl_hat', 'beta_eqtl_se', 'beta_combined', 'se_combined'])
        chunks.append(chunk)
        print("lines read: %d" % n_total, flush=True)
    gene_max_abs_z_eqtl_all = pd.concat(gene_max_chunks).groupby(level=0).max() if gene_max_chunks else pd.Series(dtype=float)
    if len(chunks) == 0:
        return pd.DataFrame(columns=cols + ['z_eqtl', 'z_pred']), gene_max_abs_z_eqtl_all
    df = pd.concat(chunks, ignore_index=True)
    # Drop pairs with a zero standard error, which would give infinite z-scores
    df = df[(df['beta_eqtl_se'] > 0) & (df['se_combined'] > 0)].reset_index(drop=True)
    df['n_peaks'] = df['n_peaks'].astype(int)
    df['gene_id'] = df['gene_id'].astype('category')
    if not has_unbiased:
        df['var_combined_unbiased'] = df['se_combined'] ** 2
    df['z_eqtl'] = df['beta_eqtl_hat'] / df['beta_eqtl_se']
    df['z_pred'] = df['beta_combined'] / df['se_combined']
    print("variant-gene pairs: %d; with prediction: %d; genes tested: %d" % (n_total, len(df), len(gene_max_abs_z_eqtl_all)), flush=True)
    return df, gene_max_abs_z_eqtl_all


########################
# Statistics helpers
########################
def bin_label(n_bins):
    return {4: 'quartile', 5: 'quintile', 10: 'decile', 100: 'percentile'}.get(n_bins, 'bin')


def equal_count_bins(x, n_bins):
    # Equal-count bins by rank of x; returns bin index 0..n_bins-1 per element
    n = len(x)
    order = np.argsort(x, kind='stable')
    bins = np.empty(n, dtype=np.int64)
    bins[order] = (np.arange(n) * n_bins) // n
    return bins


def bin_means(x, y, bins, n_bins):
    # Mean and 95% CI (1.96 * SEM) of x and y within each bin
    n = np.bincount(bins, minlength=n_bins).astype(float)

    def stats(v):
        s = np.bincount(bins, weights=v, minlength=n_bins)
        ss = np.bincount(bins, weights=v * v, minlength=n_bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = s / n
            var = (ss - n * mean ** 2) / (n - 1)
            ci = 1.96 * np.sqrt(var / n)
        return mean, ci

    x_mean, x_ci = stats(x)
    y_mean, y_ci = stats(y)
    return pd.DataFrame({'bin': np.arange(1, n_bins + 1), 'n': n.astype(int),
                         'x_mean': x_mean, 'x_ci95': x_ci, 'y_mean': y_mean, 'y_ci95': y_ci})


def ols_slope(x, y):
    # Slope (and its SE) of y regressed on x with an intercept
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    xc = x - x.mean()
    yc = y - y.mean()
    sxx = np.dot(xc, xc)
    if sxx == 0:
        return np.nan, np.nan
    slope = np.dot(xc, yc) / sxx
    resid = yc - slope * xc
    se = np.sqrt(np.dot(resid, resid) / (n - 2) / sxx)
    return slope, se


def wilson_ci(k, n):
    # 95% Wilson interval for a proportion: (p, lower, upper)
    if n == 0:
        return np.nan, np.nan, np.nan
    z = 1.96
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return p, center - half, center + half


def two_sided_p_from_z(z):
    # p = 2 * (1 - Phi(|z|)) = erfc(|z| / sqrt(2)); scipy if available, otherwise a vectorized math.erfc
    try:
        from scipy.special import erfc
        return erfc(np.abs(z) / np.sqrt(2))
    except ImportError:
        import math
        erfc_v = np.frompyfunc(math.erfc, 1, 1)
        return erfc_v(np.abs(z) / np.sqrt(2)).astype(np.float64)


def compact(n):
    if n >= 1e6:
        return '%.1fM' % (n / 1e6)
    if n >= 1e3:
        return '%.0fK' % (n / 1e3)
    return str(int(n))


def range_labels(edges, integer=False):
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isinf(hi):
            labels.append('%g+' % lo)
        elif integer:
            labels.append('%g' % lo if hi - lo == 1 else '%g–%g' % (lo, hi - 1))
        else:
            labels.append('%g–%g' % (lo, hi))
    return labels


########################
# Plot helpers
########################
def style_axes(ax):
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(GRID_GRAY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def draw_bin_scatter(ax, summary, n_bins, color=SERIES[0]):
    # Smaller marks once the bins get dense
    markersize = 6.5 if n_bins <= 20 else 4
    capsize = 2 if n_bins <= 20 else 1
    ax.axhline(0, color=GRID_GRAY, linewidth=1, zorder=0)
    ax.axvline(0, color=GRID_GRAY, linewidth=1, zorder=0)
    ax.errorbar(summary['x_mean'], summary['y_mean'], xerr=summary['x_ci95'], yerr=summary['y_ci95'],
                fmt='o', color=color, ecolor=color, elinewidth=0.8, capsize=capsize, capthick=0.8,
                markersize=markersize, markeredgecolor='white', markeredgewidth=0.8, zorder=3)


def summary_path(output_file):
    return output_file.rsplit('.', 1)[0] + '_summary.tsv'


########################
# Figures
########################
def fig_bin_scatter(x, y, n_bins, cell_type, xlabel, ylabel, output_file):
    # Mean y vs mean x within equal-count bins of x
    bins = equal_count_bins(x, n_bins)
    summary = bin_means(x, y, bins, n_bins)
    summary.to_csv(summary_path(output_file), sep='\t', index=False)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    draw_bin_scatter(ax, summary, n_bins)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title('%s cells' % cell_type, fontsize=10, color=INK)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    return summary


def fig_bin_scatter_by_confidence(df, thresholds, n_bins, cell_type, output_file):
    # Small multiples: the bin scatter restricted to increasingly confident predictions (|z_pred| > t)
    z_pred = df['z_pred'].to_numpy()
    x_all = df['beta_combined'].to_numpy()
    y_all = df['beta_eqtl_hat'].to_numpy()

    fig, axes = plt.subplots(1, len(thresholds), figsize=(3.4 * len(thresholds), 3.9))
    rows = []
    for ax, t in zip(axes, thresholds):
        mask = np.abs(z_pred) > t
        x = x_all[mask]
        y = y_all[mask]
        style_axes(ax)
        ax.set_xlabel('Mean predicted eQTL effect', color=INK, fontsize=9)
        if len(x) < n_bins:
            ax.set_title('|z_pred| > %g\nn = %s pairs (too few)' % (t, format(len(x), ',')), fontsize=9, color=INK)
            rows.append((t, len(x), np.nan, np.nan))
            continue
        bins = equal_count_bins(x, n_bins)
        summary = bin_means(x, y, bins, n_bins)
        slope, slope_se = ols_slope(x, y)
        draw_bin_scatter(ax, summary, n_bins)
        ax.set_title('|z_pred| > %g\nn = %s pairs' % (t, format(len(x), ',')), fontsize=9, color=INK)
        ax.text(0.03, 0.97, 'OLS slope = %.3f ± %.3f' % (slope, 1.96 * slope_se), transform=ax.transAxes,
                fontsize=8, color=INK_SECONDARY, va='top')
        rows.append((t, len(x), slope, slope_se))
    axes[0].set_ylabel('Mean observed eQTL effect (β_eQTL)', color=INK, fontsize=9)
    fig.suptitle('%s cells' % cell_type, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)

    summary = pd.DataFrame(rows, columns=['abs_z_pred_threshold', 'n', 'ols_slope', 'ols_slope_se'])
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary


def fig_fraction_by_group(group_idx, group_labels, series, cell_type, xlabel, ylabel, output_file,
                          ref_line=None, ref_label=None):
    # Fraction (with Wilson 95% CI) of an indicator within each group, for one or more series
    # series: list of (name, selection_mask, indicator_mask)
    n_groups = len(group_labels)
    rows = []
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    if ref_line is not None:
        ax.axhline(ref_line, color=GRID_GRAY, linewidth=1, zorder=0)
        if ref_label is not None:
            ax.text(n_groups - 0.6, ref_line, ref_label, fontsize=8, color=INK_SECONDARY, va='bottom', ha='right')

    offsets = np.linspace(-0.12, 0.12, len(series)) if len(series) > 1 else [0.0]
    for (name, selection, indicator), offset, color in zip(series, offsets, SERIES):
        ps, los, his = [], [], []
        for g in range(n_groups):
            in_group = selection & (group_idx == g)
            n = int(in_group.sum())
            k = int((in_group & indicator).sum())
            p, lo, hi = wilson_ci(k, n)
            ps.append(p)
            los.append(lo)
            his.append(hi)
            rows.append((name, group_labels[g], n, k, p, lo, hi))
        ps, los, his = np.array(ps), np.array(los), np.array(his)
        # Wilson intervals need not contain the raw proportion when it is 0 or 1; clip the bar lengths at zero
        yerr = [np.maximum(ps - los, 0), np.maximum(his - ps, 0)]
        ax.errorbar(np.arange(n_groups) + offset, ps, yerr=yerr, fmt='o', color=color, ecolor=color,
                    elinewidth=1, capsize=2, capthick=1, markersize=6.5, markeredgecolor='white', markeredgewidth=1,
                    label=name, zorder=3)

    # Group sizes (first series) under the tick labels
    first_sel = series[0][1]
    tick_labels = ['%s\nn = %s' % (lab, compact(int((first_sel & (group_idx == g)).sum())))
                   for g, lab in enumerate(group_labels)]
    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title('%s cells' % cell_type, fontsize=10, color=INK)
    style_axes(ax)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)

    summary = pd.DataFrame(rows, columns=['series', 'group', 'n', 'k', 'fraction', 'ci95_lower', 'ci95_upper'])
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary


def fig_pvalue_histograms(p_eqtl, p_pred, cell_type, output_file, n_bins=50):
    # Side-by-side p-value histograms with the uniform (all-null) expectation as a reference line
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    rows = []
    panels = [('Observed eQTL', 'eqtl', p_eqtl), ('Combined prediction', 'combined', p_pred)]
    for ax, (name, key, p) in zip(axes, panels):
        counts, edges = np.histogram(p, bins=n_bins, range=(0, 1))
        ax.bar(edges[:-1], counts, width=1.0 / n_bins, align='edge', color=SERIES[0],
               edgecolor='white', linewidth=0.5, zorder=3)
        uniform = len(p) / n_bins
        ax.axhline(uniform, color=GRID_GRAY, linewidth=1, zorder=4)
        ax.text(0.99, uniform, 'uniform', fontsize=8, color=INK_SECONDARY, va='bottom', ha='right')
        ax.set_title('%s (n = %s pairs)' % (name, format(len(p), ',')), fontsize=9, color=INK)
        ax.set_xlabel('Two-sided p-value from z', color=INK, fontsize=9)
        ax.set_xlim(0, 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: compact(v)))
        style_axes(ax)
        for lo, hi, c in zip(edges[:-1], edges[1:], counts):
            rows.append((key, lo, hi, int(c)))
    axes[0].set_ylabel('Number of variant-gene pairs', color=INK, fontsize=9)
    fig.suptitle('%s cells' % cell_type, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)

    summary = pd.DataFrame(rows, columns=['series', 'p_lower', 'p_upper', 'n'])
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary


def fig_effect_size_density(x, y, cell_type, output_file, n_bins=400, note=None):
    # Scatter of every pair rendered as a 2D histogram with a log color scale (a point scatter of
    # tens of millions of pairs is unreadable and enormous); axes trimmed to the central quantile range
    q = SCATTER_TAIL_QUANTILE
    xlim = np.quantile(x, [q, 1 - q])
    ylim = np.quantile(y, [q, 1 - q])
    counts, xedges, yedges = np.histogram2d(x, y, bins=n_bins, range=[xlim, ylim])
    n_outside = len(x) - int(counts.sum())

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seq_blue', ['#d6e6f7', SERIES[0], '#0a2d5e'])
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.axhline(0, color=GRID_GRAY, linewidth=1, zorder=0)
    ax.axvline(0, color=GRID_GRAY, linewidth=1, zorder=0)
    masked = np.ma.masked_where(counts.T == 0, counts.T)
    im = ax.imshow(masked, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto',
                   cmap=cmap, norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(counts.max(), 2)),
                   interpolation='nearest', zorder=2)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Variant-gene pairs per cell', color=INK, fontsize=9)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    cbar.outline.set_edgecolor(GRID_GRAY)
    ax.set_xlabel('Predicted eQTL effect (Σ β_caQTL × β_link)', color=INK)
    ax.set_ylabel('Observed eQTL effect (β_eQTL)', color=INK)
    ax.set_title('%s cells' % cell_type, fontsize=10, color=INK)
    if note is not None:
        ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, va='top', zorder=5)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print("effect-size scatter: %d pairs shown; %d outside the trimmed axis range (x %.3g to %.3g, y %.3g to %.3g)"
          % (int(counts.sum()), n_outside, xlim[0], xlim[1], ylim[0], ylim[1]), flush=True)



def roc_pr_curve(score, label):
    # ROC and precision-recall curves from a ranking score (higher = more likely positive), tie groups collapsed
    order = np.argsort(-score, kind='stable')
    s = score[order]
    lab = label[order]
    tp = np.cumsum(lab)
    fp = np.cumsum(~lab)
    last_of_tie = np.r_[s[1:] != s[:-1], True]
    tp = tp[last_of_tie].astype(float)
    fp = fp[last_of_tie].astype(float)
    n_pos = tp[-1]
    n_neg = fp[-1]
    tpr = np.r_[0.0, tp / n_pos]
    fpr = np.r_[0.0, fp / n_neg]
    precision = tp / (tp + fp)
    auroc = trapezoid(tpr, fpr)
    auprc = trapezoid(precision, tp / n_pos)
    return fpr, tpr, tp / n_pos, precision, auroc, auprc


def thin(*arrays, n=4000):
    # Keep at most n evenly spaced points from each (equal-length) array, for plotting
    m = len(arrays[0])
    idx = np.unique(np.linspace(0, m - 1, min(n, m)).astype(int))
    return [a[idx] for a in arrays]


def fig_roc_pr(scores, label, label_name, cell_type, output_file):
    # ROC and precision-recall for detecting label using each score; scores: list of (name, array)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(8.6, 4.0))
    base_rate = label.mean()
    ax_roc.plot([0, 1], [0, 1], color=GRID_GRAY, linewidth=1, zorder=0)
    ax_pr.axhline(base_rate, color=GRID_GRAY, linewidth=1, zorder=0)
    ax_pr.text(0.99, base_rate, 'base rate', fontsize=8, color=INK_SECONDARY, va='bottom', ha='right')
    rows = []
    for (name, score), color in zip(scores, SERIES):
        fpr, tpr, recall, precision, auroc, auprc = roc_pr_curve(score, label)
        fpr_t, tpr_t = thin(fpr, tpr)
        recall_t, precision_t = thin(recall, precision)
        ax_roc.plot(fpr_t, tpr_t, color=color, linewidth=1.5, label='%s (AUC = %.3f)' % (name, auroc), zorder=3)
        ax_pr.plot(recall_t, precision_t, color=color, linewidth=1.5, label='%s (AUC = %.3f)' % (name, auprc), zorder=3)
        rows.append((name, int(label.sum()), int((~label).sum()), auroc, auprc))
    ax_roc.set_xlabel('False positive rate', color=INK, fontsize=9)
    ax_roc.set_ylabel('True positive rate', color=INK, fontsize=9)
    ax_pr.set_xlabel('Recall', color=INK, fontsize=9)
    ax_pr.set_ylabel('Precision', color=INK, fontsize=9)
    ax_pr.set_yscale('log')
    for ax in (ax_roc, ax_pr):
        ax.set_xlim(0, 1)
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc='lower right' if ax is ax_roc else 'upper right')
    ax_roc.set_ylim(0, 1)
    fig.suptitle('%s cells: detecting %s (n = %s positive of %s pairs)'
                 % (cell_type, label_name, format(int(label.sum()), ','), format(len(label), ',')), fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)

    summary = pd.DataFrame(rows, columns=['score', 'n_positive', 'n_negative', 'auroc', 'auprc'])
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary


def per_gene_table(df, gene_max_abs_z_eqtl_all):
    # One row per gene with at least one predicted variant:
    #   n_variants          number of variants with a prediction
    #   lead_*              the eQTL lead among predicted variants (largest |z_eQTL|)
    #   max_abs_z_pred      largest |z_pred| in the gene; top_pred_abs_z_eqtl is |z_eQTL| at that variant
    #   n_better_predicted  number of variants ranked above the lead by |z_pred|; percentile = rank / (n - 1)
    #   spearman            Spearman correlation of beta_pred vs beta_eQTL across the gene's variants
    #   max_abs_z_eqtl_all  largest |z_eQTL| over every tested pair for the gene, predicted or not
    # Group on integer category codes (cheap); map back to gene names at the end
    key = df['gene_id'].cat.codes.to_numpy()
    gene_names = np.asarray(df['gene_id'].cat.categories)
    g = pd.DataFrame({'gene': key, 'abs_z_eqtl': np.abs(df['z_eqtl'].to_numpy()), 'abs_z_pred': np.abs(df['z_pred'].to_numpy()),
                      'beta_pred': df['beta_combined'].to_numpy(), 'beta_eqtl': df['beta_eqtl_hat'].to_numpy(),
                      'n_peaks': df['n_peaks'].to_numpy()})
    grouped = g.groupby('gene', sort=False)
    n_var = grouped.size()
    lead = g.loc[grouped['abs_z_eqtl'].idxmax()].set_index('gene')
    top = g.loc[grouped['abs_z_pred'].idxmax()].set_index('gene')
    lead_pred_per_row = g['gene'].map(lead['abs_z_pred']).to_numpy(dtype=float)
    n_better = pd.Series(g['abs_z_pred'].to_numpy() > lead_pred_per_row).groupby(key, sort=False).sum()

    # Spearman via ranks within gene and per-gene sums
    rx = grouped['beta_pred'].rank().to_numpy()
    ry = grouped['beta_eqtl'].rank().to_numpy()
    def gsum(v):
        return pd.Series(v).groupby(key, sort=False).sum()
    sx, sy, sxx, syy, sxy = gsum(rx), gsum(ry), gsum(rx * rx), gsum(ry * ry), gsum(rx * ry)
    n = n_var.reindex(sx.index).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = (sxy - sx * sy / n) / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))

    genes = pd.DataFrame({'n_variants': n_var, 'lead_abs_z_eqtl': lead['abs_z_eqtl'], 'lead_abs_z_pred': lead['abs_z_pred'],
                          'max_abs_z_pred': top['abs_z_pred'], 'top_pred_abs_z_eqtl': top['abs_z_eqtl'],
                          'max_n_peaks': grouped['n_peaks'].max(), 'n_better_predicted': n_better.reindex(n_var.index),
                          'spearman': rho.reindex(n_var.index)})
    with np.errstate(divide='ignore', invalid='ignore'):
        genes['percentile'] = np.where(genes['n_variants'] > 1, genes['n_better_predicted'] / (genes['n_variants'] - 1), np.nan)
    genes.index = pd.Index(gene_names[genes.index.to_numpy()], name='gene_id')
    genes['max_abs_z_eqtl_all'] = gene_max_abs_z_eqtl_all.reindex(genes.index).to_numpy()
    return genes


def fig_lead_variant_rank(genes, cell_type, output_file, min_variants=20, egene_z=4, n_bins=20):
    # For each gene, the percentile rank of its eQTL lead variant (largest |z_eQTL|) by |z_pred| among the gene's
    # predicted variants (0 = lead is also the top predicted variant). Under the null the percentile is uniform.
    # Genes are split by whether the lead is a confident eQTL (|z_eQTL| > egene_z).
    genes = genes[genes['n_variants'] >= min_variants]
    is_egene = genes['lead_abs_z_eqtl'] > egene_z
    series = [('Lead |z_eQTL| > %g' % egene_z, is_egene), ('Lead |z_eQTL| ≤ %g' % egene_z, ~is_egene)]
    edges = np.linspace(0, 1, n_bins + 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.axhline(1.0 / n_bins, color=GRID_GRAY, linewidth=1, zorder=0)
    ax.text(0.99, 1.0 / n_bins, 'uniform', fontsize=8, color=INK_SECONDARY, va='bottom', ha='right', transform=ax.get_yaxis_transform())
    rows = []
    width = 1.0 / n_bins
    for k, ((name, sel), color) in enumerate(zip(series, SERIES)):
        pct = genes.loc[sel, 'percentile'].to_numpy()
        if len(pct) == 0:
            continue
        counts, _ = np.histogram(np.clip(pct, 0, 1 - 1e-12), bins=edges)
        frac = counts / len(pct)
        ax.bar(edges[:-1] + k * width / 2, frac, width=width / 2, align='edge', color=color, edgecolor='white', linewidth=0.5,
               label='%s (n = %s genes; top 5%%: %.1f%%, top 10%%: %.1f%%)'
               % (name, format(len(pct), ','), 100 * (pct <= 0.05).mean(), 100 * (pct <= 0.10).mean()), zorder=3)
        for lo, hi, c, f in zip(edges[:-1], edges[1:], counts, frac):
            rows.append((name, lo, hi, int(c), f))
    ax.set_xlim(0, 1)
    ax.set_xlabel('Percentile rank of eQTL lead variant by |z_pred| within gene (0 = top predicted)', color=INK, fontsize=9)
    ax.set_ylabel('Fraction of genes', color=INK, fontsize=9)
    ax.set_title('%s cells (genes with ≥ %d predicted variants)' % (cell_type, min_variants), fontsize=10, color=INK)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)

    summary = pd.DataFrame(rows, columns=['series', 'percentile_lower', 'percentile_upper', 'n_genes', 'fraction'])
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary


def fig_gene_overlap(genes, cell_type, output_file, egene_z=4):
    # For each gene, classify by where the eQTL is relative to the gene's top predicted variant, stacked by max |z_pred| bin
    cats = ['Top predicted variant is an eQTL', 'eQTL elsewhere in gene', 'No eQTL in gene']
    top_is_eqtl = (genes['top_pred_abs_z_eqtl'] > egene_z).to_numpy()
    gene_has_eqtl = (genes['max_abs_z_eqtl_all'] > egene_z).to_numpy()
    cat = np.where(top_is_eqtl, 0, np.where(gene_has_eqtl, 1, 2))
    group = np.digitize(genes['max_abs_z_pred'].to_numpy(), LEAD_Z_EDGES[1:-1])
    labels = range_labels(LEAD_Z_EDGES)
    n_groups = len(labels)
    counts = np.array([[int(((group == g) & (cat == c)).sum()) for c in range(3)] for g in range(n_groups)], dtype=float)
    n = counts.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = counts / n[:, None]

    colors = [SERIES[0], SERIES[2], GRID_GRAY]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bottom = np.zeros(n_groups)
    x = np.arange(n_groups)
    for c in range(3):
        ax.bar(x, frac[:, c], bottom=bottom, color=colors[c], edgecolor='white', linewidth=0.8, width=0.7, label=cats[c], zorder=3)
        for xi, f, b in zip(x, frac[:, c], bottom):
            if f > 0.04:
                ax.text(xi, b + f / 2, '%.0f%%' % (100 * f), ha='center', va='center', fontsize=7.5,
                        color='white' if c < 2 else INK_SECONDARY, zorder=4)
        bottom += frac[:, c]
    ax.set_xticks(x)
    ax.set_xticklabels(['%s\nn = %s' % (lab, compact(int(k))) for lab, k in zip(labels, n)])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Maximum |z_pred| across the gene's predicted variants", color=INK)
    ax.set_ylabel('Fraction of genes', color=INK)
    ax.set_title('%s cells (eQTL: |z_eQTL| > %g)' % (cell_type, egene_z), fontsize=10, color=INK)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)

    rows = [(labels[g], int(n[g]), cats[c], int(counts[g, c]), frac[g, c]) for g in range(n_groups) for c in range(3)]
    summary = pd.DataFrame(rows, columns=['max_abs_z_pred_bin', 'n_genes', 'category', 'n', 'fraction'])
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary


def ecdf_step(ax, v, color, label, log=False):
    v = np.sort(np.asarray(v, dtype=float))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where='post', color=color, linewidth=1.5, label=label, zorder=3)
    if log:
        ax.set_xscale('log')


def fig_egene_coverage(genes, gene_max_abs_z_eqtl_all, cell_type, output_file, egene_z=4, confident_z=3):
    # Why are most eGenes silent? Left: every tested gene split into no predicted variant / silent prediction /
    # confident prediction, for eGenes and non-eGenes. Middle and right: number of predicted variants and max n_peaks
    # per gene for silent vs confident eGenes.
    all_genes = gene_max_abs_z_eqtl_all.index
    is_egene = (gene_max_abs_z_eqtl_all > egene_z).to_numpy()
    max_pred = genes['max_abs_z_pred'].reindex(all_genes).to_numpy()
    status = np.where(np.isnan(max_pred), 0, np.where(max_pred > confident_z, 2, 1))
    status_labels = ['No predicted variant', 'Silent (max |z_pred| ≤ %g)' % confident_z, 'Confident (max |z_pred| > %g)' % confident_z]

    fig, (ax_s, ax_v, ax_p) = plt.subplots(1, 3, figsize=(13.0, 4.1))
    rows = []
    groups = [('eGenes', is_egene), ('Non-eGenes', ~is_egene)]
    x = np.arange(3)
    for k, ((name, sel), color) in enumerate(zip(groups, SERIES)):
        n = int(sel.sum())
        cnt = np.array([int((sel & (status == st)).sum()) for st in range(3)])
        frac = cnt / n if n > 0 else np.zeros(3)
        ax_s.bar(x + (k - 0.5) * 0.36, frac, width=0.36, color=color, edgecolor='white', linewidth=0.8,
                 label='%s (n = %s)' % (name, format(n, ',')), zorder=3)
        for xi, f, c in zip(x + (k - 0.5) * 0.36, frac, cnt):
            ax_s.text(xi, f, compact(c), ha='center', va='bottom', fontsize=7, color=INK_SECONDARY)
        for st in range(3):
            rows.append((name, status_labels[st], int(cnt[st]), frac[st]))
    ax_s.set_xticks(x)
    ax_s.set_xticklabels([l.replace(' (', '\n(') for l in status_labels], fontsize=7.5)
    ax_s.set_ylabel('Fraction of genes', color=INK, fontsize=9)
    ax_s.set_title('Prediction status by gene (eGene: |z_eQTL| > %g anywhere)' % egene_z, fontsize=9, color=INK)
    ax_s.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    style_axes(ax_s)

    egene_pred = genes[genes['max_abs_z_eqtl_all'] > egene_z]
    silent = egene_pred[egene_pred['max_abs_z_pred'] <= confident_z]
    confident = egene_pred[egene_pred['max_abs_z_pred'] > confident_z]
    for ax, col, xlabel, log in [(ax_v, 'n_variants', 'Predicted variants per gene', True),
                                 (ax_p, 'max_n_peaks', 'Max peaks contributing to a prediction', False)]:
        for (name, sub), color in zip([('Silent eGenes', silent), ('Confident eGenes', confident)], [SERIES[1], SERIES[0]]):
            if len(sub) == 0:
                continue
            ecdf_step(ax, sub[col], color, '%s (n = %s, median = %g)' % (name, format(len(sub), ','), np.median(sub[col])), log=log)
        ax.set_xlabel(xlabel, color=INK, fontsize=9)
        ax.set_ylabel('Cumulative fraction of eGenes', color=INK, fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY, loc='lower right')
        style_axes(ax)
    fig.suptitle('%s cells' % cell_type, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)

    summary = pd.DataFrame(rows, columns=['group', 'status', 'n', 'fraction'])
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary


def fig_gene_correlation(genes, cell_type, output_file, min_variants=20, egene_z=4, n_bins=20):
    # Per-gene Spearman correlation between predicted and observed effects across the gene's variants,
    # eGenes vs non-eGenes (the latter is the null: no true effect to track)
    genes = genes[(genes['n_variants'] >= min_variants) & genes['spearman'].notna()]
    is_egene = genes['lead_abs_z_eqtl'] > egene_z
    series = [('Lead |z_eQTL| > %g' % egene_z, is_egene), ('Lead |z_eQTL| ≤ %g' % egene_z, ~is_egene)]
    edges = np.linspace(-1, 1, n_bins + 1)
    width = edges[1] - edges[0]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.axvline(0, color=GRID_GRAY, linewidth=1, zorder=0)
    rows = []
    for (name, sel), color in zip(series, SERIES):
        r = genes.loc[sel, 'spearman'].to_numpy()
        if len(r) == 0:
            continue
        counts, _ = np.histogram(r, bins=edges)
        dens = counts / len(r) / width
        ax.step(edges, np.r_[dens, dens[-1]], where='post', color=color, linewidth=1.5,
                label='%s (n = %s; median = %.2f; ρ > 0.3: %.0f%%)' % (name, format(len(r), ','), np.median(r), 100 * (r > 0.3).mean()), zorder=3)
        for lo, hi, c, d in zip(edges[:-1], edges[1:], counts, dens):
            rows.append((name, lo, hi, int(c), d))
    ax.set_xlim(-1, 1)
    ax.set_xlabel('Spearman correlation of β_pred vs β_eQTL across variants within gene', color=INK, fontsize=9)
    ax.set_ylabel('Density of genes', color=INK, fontsize=9)
    ax.set_title('%s cells (genes with ≥ %d predicted variants)' % (cell_type, min_variants), fontsize=10, color=INK)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY, loc='upper left')
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)

    summary = pd.DataFrame(rows, columns=['series', 'rho_lower', 'rho_upper', 'n_genes', 'density'])
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary

def fig_undiscovered_pairs(z_pred, z_eqtl, beta_pred, cell_type, output_file, uncalled_z=4, n_hist_bins=40):
    # Restricted to pairs the eQTL study did not call (|z_eQTL| < uncalled_z): does a confident prediction
    # mark pairs with real but sub-threshold eQTL signal? Left: z_eQTL oriented to the predicted direction
    # (positive = same sign as the prediction), for confident vs low-confidence predictions, against a
    # standard normal. Middle / right: sign concordance and the rate of 2 < |z_eQTL| < uncalled_z by |z_pred| bin.
    uncalled = np.abs(z_eqtl) < uncalled_z
    abs_zp = np.abs(z_pred)
    oriented = z_eqtl * np.sign(beta_pred)
    groups = [('|z_pred| < 1 (control)', abs_zp < 1), ('3 < |z_pred| ≤ 5', (abs_zp > 3) & (abs_zp <= 5)), ('|z_pred| > 5', abs_zp > 5)]

    fig, (ax_h, ax_c, ax_m) = plt.subplots(1, 3, figsize=(13.0, 4.1))

    # (a) oriented z_eQTL densities
    edges = np.linspace(-uncalled_z, uncalled_z, n_hist_bins + 1)
    width = edges[1] - edges[0]
    hist_rows = []
    for (name, sel), color in zip(groups, SERIES):
        m = uncalled & sel
        n = int(m.sum())
        if n == 0:
            continue
        counts, _ = np.histogram(oriented[m], bins=edges)
        dens = counts / n / width
        ax_h.step(edges, np.r_[dens, dens[-1]], where='post', color=color, linewidth=1.4,
                  label='%s (n = %s, mean = %.2f)' % (name, compact(n), oriented[m].mean()), zorder=3)
        for lo, hi, c, d in zip(edges[:-1], edges[1:], counts, dens):
            hist_rows.append((name, lo, hi, int(c), d))
    xs = np.linspace(-uncalled_z, uncalled_z, 400)
    p_inside = 1 - two_sided_p_from_z(np.array([uncalled_z]))[0]
    ax_h.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi) / p_inside, color=GRID_GRAY, linewidth=1.2, label='N(0,1)', zorder=2)
    ax_h.axvline(0, color=GRID_GRAY, linewidth=1, zorder=0)
    ax_h.set_xlabel('z_eQTL oriented to the predicted direction', color=INK, fontsize=9)
    ax_h.set_ylabel('Density', color=INK, fontsize=9)
    ax_h.set_xlim(-uncalled_z, uncalled_z)
    ax_h.legend(frameon=False, fontsize=7, labelcolor=INK_SECONDARY, loc='upper left')
    style_axes(ax_h)

    # (b), (c) fractions by |z_pred| bin among uncalled pairs
    z_group = np.digitize(abs_zp, Z_EDGES[1:-1])
    labels = range_labels(Z_EDGES)
    n_groups = len(labels)
    frac_rows = []
    panels = [(ax_c, oriented > 0, 'Fraction with concordant sign', 0.5, 'concordant_sign'),
              (ax_m, np.abs(z_eqtl) > 2, 'Fraction with 2 < |z_eQTL| < %g' % uncalled_z, 0.0455, 'mid_z')]
    for ax, indicator, ylabel, ref, key in panels:
        ps, los, his, ns = [], [], [], []
        for g in range(n_groups):
            m = uncalled & (z_group == g)
            n = int(m.sum())
            k = int((m & indicator).sum())
            p_, lo, hi = wilson_ci(k, n)
            ps.append(p_); los.append(lo); his.append(hi); ns.append(n)
            frac_rows.append((key, labels[g], n, k, p_, lo, hi))
        ps, los, his = np.array(ps), np.array(los), np.array(his)
        ax.axhline(ref, color=GRID_GRAY, linewidth=1, zorder=0)
        ax.text(n_groups - 0.6, ref, 'null', fontsize=8, color=INK_SECONDARY, va='bottom', ha='right')
        ax.errorbar(np.arange(n_groups), ps, yerr=[np.maximum(ps - los, 0), np.maximum(his - ps, 0)], fmt='o',
                    color=SERIES[0], ecolor=SERIES[0], elinewidth=1, capsize=2, capthick=1, markersize=6,
                    markeredgecolor='white', markeredgewidth=1, zorder=3)
        ax.set_xticks(np.arange(n_groups))
        ax.set_xticklabels(['%s\nn = %s' % (lab, compact(n)) for lab, n in zip(labels, ns)], fontsize=7.5)
        ax.set_xlabel('|z_pred|', color=INK, fontsize=9)
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
        style_axes(ax)

    fig.suptitle('%s cells: pairs not called by the eQTL study (|z_eQTL| < %g)' % (cell_type, uncalled_z), fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)

    pd.DataFrame(hist_rows, columns=['series', 'z_lower', 'z_upper', 'n', 'density']).to_csv(
        output_file.rsplit('.', 1)[0] + '_hist_summary.tsv', sep='\t', index=False)
    summary = pd.DataFrame(frac_rows, columns=['panel', 'abs_z_pred_bin', 'n', 'k', 'fraction', 'ci95_lower', 'ci95_upper'])
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary



def moment_correlation(x, y, var_x, var_y, n_blocks=200):
    # Noise-corrected (disattenuated) correlation and slope of true effects from noisy estimates with known error
    # variances, using only second moments (no normality assumed), over ALL supplied pairs (no selection):
    #   cov(x_hat, y_hat) estimates cov(x, y) when errors are independent
    #   var(x_hat) - mean(var_x) estimates var(x); likewise for y
    # var_x / var_y only need the right mean, so individual entries may be negative (unbiased product variance).
    # Standard errors from a block jackknife over n_blocks contiguous row blocks (rows are in genomic order,
    # so blocks are genomic segments and LD between neighbouring pairs stays within blocks).
    n = len(x)
    idx = (np.arange(n) * n_blocks) // n
    def bsum(v):
        return np.bincount(idx, weights=v, minlength=n_blocks)
    S = {'n': np.bincount(idx, minlength=n_blocks).astype(float), 'x': bsum(x), 'y': bsum(y), 'xx': bsum(x * x),
         'yy': bsum(y * y), 'xy': bsum(x * y), 'sx2': bsum(var_x), 'sy2': bsum(var_y)}
    names = ['r_naive', 'r_true', 'slope_naive', 'slope_true', 'reliability_x', 'reliability_y']

    def estimates(T):
        m = T['n']
        mx, my = T['x'] / m, T['y'] / m
        vx = T['xx'] / m - mx ** 2
        vy = T['yy'] / m - my ** 2
        cxy = T['xy'] / m - mx * my
        vx_true = vx - T['sx2'] / m
        vy_true = vy - T['sy2'] / m
        with np.errstate(invalid='ignore', divide='ignore'):
            r_true = cxy / np.sqrt(vx_true * vy_true) if vx_true > 0 and vy_true > 0 else np.nan
            slope_true = cxy / vx_true if vx_true > 0 else np.nan
            return np.array([cxy / np.sqrt(vx * vy), r_true, cxy / vx, slope_true, vx_true / vx, vy_true / vy])

    total = {k: v.sum() for k, v in S.items()}
    est = estimates(total)
    jk = np.array([estimates({k: total[k] - S[k][b] for k in S}) for b in range(n_blocks)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        se = np.sqrt((n_blocks - 1) / n_blocks * np.nansum((jk - np.nanmean(jk, axis=0)) ** 2, axis=0))
    se[np.isnan(est)] = np.nan
    out = {name: est[i] for i, name in enumerate(names)}
    out.update({name + '_se': se[i] for i, name in enumerate(names)})
    out['n'] = n
    return out


def fig_moment_correlation(df, cell_type, output_file):
    # Noise-corrected correlation of true predicted vs observed eQTL effects, with sensitivity analyses:
    # trimming extreme predicted effects, and stratifying by minor allele frequency and number of peaks.
    x = df['beta_combined'].to_numpy()
    y = df['beta_eqtl_hat'].to_numpy()
    vx = df['var_combined_unbiased'].to_numpy()
    vy = df['beta_eqtl_se'].to_numpy() ** 2
    af = df['af'].to_numpy()
    maf = np.minimum(af, 1 - af)
    n_peaks = df['n_peaks'].to_numpy()

    # Sensitivity rows stratify on covariates only (MAF, n_peaks), never on the estimated effect or its SE: selecting
    # on a noisy estimate removes real variance while the noise correction stays, and the prediction's SE is itself a
    # function of the estimated betas, so filtering on it is indirect selection too.
    analyses = [('All pairs', np.ones(len(x), dtype=bool))]
    for lo, hi in [(0, 0.05), (0.05, 0.2), (0.2, 0.5)]:
        analyses.append(('MAF %g–%g' % (lo, hi), (maf >= lo) & (maf < hi) if hi < 0.5 else (maf >= lo)))
    for lo, hi in [(1, 3), (3, 11), (11, np.inf)]:
        analyses.append(('n_peaks %s' % ('%d+' % lo if np.isinf(hi) else '%d–%d' % (lo, hi - 1)), (n_peaks >= lo) & (n_peaks < hi)))

    rows = []
    for name, mask in analyses:
        if mask.sum() < 1000:
            continue
        res = moment_correlation(x[mask], y[mask], vx[mask], vy[mask])
        res['analysis'] = name
        rows.append(res)
    summary = pd.DataFrame(rows).set_index('analysis')
    summary.to_csv(summary_path(output_file), sep='\t')

    fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(10.4, 0.45 * len(summary) + 1.8), sharey=True)
    ypos = np.arange(len(summary))[::-1]
    for ax, naive, true, xlabel, ref in [(ax_r, 'r_naive', 'r_true', 'Correlation of predicted and observed effects', 0),
                                         (ax_s, 'slope_naive', 'slope_true', 'Slope of observed on predicted effect', 1)]:
        ax.axvline(ref, color=GRID_GRAY, linewidth=1, zorder=0)
        ax.errorbar(summary[naive], ypos + 0.15, xerr=1.96 * summary[naive + '_se'], fmt='o', color=SERIES[1], ecolor=SERIES[1],
                    elinewidth=1, capsize=2, markersize=5, markeredgecolor='white', label='Naive (attenuated)', zorder=3)
        ax.errorbar(summary[true], ypos - 0.15, xerr=1.96 * summary[true + '_se'], fmt='o', color=SERIES[0], ecolor=SERIES[0],
                    elinewidth=1, capsize=2, markersize=6.5, markeredgecolor='white', label='Noise-corrected', zorder=4)
        ax.set_xlabel(xlabel, color=INK, fontsize=9)
        style_axes(ax)
    ax_r.set_yticks(ypos)
    ax_r.set_yticklabels(['%s\n(n = %s)' % (name, compact(n)) for name, n in zip(summary.index, summary['n'])], fontsize=8)
    ax_r.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc='lower right')
    fig.suptitle('%s cells: noise-corrected correlation (block jackknife 95%% CI)' % cell_type, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    return summary


########################
# Main
########################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta_combined_file', type=str, required=True)
    parser.add_argument('--cell_type', type=str, required=True)
    parser.add_argument('--output_prefix', type=str, required=True)
    parser.add_argument('--n_bins', type=int, default=100)
    parser.add_argument('--n_bins_stratified', type=int, default=20)
    parser.add_argument('--format', type=str, default='pdf')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    df, gene_max_abs_z_eqtl_all = load_pairs_with_prediction(args.beta_combined_file)
    if len(df) < args.n_bins:
        print("Not enough variant-gene pairs with a prediction to form %d bins: %d" % (args.n_bins, len(df)))
        sys.exit(1)

    beta_pred = df['beta_combined'].to_numpy()
    beta_eqtl = df['beta_eqtl_hat'].to_numpy()
    z_pred = df['z_pred'].to_numpy()
    z_eqtl = df['z_eqtl'].to_numpy()
    n_peaks = df['n_peaks'].to_numpy()
    label = bin_label(args.n_bins)
    if label == 'bin':
        label = '%dbin' % args.n_bins
    fmt = args.format

    # 1. Observed vs predicted effect, by bin of predicted effect
    summary = fig_bin_scatter(beta_pred, beta_eqtl, args.n_bins, args.cell_type,
                              'Mean predicted eQTL effect (Σ β_caQTL × β_link)', 'Mean observed eQTL effect (β_eQTL)',
                              '%s_%s_scatter.%s' % (args.output_prefix, label, fmt))
    print(summary.to_string(index=False), flush=True)

    # 2. Same thing on z-scores: observed z vs predicted z, by bin of predicted z
    fig_bin_scatter(z_pred, z_eqtl, args.n_bins, args.cell_type,
                    'Mean predicted z (β_combined / SE_combined)', 'Mean observed eQTL z (β_eQTL / SE_eQTL)',
                    '%s_z_%s_scatter.%s' % (args.output_prefix, label, fmt))

    # 3. Observed vs predicted effect, restricted to increasingly confident predictions
    summary = fig_bin_scatter_by_confidence(df, Z_THRESHOLDS, args.n_bins_stratified, args.cell_type,
                                            '%s_scatter_by_pred_confidence.%s' % (args.output_prefix, fmt))
    print(summary.to_string(index=False), flush=True)

    # 4. Sign concordance between prediction and observation, by |z_pred| bin
    z_group = np.digitize(np.abs(z_pred), Z_EDGES[1:-1])
    z_labels = range_labels(Z_EDGES)
    all_pairs = np.ones(len(df), dtype=bool)
    same_sign = np.sign(beta_eqtl) == np.sign(beta_pred)
    summary = fig_fraction_by_group(z_group, z_labels,
                                    [('All pairs', all_pairs, same_sign),
                                     ('Pairs with |z_eQTL| > 2', np.abs(z_eqtl) > 2, same_sign),
                                     ('Pairs with |z_eQTL| > 4', np.abs(z_eqtl) > 4, same_sign)],
                                    args.cell_type, '|z_pred|', 'Fraction with concordant sign',
                                    '%s_sign_concordance_by_pred_confidence.%s' % (args.output_prefix, fmt),
                                    ref_line=0.5, ref_label='null')
    print(summary.to_string(index=False), flush=True)

    # 5. Observed eQTL signal, by |z_pred| bin
    fig_fraction_by_group(z_group, z_labels,
                          [('|z_eQTL| > 2', all_pairs, np.abs(z_eqtl) > 2),
                           ('|z_eQTL| > 4', all_pairs, np.abs(z_eqtl) > 4)],
                          args.cell_type, '|z_pred|', 'Fraction of pairs with observed eQTL signal',
                          '%s_eqtl_signal_by_pred_confidence.%s' % (args.output_prefix, fmt),
                          ref_line=0.0455, ref_label='null (|z| > 2)')

    # 6. Sign concordance among confident pairs, by number of contributing peaks
    peak_group = np.digitize(n_peaks, N_PEAK_EDGES[1:-1])
    peak_labels = range_labels(N_PEAK_EDGES, integer=True)
    confident = (np.abs(z_pred) > 2) & (np.abs(z_eqtl) > 2)
    fig_fraction_by_group(peak_group, peak_labels,
                          [('|z_pred| > 2 and |z_eQTL| > 2', confident, same_sign)],
                          args.cell_type, 'Number of peaks contributing to the prediction', 'Fraction with concordant sign',
                          '%s_sign_concordance_by_n_peaks.%s' % (args.output_prefix, fmt),
                          ref_line=0.5, ref_label='null')

    # 7. P-value histograms for the observed eQTL and the combined prediction
    fig_pvalue_histograms(two_sided_p_from_z(z_eqtl), two_sided_p_from_z(z_pred), args.cell_type,
                          '%s_pvalue_histograms.%s' % (args.output_prefix, fmt))

    # 8. Scatter of all effect sizes, rendered as a density and saved as png
    fig_effect_size_density(beta_pred, beta_eqtl, args.cell_type, '%s_effect_size_scatter.png' % args.output_prefix)

    # 9. Same scatter restricted to pairs confident on both sides, at each threshold
    for t in SCATTER_Z_THRESHOLDS:
        confident_both = (np.abs(z_pred) > t) & (np.abs(z_eqtl) > t)
        if confident_both.sum() > 1:
            fig_effect_size_density(beta_pred[confident_both], beta_eqtl[confident_both], args.cell_type,
                                    '%s_effect_size_scatter_z%g.png' % (args.output_prefix, t),
                                    note='|z_pred| > %g and |z_eQTL| > %g, n = %s pairs' % (t, t, format(int(confident_both.sum()), ',')))
        else:
            print("No pairs with |z| > %g in both the prediction and the eQTL; skipping that scatter" % t, flush=True)

    # 10. Same scatter restricted to confident predictions only (|z_pred| > 5), with no restriction on the eQTL z
    confident_pred = np.abs(z_pred) > 5
    if confident_pred.sum() > 1:
        fig_effect_size_density(beta_pred[confident_pred], beta_eqtl[confident_pred], args.cell_type,
                                '%s_effect_size_scatter_pred_z5.png' % args.output_prefix,
                                note='|z_pred| > 5 (no restriction on z_eQTL), n = %s pairs' % format(int(confident_pred.sum()), ','))
    else:
        print("No pairs with |z_pred| > 5; skipping that scatter", flush=True)

    # 11. Reverse of figure 5: fraction of pairs with a confident prediction, by |z_eQTL| bin
    z_eqtl_group = np.digitize(np.abs(z_eqtl), Z_EDGES[1:-1])
    summary = fig_fraction_by_group(z_eqtl_group, z_labels,
                                    [('|z_pred| > 2', all_pairs, np.abs(z_pred) > 2),
                                     ('|z_pred| > 3', all_pairs, np.abs(z_pred) > 3),
                                     ('|z_pred| > 5', all_pairs, np.abs(z_pred) > 5)],
                                    args.cell_type, '|z_eQTL|', 'Fraction of pairs with confident prediction',
                                    '%s_pred_confidence_by_eqtl_signal.%s' % (args.output_prefix, fmt),
                                    ref_line=0.0455, ref_label='null (|z| > 2)')
    print(summary.to_string(index=False), flush=True)

    # 12. ROC / precision-recall for detecting observed eQTLs (|z_eQTL| > 4) from |z_pred|, with n_peaks as a baseline
    is_eqtl = np.abs(z_eqtl) > 4
    if is_eqtl.sum() > 0 and (~is_eqtl).sum() > 0:
        summary = fig_roc_pr([('|z_pred|', np.abs(z_pred)), ('n_peaks', n_peaks.astype(float))], is_eqtl, '|z_eQTL| > 4',
                             args.cell_type, '%s_eqtl_detection_roc_pr.%s' % (args.output_prefix, fmt))
        print(summary.to_string(index=False), flush=True)

    # 13. Within-gene localization: percentile rank of each gene's eQTL lead variant by |z_pred|
    genes = per_gene_table(df, gene_max_abs_z_eqtl_all)
    genes.to_csv('%s_per_gene.tsv' % args.output_prefix, sep='\t')
    summary = fig_lead_variant_rank(genes, args.cell_type, '%s_lead_variant_pred_rank.%s' % (args.output_prefix, fmt))
    print(summary.to_string(index=False), flush=True)
    genes20 = genes[genes['n_variants'] >= 20]

    # 14. Lead localization as a function of prediction confidence: fraction of genes whose eQTL lead is in the
    #     top 5% by |z_pred|, stratified by (a) the gene's maximum |z_pred| over all its variants and (b) the lead's own |z_pred|
    is_egene = (genes20['lead_abs_z_eqtl'] > 4).to_numpy()
    lead_top5 = (genes20['percentile'] <= 0.05).to_numpy()
    gene_series = [('Lead |z_eQTL| > 4', is_egene, lead_top5), ('Lead |z_eQTL| ≤ 4', ~is_egene, lead_top5)]
    for key, col, xlabel in [('gene_max', 'max_abs_z_pred', 'Maximum |z_pred| across the gene\'s variants'),
                             ('lead', 'lead_abs_z_pred', '|z_pred| of the eQTL lead variant')]:
        conf_group = np.digitize(genes20[col].to_numpy(), LEAD_Z_EDGES[1:-1])
        summary = fig_fraction_by_group(conf_group, range_labels(LEAD_Z_EDGES), gene_series, args.cell_type, xlabel,
                                        'Fraction of genes with eQTL lead in top 5% by |z_pred|',
                                        '%s_lead_variant_top5_by_%s_z_pred.%s' % (args.output_prefix, key, fmt),
                                        ref_line=0.05, ref_label='uniform')
        print(summary.to_string(index=False), flush=True)

    # 15. Discovery: among pairs the eQTL study did not call, is a confident prediction enriched for sub-threshold signal?
    summary = fig_undiscovered_pairs(z_pred, z_eqtl, beta_pred, args.cell_type,
                                     '%s_undiscovered_pairs.%s' % (args.output_prefix, fmt))
    print(summary.to_string(index=False), flush=True)

    # 16. Gene-level overlap: where the eQTL sits relative to the gene's top predicted variant, by prediction confidence
    summary = fig_gene_overlap(genes, args.cell_type, '%s_gene_overlap.%s' % (args.output_prefix, fmt))
    print(summary.to_string(index=False), flush=True)

    # 17. Why eGenes are silent: prediction status of every tested gene, and coverage of silent vs confident eGenes
    summary = fig_egene_coverage(genes, gene_max_abs_z_eqtl_all, args.cell_type, '%s_egene_coverage.%s' % (args.output_prefix, fmt))
    print(summary.to_string(index=False), flush=True)

    # 18. Per-gene correlation between predicted and observed effects
    summary = fig_gene_correlation(genes, args.cell_type, '%s_gene_correlation.%s' % (args.output_prefix, fmt))
    print(summary.to_string(index=False), flush=True)

    # 19. Noise-corrected correlation of true predicted vs observed effects (method of moments, block jackknife)
    summary = fig_moment_correlation(df, args.cell_type, '%s_moment_correlation.%s' % (args.output_prefix, fmt))
    print(summary[['n', 'r_naive', 'r_true', 'r_true_se', 'slope_true', 'slope_true_se', 'reliability_x', 'reliability_y']].to_string(), flush=True)
