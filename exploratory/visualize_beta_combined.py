import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker

SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
GRID_GRAY = '#d9d8d4'

# |z_pred| bins used for the confidence-stratified figures
Z_EDGES = [0, 1, 2, 3, 4, 6, 8, np.inf]
# |z_pred| thresholds used for the small-multiple bin scatter
Z_THRESHOLDS = [0, 2, 4, 6]
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
    cols = ['beta_eqtl_hat', 'beta_eqtl_se', 'beta_combined', 'se_combined', 'n_peaks']
    chunks = []
    n_total = 0
    for chunk in pd.read_csv(beta_combined_file, sep='\t', usecols=cols, chunksize=5000000, na_values=['NA']):
        n_total += len(chunk)
        chunk = chunk.dropna(subset=['beta_eqtl_hat', 'beta_eqtl_se', 'beta_combined', 'se_combined'])
        chunks.append(chunk)
        print("lines read: %d" % n_total, flush=True)
    if len(chunks) == 0:
        return pd.DataFrame(columns=cols + ['z_eqtl', 'z_pred'])
    df = pd.concat(chunks, ignore_index=True)
    # Drop pairs with a zero standard error, which would give infinite z-scores
    df = df[(df['beta_eqtl_se'] > 0) & (df['se_combined'] > 0)].reset_index(drop=True)
    df['n_peaks'] = df['n_peaks'].astype(int)
    df['z_eqtl'] = df['beta_eqtl_hat'] / df['beta_eqtl_se']
    df['z_pred'] = df['beta_combined'] / df['se_combined']
    print("variant-gene pairs: %d; with prediction: %d" % (n_total, len(df)), flush=True)
    return df


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

    df = load_pairs_with_prediction(args.beta_combined_file)
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
