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
# Per-gene Bonferroni significance levels for the LOEUF-decile figures (one panel per level)
BONF_ALPHAS = [0.05, 0.001]


########################
# Data loading
########################
def load_pairs_with_prediction(beta_combined_file):
    # Stream the file in chunks, keeping only variant-gene pairs that have a beta_combined prediction.
    # Also returns gene_all, one row per tested gene (predicted or not): number of tested variants, max |z_eQTL| and the
    # eQTL effect at that lead variant. These feed the per-gene Bonferroni eQTL calls.
    cols = ['gene_id', 'beta_eqtl_hat', 'beta_eqtl_se', 'beta_combined', 'se_combined', 'af', 'n_peaks']
    # Unbiased prediction variance (may be negative) is written by newer versions of generate_beta_combined.py
    header = pd.read_csv(beta_combined_file, sep='\t', nrows=0).columns
    has_unbiased = 'var_combined_unbiased' in header
    if has_unbiased:
        cols = cols + ['var_combined_unbiased']
    else:
        print("WARNING: no var_combined_unbiased column; the noise-corrected correlation will use the conservative "
              "se_combined^2, which overstates the noise and can give a negative corrected variance", flush=True)
    # Components of the top contributing peak, written by newer versions of generate_beta_combined.py
    has_components = 'beta_caqtl_top' in header and 'beta_link_top' in header
    if has_components:
        cols = cols + ['beta_caqtl_top', 'beta_link_top']
    else:
        print("WARNING: no beta_caqtl_top / beta_link_top columns; the effect-size decomposition by LOEUF decile will be skipped", flush=True)
    chunks = []
    gene_chunks = []
    n_total = 0
    n_with_pred_raw = 0
    genes_with_pred_raw = set()
    for chunk in pd.read_csv(beta_combined_file, sep='\t', usecols=cols, chunksize=5000000, na_values=['NA']):
        n_total += len(chunk)
        # Per gene within the chunk: number of tested pairs and the pair with the largest |z_eQTL|
        ok = chunk['beta_eqtl_se'] > 0
        sub = pd.DataFrame({'gene_id': chunk.loc[ok, 'gene_id'].to_numpy(),
                            'abs_z': (chunk.loc[ok, 'beta_eqtl_hat'] / chunk.loc[ok, 'beta_eqtl_se']).abs().to_numpy(),
                            'beta': chunk.loc[ok, 'beta_eqtl_hat'].to_numpy()})
        if len(sub) > 0:
            g = sub.groupby('gene_id', sort=False)
            lead = sub.loc[g['abs_z'].idxmax()].set_index('gene_id')
            lead['n_tested'] = g.size()
            gene_chunks.append(lead)
        chunk = chunk.dropna(subset=['beta_eqtl_hat', 'beta_eqtl_se', 'beta_combined', 'se_combined'])
        n_with_pred_raw += len(chunk)
        genes_with_pred_raw.update(chunk['gene_id'].unique())
        chunks.append(chunk)
        print("lines read: %d" % n_total, flush=True)
    gene_cols = ['max_abs_z_eqtl_all', 'lead_beta_eqtl_all', 'n_tested_all']
    if gene_chunks:
        allg = pd.concat(gene_chunks).reset_index()
        top = allg.loc[allg.groupby('gene_id')['abs_z'].idxmax()].set_index('gene_id')
        gene_all = pd.DataFrame({'max_abs_z_eqtl_all': top['abs_z'], 'lead_beta_eqtl_all': top['beta'],
                                 'n_tested_all': allg.groupby('gene_id')['n_tested'].sum().reindex(top.index)})
    else:
        gene_all = pd.DataFrame(columns=gene_cols, index=pd.Index([], name='gene_id'))
    if len(chunks) == 0:
        return pd.DataFrame(columns=cols + ['z_eqtl', 'z_pred']), gene_all, {}
    df = pd.concat(chunks, ignore_index=True)
    # Drop pairs with a zero standard error, which would give infinite z-scores, and pairs with a non-finite SE or
    # unbiased variance (an infinite link SE gives se_combined = inf and var_combined_unbiased = inf - inf = NaN)
    n_before = len(df)
    keep = (df['beta_eqtl_se'] > 0) & (df['se_combined'] > 0) & np.isfinite(df['se_combined'])
    if has_unbiased:
        keep = keep & np.isfinite(df['var_combined_unbiased'])
    df = df[keep].reset_index(drop=True)
    print("pairs with prediction dropped for zero or non-finite SE / variance: %d" % (n_before - len(df)), flush=True)
    df['n_peaks'] = df['n_peaks'].astype(int)
    df['gene_id'] = df['gene_id'].astype('category')
    if not has_unbiased:
        df['var_combined_unbiased'] = df['se_combined'] ** 2
    df['z_eqtl'] = df['beta_eqtl_hat'] / df['beta_eqtl_se']
    df['z_pred'] = df['beta_combined'] / df['se_combined']
    print("variant-gene pairs: %d; with prediction: %d; genes tested: %d" % (n_total, len(df), len(gene_all)), flush=True)
    # Funnel counts (pairs, genes) at each stage of the pipeline that is recoverable from the output file
    funnel = {'tested': (n_total, len(gene_all)),
              'with_prediction': (n_with_pred_raw, len(genes_with_pred_raw)),
              'finite_se': (len(df), df['gene_id'].nunique())}
    return df, gene_all, funnel


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
    # Smaller marks once the bins get dense; white ring so overlapping marks stay distinct
    markersize = 6.5 if n_bins <= 20 else 4.5
    capsize = 2 if n_bins <= 20 else 1
    ax.grid(True, color=GRID_GRAY, linewidth=0.5, alpha=0.6, zorder=0)
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
def symlog_threshold(v):
    # Linear window half-width for a symlog axis: the smallest nonzero |value|, so every point sits in the log region
    a = np.abs(np.asarray(v, dtype=float))
    a = a[a > 0]
    return a.min() if len(a) > 0 else 1.0


def fig_bin_scatter(x, y, n_bins, cell_type, xlabel, ylabel, output_file, symlog=False, linthresh=(5e-5, 5e-4), fit_line=True,
                    subtitle=None):
    # Mean y vs mean x within equal-count bins of x, with the OLS fit of y on x over all supplied pairs (not the bin means).
    # With symlog, both axes are log-distance from zero (linear inside +/- linthresh), with ticks in the original units,
    # and the fit is a power law to the bin means, which is a straight line on those axes.
    bins = equal_count_bins(x, n_bins)
    summary = bin_means(x, y, bins, n_bins)
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    slope, slope_se = ols_slope(x, y)
    intercept = y.mean() - slope * x.mean()

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    draw_bin_scatter(ax, summary, n_bins)
    xm, ym = summary['x_mean'].to_numpy(), summary['y_mean'].to_numpy()
    fit_label = None
    if symlog:
        # log10|y| = a + b log10|x| over bins outside the linear window with matching signs; b ~ 1 means linear with scale factor c
        use = (np.abs(xm) > linthresh[0]) & (np.abs(ym) > linthresh[1]) & (np.sign(xm) == np.sign(ym))
        b, b_se = ols_slope(np.log10(np.abs(xm[use])), np.log10(np.abs(ym[use])))
        a = np.log10(np.abs(ym[use])).mean() - b * np.log10(np.abs(xm[use])).mean()
        fit_label = 'Power-law fit to %d bins: |y| = %.2f |x|$^{%.2f}$ (exponent SE %.2f)' % (use.sum(), 10 ** a, b, b_se)
        for sign, edge in [(-1, -xm.min()), (1, xm.max())]:
            xs = np.logspace(np.log10(linthresh[0]), np.log10(edge), 200)
            ax.plot(sign * xs, sign * 10 ** a * xs ** b, color=SERIES[1], linewidth=1.5, zorder=2,
                    label=fit_label if sign == 1 else None)
        ax.set_xscale('symlog', linthresh=linthresh[0], linscale=0.3)
        ax.set_yscale('symlog', linthresh=linthresh[1], linscale=0.3)
        # Decade ticks outside the linear window only, plus zero, so labels do not collide around the origin
        for axis, thresh, vmax in [(ax.xaxis, linthresh[0], np.abs(xm).max()), (ax.yaxis, linthresh[1], np.abs(ym).max())]:
            decades = 10.0 ** np.arange(np.ceil(np.log10(thresh)), np.floor(np.log10(vmax)) + 1)
            axis.set_ticks(np.concatenate([-decades[::-1], [0], decades]))
            axis.set_minor_locator(matplotlib.ticker.NullLocator())
    elif fit_line:
        fit_label = 'OLS fit over %s pairs: slope = %.2f (SE %.2f)' % (compact(len(x)), slope, slope_se)
        xs = np.linspace(xm.min(), xm.max(), 400)
        ax.plot(xs, intercept + slope * xs, color=SERIES[1], linewidth=1.5, zorder=2, label=fit_label)
    if fit_label is not None:
        leg = ax.legend(loc='upper left', fontsize=8, frameon=False, handlelength=1.6)
        for t in leg.get_texts():
            t.set_color(INK_SECONDARY)
    ax.margins(0.06)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    if subtitle is None:
        subtitle = '%s pairs in %d equal-count bins; bars are 95%% CIs of bin means' % (compact(len(x)), n_bins)
    ax.set_title('%s cells' % cell_type, fontsize=10, color=INK, loc='left', pad=18)
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, va='bottom', ha='left')
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


def fig_effect_size_density(x, y, cell_type, output_file, n_bins=400, note=None, fit_line=True,
                            xlabel='Predicted eQTL effect (Σ β_caQTL × β_link)', ylabel='Observed eQTL effect (β_eQTL)'):
    # Scatter of every pair rendered as a 2D histogram with a log color scale (a point scatter of
    # tens of millions of pairs is unreadable and enormous); axes trimmed to the central quantile range
    q = SCATTER_TAIL_QUANTILE
    xlim = np.quantile(x, [q, 1 - q])
    ylim = np.quantile(y, [q, 1 - q])
    counts, xedges, yedges = np.histogram2d(x, y, bins=n_bins, range=[xlim, ylim])
    n_outside = len(x) - int(counts.sum())

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seq_blue', ['#d6e6f7', SERIES[0], '#0a2d5e'])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.grid(True, color=GRID_GRAY, linewidth=0.5, alpha=0.6, zorder=0)
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
    if fit_line:
        # OLS of observed on predicted over the shown pairs. Selecting on |z_pred| is selection on x, which leaves this
        # regression unbiased apart from errors-in-variables attenuation; selecting on |z_eQTL| as well biases it upward.
        slope, slope_se = ols_slope(x, y)
        intercept = y.mean() - slope * x.mean()
        xs = np.array(xlim)
        ax.plot(xs, intercept + slope * xs, color=SERIES[1], linewidth=1.5, zorder=3,
                label='OLS fit: slope = %.2f (SE %.2f)' % (slope, slope_se))
        leg = ax.legend(loc='upper left', fontsize=8, frameon=False, handlelength=1.6)
        for t in leg.get_texts():
            t.set_color(INK_SECONDARY)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title('%s cells' % cell_type, fontsize=10, color=INK, loc='left', pad=18 if note is not None else 6)
    if note is not None:
        ax.text(0, 1.012, note, transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, va='bottom', ha='left')
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


def per_gene_table(df, gene_all):
    # One row per gene with at least one predicted variant:
    #   n_variants          number of variants with a prediction
    #   lead_*              the eQTL lead among predicted variants (largest |z_eQTL|)
    #   max_abs_z_pred      largest |z_pred| in the gene; top_pred_* are values at that variant (|z_eQTL|, beta_combined,
    #                       and the caQTL / link components of its top contributing peak when available)
    #   n_better_predicted  number of variants ranked above the lead by |z_pred|; percentile = rank / (n - 1)
    #   spearman            Spearman correlation of beta_pred vs beta_eQTL across the gene's variants
    #   *_all               from gene_all: over every tested pair for the gene, predicted or not (max |z_eQTL|, the
    #                       eQTL effect at that lead, number of tested variants)
    # Group on integer category codes (cheap); map back to gene names at the end
    key = df['gene_id'].cat.codes.to_numpy()
    gene_names = np.asarray(df['gene_id'].cat.categories)
    g = pd.DataFrame({'gene': key, 'abs_z_eqtl': np.abs(df['z_eqtl'].to_numpy()), 'abs_z_pred': np.abs(df['z_pred'].to_numpy()),
                      'beta_pred': df['beta_combined'].to_numpy(), 'beta_eqtl': df['beta_eqtl_hat'].to_numpy(),
                      'n_peaks': df['n_peaks'].to_numpy()})
    has_components = 'beta_caqtl_top' in df.columns
    if has_components:
        g['beta_caqtl_top'] = df['beta_caqtl_top'].to_numpy()
        g['beta_link_top'] = df['beta_link_top'].to_numpy()
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
                          'top_pred_beta_combined': top['beta_pred'], 'top_pred_beta_eqtl': top['beta_eqtl'],
                          'max_n_peaks': grouped['n_peaks'].max(), 'n_better_predicted': n_better.reindex(n_var.index),
                          'spearman': rho.reindex(n_var.index)})
    if has_components:
        genes['top_pred_beta_caqtl'] = top['beta_caqtl_top']
        genes['top_pred_beta_link'] = top['beta_link_top']
    with np.errstate(divide='ignore', invalid='ignore'):
        genes['percentile'] = np.where(genes['n_variants'] > 1, genes['n_better_predicted'] / (genes['n_variants'] - 1), np.nan)
    genes.index = pd.Index(gene_names[genes.index.to_numpy()], name='gene_id')
    for col in ['max_abs_z_eqtl_all', 'lead_beta_eqtl_all', 'n_tested_all']:
        genes[col] = gene_all[col].reindex(genes.index).to_numpy()
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


def fig_moment_correlation(df, cell_type, output_file, loeuf=None):
    # Noise-corrected correlation of true predicted vs observed eQTL effects, with sensitivity analyses:
    # trimming extreme predicted effects, and stratifying by minor allele frequency and number of peaks.
    # Short-term stand-in for the se_link filter in generate_beta_combined.py, for output files written before it existed:
    # a legitimate link SE (< ~5 per read) gives a prediction SE of order 0.01, so se_combined above max_se_pred can only
    # come from a degenerate link fit. This is a fit-quality filter, not selection on the effect.
    # Sensitivity of the all-pairs estimate to the cap. A cap at which r_true stops moving is one that has removed the
    # noise tail without removing signal; the main table below uses max_se_pred.
    print("Moment correlation, all pairs, by cap on se_combined:", flush=True)
    for cap in [1.0, 0.3, 0.1, 0.03, 0.01]:
        k = (df['se_combined'] <= cap).to_numpy()
        m = moment_correlation(df.loc[k, 'beta_combined'].to_numpy(), df.loc[k, 'beta_eqtl_hat'].to_numpy(),
                               df.loc[k, 'var_combined_unbiased'].to_numpy(), df.loc[k, 'beta_eqtl_se'].to_numpy() ** 2)
        print("  cap %.2f: n = %d (dropped %d), r_naive = %.3f, r_true = %.3f (se %.3f), slope_true = %.2f, reliability_x = %.3f"
              % (cap, m['n'], len(df) - m['n'], m['r_naive'], m['r_true'], m['r_true_se'], m['slope_true'], m['reliability_x']), flush=True)
    max_se_pred = 0.1
    keep = (df['se_combined'] <= max_se_pred).to_numpy()
    print("Moment correlation: dropping %d of %d pairs with se_combined > %.1f (degenerate link fits)"
          % ((~keep).sum(), len(df), max_se_pred), flush=True)
    df = df[keep]
    x = df['beta_combined'].to_numpy()
    y = df['beta_eqtl_hat'].to_numpy()
    vx = df['var_combined_unbiased'].to_numpy()
    vy = df['beta_eqtl_se'].to_numpy() ** 2
    af = df['af'].to_numpy()
    maf = np.minimum(af, 1 - af)
    n_peaks = df['n_peaks'].to_numpy()

    # Diagnostic for reliability_x: var(x) must exceed mean(var_x). Report how concentrated the error variance is:
    # if the top 0.1% of pairs carry most of the mean, a few links with huge SEs are driving the correction
    order = np.argsort(vx)[::-1]
    top = int(np.ceil(len(vx) * 0.001))
    print("Noise-model diagnostic (all pairs): var(beta_combined) = %.3e, mean(var_combined_unbiased) = %.3e, "
          "median = %.3e, 99th pct = %.3e, 99.9th pct = %.3e, share of sum(var_x) in top 0.1%% of pairs = %.3f"
          % (np.var(x), np.mean(vx), np.median(vx), np.percentile(vx, 99), np.percentile(vx, 99.9),
             vx[order[:top]].sum() / vx.sum()), flush=True)

    # Sensitivity rows stratify on covariates only (MAF, n_peaks), never on the estimated effect or its SE: selecting
    # on a noisy estimate removes real variance while the noise correction stays, and the prediction's SE is itself a
    # function of the estimated betas, so filtering on it is indirect selection too.
    analyses = [('All pairs', np.ones(len(x), dtype=bool))]
    for lo, hi in [(0, 0.05), (0.05, 0.2), (0.2, 0.5)]:
        analyses.append(('MAF %g–%g' % (lo, hi), (maf >= lo) & (maf < hi) if hi < 0.5 else (maf >= lo)))
    for lo, hi in [(1, 3), (3, 11), (11, np.inf)]:
        analyses.append(('n_peaks %s' % ('%d+' % lo if np.isinf(hi) else '%d–%d' % (lo, hi - 1)), (n_peaks >= lo) & (n_peaks < hi)))
    if loeuf is not None:
        # Gene constraint is a gene-level covariate, so stratifying on it is legitimate; -1 = gene without a LOEUF value
        decile = decile_index(loeuf, pd.Index(df['gene_id'].astype(str))) + 1
        print("Moment correlation: %d of %d pairs have a LOEUF decile" % (int((decile > 0).sum()), len(decile)), flush=True)
        for lo, hi in [(1, 2), (3, 5), (6, 10)]:
            analyses.append(('LOEUF decile %d–%d' % (lo, hi), (decile >= lo) & (decile <= hi)))

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
# LOEUF
########################
def load_loeuf_deciles(loeuf_file, gene_ids):
    # gnomAD constraint metrics -> one LOEUF decile (1 = most constrained, 10 = least) per gene in gene_ids.
    # Uses gnomAD's own genome-wide decile column when present, otherwise deciles of LOEUF across all MANE/canonical
    # transcripts. Genes are matched on Ensembl ID (version stripped), falling back to gene symbol.
    tab = pd.read_csv(loeuf_file, sep='\t', low_memory=False)
    loeuf_col = next((c for c in ['lof.oe_ci.upper', 'oe_lof_upper'] if c in tab.columns), None)
    if loeuf_col is None:
        raise ValueError('No LOEUF column found in %s; columns: %s' % (loeuf_file, list(tab.columns)))
    # One transcript per gene: MANE select if annotated, else canonical, else lowest LOEUF
    for flag in ['mane_select', 'canonical']:
        if flag in tab.columns:
            sel = tab[flag].astype(str).str.lower().isin(['true', '1', 'yes'])
            if sel.sum() > 0:
                tab = tab[sel]
                break
    tab = tab.dropna(subset=[loeuf_col]).sort_values(loeuf_col)
    decile_col = 'lof.oe_ci.upper_bin_decile' if 'lof.oe_ci.upper_bin_decile' in tab.columns else None
    if decile_col is not None and tab[decile_col].notna().sum() > 0:
        tab = tab.dropna(subset=[decile_col])
        decile = tab[decile_col].astype(int).to_numpy() + 1
        print("LOEUF deciles: gnomAD genome-wide decile column", flush=True)
    else:
        decile = equal_count_bins(tab[loeuf_col].to_numpy(), 10) + 1
        print("LOEUF deciles: computed from LOEUF across %d transcripts" % len(tab), flush=True)
    tab = pd.DataFrame({'loeuf': tab[loeuf_col].to_numpy(), 'decile': decile,
                        'ensg': tab['gene_id'].astype(str).str.split('.').str[0].to_numpy() if 'gene_id' in tab.columns else None,
                        'symbol': tab['gene'].astype(str).to_numpy() if 'gene' in tab.columns else None})
    tab = tab.drop_duplicates('ensg', keep='first')

    query = pd.Index(gene_ids)
    by_ensg = tab.set_index('ensg').reindex(query.str.split('.').str[0])
    by_symbol = tab.drop_duplicates('symbol').set_index('symbol').reindex(query)
    n_ensg, n_symbol = int(by_ensg['decile'].notna().sum()), int(by_symbol['decile'].notna().sum())
    matched = by_ensg if n_ensg >= n_symbol else by_symbol
    print("LOEUF matched %d of %d genes (by %s)" % (int(matched['decile'].notna().sum()), len(query),
                                                     'Ensembl ID' if n_ensg >= n_symbol else 'symbol'), flush=True)
    out = pd.DataFrame({'loeuf': matched['loeuf'].to_numpy(), 'loeuf_decile': matched['decile'].to_numpy()}, index=query)
    return out


def gene_calls(genes, gene_all, alpha):
    # Per-gene calibrated QTL calls over every tested gene. eqtl_call: Bonferroni on the minimum two-sided p across the
    # gene's tested variants; pred_call: the same across the gene's predicted variants (False when there is none).
    # Bonferroni is conservative under LD, so the calls are strict rather than exact, but unlike a fixed |z| cutoff on a
    # max statistic they do not saturate with the number of variants.
    calls = gene_all.copy()
    calls['eqtl_p_bonf'] = np.minimum(1.0, two_sided_p_from_z(calls['max_abs_z_eqtl_all'].to_numpy(dtype=float))
                                      * calls['n_tested_all'].to_numpy(dtype=float))
    calls['eqtl_call'] = calls['eqtl_p_bonf'] < alpha
    g = genes.reindex(calls.index)
    calls['has_pred'] = g['n_variants'].notna().to_numpy()
    calls['n_variants'] = g['n_variants'].to_numpy(dtype=float)
    calls['max_abs_z_pred'] = g['max_abs_z_pred'].to_numpy(dtype=float)
    p = two_sided_p_from_z(np.nan_to_num(calls['max_abs_z_pred'].to_numpy(), nan=0.0)) * np.nan_to_num(calls['n_variants'].to_numpy(), nan=1.0)
    calls['pred_p_bonf'] = np.where(calls['has_pred'], np.minimum(1.0, p), np.nan)
    calls['pred_call'] = calls['has_pred'].to_numpy() & (np.nan_to_num(calls['pred_p_bonf'].to_numpy(), nan=1.0) < alpha)
    return calls


def decile_index(loeuf, index):
    # LOEUF decile as a 0-based group index aligned to index; -1 where the gene has no LOEUF value
    d = loeuf['loeuf_decile'].reindex(index).to_numpy(dtype=float)
    return np.where(np.isnan(d), -1, np.nan_to_num(d, nan=0) - 1).astype(int)


DECILE_LABELS = [str(d) for d in range(1, 11)]
DECILE_XLABEL = 'LOEUF decile (1 = most constrained); n genes below'


def draw_fraction_by_decile(ax, group, series, colors=SERIES):
    # Fraction (Wilson 95% CI) of an indicator within each LOEUF decile, for several (name, selection, indicator) series.
    # Returns rows of (series, decile, n, k, fraction, lo, hi). Tick labels carry the first series' group sizes.
    rows = []
    offsets = np.linspace(-0.2, 0.2, len(series)) if len(series) > 1 else [0.0]
    for (name, selection, indicator), offset, color in zip(series, offsets, colors):
        ps, los, his = [], [], []
        for d in range(10):
            in_group = selection & (group == d)
            n = int(in_group.sum())
            k = int((in_group & indicator).sum())
            p, lo, hi = wilson_ci(k, n)
            ps.append(p)
            los.append(lo)
            his.append(hi)
            rows.append((name, d + 1, n, k, p, lo, hi))
        ps, los, his = np.array(ps), np.array(los), np.array(his)
        ax.errorbar(np.arange(10) + offset, ps, yerr=[np.maximum(ps - los, 0), np.maximum(his - ps, 0)],
                    fmt='o', color=color, ecolor=color, elinewidth=1, capsize=2, capthick=1, markersize=5.5,
                    markeredgecolor='white', markeredgewidth=1, label=name, zorder=3)
    first_sel = series[0][1]
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(['%s\n%s' % (lab, compact(int((first_sel & (group == d)).sum()))) for d, lab in enumerate(DECILE_LABELS)], fontsize=7.5)
    ax.set_xlabel(DECILE_XLABEL, color=INK, fontsize=9)
    ax.set_ylim(bottom=0)
    style_axes(ax)
    return rows


def fig_qtl_fraction_by_loeuf_decile(genes, gene_all, loeuf, alphas, cell_type, output_file):
    # Fraction of genes with a calibrated QTL call (per-gene Bonferroni p < alpha) in each LOEUF decile, for the observed
    # eQTL and the chromatin prediction. Universe = every tested gene with a LOEUF value; a gene with no predicted variant
    # counts as no call for the prediction. The third series conditions on having a prediction (coverage vs power).
    group = decile_index(loeuf, gene_all.index)
    has_loeuf = group >= 0
    print("LOEUF figures: %d of %d tested genes have a LOEUF decile" % (int(has_loeuf.sum()), len(gene_all)), flush=True)
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.6 * len(alphas), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, alpha in zip(axes, alphas):
        calls = gene_calls(genes, gene_all, alpha)
        eqtl_call, pred_call, has_pred = calls['eqtl_call'].to_numpy(), calls['pred_call'].to_numpy(), calls['has_pred'].to_numpy()
        series = [('Observed eQTL (all genes)', has_loeuf, eqtl_call),
                  ('Chromatin-predicted (all genes)', has_loeuf, pred_call),
                  ('Chromatin-predicted (genes with a prediction)', has_loeuf & has_pred, pred_call)]
        for r in draw_fraction_by_decile(ax, group, series):
            rows.append((alpha,) + r)
        ax.set_title('Per-gene Bonferroni p < %g' % alpha, fontsize=9, color=INK)
    axes[0].set_ylabel('Fraction of genes with a QTL call', color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY, loc='upper left')
    fig.suptitle('%s cells' % cell_type, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    summary = pd.DataFrame(rows, columns=['alpha', 'series', 'loeuf_decile', 'n_genes', 'n_called', 'fraction', 'ci95_lower', 'ci95_upper'])
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary


def fig_rescue_by_loeuf_decile(genes, gene_all, loeuf, alphas, cell_type, output_file):
    # Recovery of missing regulation: among genes with no calibrated eQTL call, the fraction with a calibrated chromatin
    # prediction, by LOEUF decile. Genes with an eQTL call are the comparison series. Universe as in the fraction figure.
    group = decile_index(loeuf, gene_all.index)
    has_loeuf = group >= 0
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.6 * len(alphas), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, alpha in zip(axes, alphas):
        calls = gene_calls(genes, gene_all, alpha)
        eqtl_call, pred_call = calls['eqtl_call'].to_numpy(), calls['pred_call'].to_numpy()
        series = [('Genes with no eQTL call', has_loeuf & ~eqtl_call, pred_call),
                  ('Genes with an eQTL call', has_loeuf & eqtl_call, pred_call)]
        for r in draw_fraction_by_decile(ax, group, series, colors=[SERIES[1], SERIES[0]]):
            rows.append((alpha,) + r)
        ax.set_title('Per-gene Bonferroni p < %g (both calls)' % alpha, fontsize=9, color=INK)
    axes[0].set_ylabel('Fraction of genes with a chromatin-predicted QTL call', color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY, loc='upper right')
    fig.suptitle('%s cells: chromatin-predicted calls at genes the eQTL scan did and did not call (tick counts: genes with no eQTL call)' % cell_type,
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    summary = pd.DataFrame(rows, columns=['alpha', 'series', 'loeuf_decile', 'n_genes', 'n_pred_called', 'fraction', 'ci95_lower', 'ci95_upper'])
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary


def spearman_with_p(x, y):
    # Spearman rho and an approximate two-sided p from the Fisher transform
    n = len(x)
    if n < 4:
        return np.nan, np.nan
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rho = np.corrcoef(rx, ry)[0, 1]
    if not np.isfinite(rho) or abs(rho) >= 1:
        return rho, np.nan
    z = np.arctanh(rho) * np.sqrt((n - 3) / 1.06)
    return rho, float(two_sided_p_from_z(np.array([z]))[0])


def median_boot_ci(v, rng, n_boot=500):
    # Median with a percentile bootstrap 95% CI
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    med = np.median(v)
    if len(v) < 5:
        return med, np.nan, np.nan
    boots = np.median(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    return med, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def fig_effect_size_by_loeuf_decile(genes, calls, loeuf, alpha, cell_type, output_file):
    # Kanai et al. Fig. 5e/g/h/j analogue: median absolute effect at each gene's strongest variant, by LOEUF decile, with a
    # bootstrap CI and the Spearman correlation between decile and effect. The variant is selected on significance
    # (eQTL lead among genes with an eQTL call; top |z_pred| variant among genes with a prediction call), so the
    # plotted effects carry winner's curse; the decile gradient is the comparison of interest, not the absolute level.
    # Units differ between panels (Kanai: beta_eQTL in log counts per allele; beta_combined has no comparable unit).
    group = decile_index(loeuf, calls.index)
    g = genes.reindex(calls.index)
    eqtl_call, pred_call = calls['eqtl_call'].to_numpy(), calls['pred_call'].to_numpy()
    panels = [('|β_eQTL| at eQTL lead (genes with eQTL call)', 'eqtl', eqtl_call, np.abs(calls['lead_beta_eqtl_all'].to_numpy(dtype=float))),
              ('|β_combined| at top predicted variant (genes with prediction call)', 'combined', pred_call,
               np.abs(g['top_pred_beta_combined'].to_numpy(dtype=float)))]
    if 'top_pred_beta_caqtl' in genes.columns:
        panels += [('|β_caQTL| of top peak at top predicted variant', 'caqtl', pred_call, np.abs(g['top_pred_beta_caqtl'].to_numpy(dtype=float))),
                   ('|β_link| of top peak at top predicted variant', 'link', pred_call, np.abs(g['top_pred_beta_link'].to_numpy(dtype=float)))]
    rng = np.random.default_rng(0)
    n_panels = len(panels)
    n_cols = 2 if n_panels > 2 else n_panels
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.9 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    rows = []
    for ax, (title, key, sel, v) in zip(axes, panels):
        sel = sel & (group >= 0) & np.isfinite(v)
        meds, los, his, ns = [], [], [], []
        for d in range(10):
            vals = v[sel & (group == d)]
            med, lo, hi = median_boot_ci(vals, rng)
            meds.append(med)
            los.append(lo)
            his.append(hi)
            ns.append(len(vals))
            rows.append((key, d + 1, len(vals), med, lo, hi))
        meds, los, his = np.array(meds), np.array(los), np.array(his)
        yerr = [np.nan_to_num(meds - los), np.nan_to_num(his - meds)]
        ax.errorbar(np.arange(10), meds, yerr=yerr, fmt='o', color=SERIES[0], ecolor=SERIES[0], elinewidth=1, capsize=2,
                    capthick=1, markersize=6, markeredgecolor='white', markeredgewidth=1, zorder=3)
        rho, pval = spearman_with_p(group[sel] + 1, v[sel])
        ax.text(0.03, 0.97, 'Spearman ρ = %.3f, p = %.1e (n = %s genes)' % (rho, pval, format(int(sel.sum()), ',')),
                transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, va='top')
        ax.set_xticks(np.arange(10))
        ax.set_xticklabels(['%s\n%s' % (lab, compact(n)) for lab, n in zip(DECILE_LABELS, ns)], fontsize=7.5)
        ax.set_xlabel(DECILE_XLABEL, color=INK, fontsize=9)
        ax.set_ylabel('Median absolute effect (bootstrap 95% CI)', color=INK, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=9, color=INK)
        style_axes(ax)
        rows.append((key + '_spearman', np.nan, int(sel.sum()), rho, pval, np.nan))
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    fig.suptitle('%s cells: effect size at the strongest variant by gene constraint (calls at Bonferroni p < %g)' % (cell_type, alpha),
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    summary = pd.DataFrame(rows, columns=['effect', 'loeuf_decile', 'n_genes', 'median_or_rho', 'ci95_lower_or_p', 'ci95_upper'])
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary


########################
# Main
########################
def fig_null_qq(z_pred, z_eqtl, cell_type, output_file, null_z=1, n_points=2000):
    # QQ plot of z_pred among pairs with no observed eQTL signal (|z_eQTL| < null_z). If the prediction carried no
    # information beyond its stated SE where there is no eQTL, z_pred would be N(0,1); inflation means the SE is too
    # small or the link model leaks signal. Shown for all such pairs and for the extreme tail (|z_eQTL| < 0.5)
    import scipy.stats
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1))
    rows = []
    for ax, t in zip(axes, [null_z, null_z / 2]):
        z = np.sort(z_pred[np.abs(z_eqtl) < t])
        n = len(z)
        if n < 10:
            ax.text(0.5, 0.5, 'n = %d' % n, transform=ax.transAxes, ha='center')
            continue
        probs = (np.arange(1, n + 1) - 0.5) / n
        expected = scipy.stats.norm.ppf(probs)
        # Thin the middle of the distribution for plotting, keep the tails intact
        keep = np.unique(np.concatenate([np.linspace(0, n - 1, n_points).astype(int), np.arange(min(200, n)),
                                         np.arange(max(0, n - 200), n)]))
        lam = np.median(z ** 2) / scipy.stats.chi2.ppf(0.5, 1)
        sd = np.std(z)
        lim = max(abs(expected[0]), abs(expected[-1]), abs(z[0]), abs(z[-1])) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color=GRID_GRAY, linewidth=1, zorder=1)
        ax.plot(expected[keep], z[keep], 'o', color=SERIES[0], markersize=2.5, markeredgewidth=0, zorder=3)
        ax.text(0.03, 0.97, 'n = %s pairs\nλ_GC = %.3f\nSD(z_pred) = %.3f' % (format(n, ','), lam, sd),
                transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, va='top')
        ax.set_title('Pairs with |z_eQTL| < %g' % t, fontsize=9, color=INK)
        ax.set_xlabel('Expected N(0,1) quantile', color=INK, fontsize=9)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        style_axes(ax)
        rows.append((t, n, lam, sd, np.mean(np.abs(z) > 2), np.mean(np.abs(z) > 4)))
    axes[0].set_ylabel('Observed z_pred quantile', color=INK, fontsize=9)
    fig.suptitle('%s cells: z_pred where there is no observed eQTL signal' % cell_type, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)

    summary = pd.DataFrame(rows, columns=['max_abs_z_eqtl', 'n', 'lambda_gc', 'sd_z_pred', 'frac_abs_z_pred_gt2', 'frac_abs_z_pred_gt4'])
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary


def fig_coverage_funnel(stages, cell_type, output_file):
    # Horizontal bar chart of how many variant-gene pairs (left) and genes (right) survive each stage of the pipeline.
    # stages: list of (label, n_pairs, n_genes), in order. Each bar is annotated with the count and the fraction of stage 1
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    rows = []
    labels = [s[0] for s in stages]
    y = np.arange(len(stages))[::-1]
    for ax, j, name in [(axes[0], 1, 'Variant-gene pairs'), (axes[1], 2, 'Genes')]:
        counts = np.array([s[j] for s in stages], dtype=float)
        ax.barh(y, counts, color=SERIES[0], height=0.65, zorder=3)
        for yi, c in zip(y, counts):
            ax.text(c + counts[0] * 0.01, yi, '%s (%.1f%%)' % (format(int(c), ','), 100 * c / counts[0] if counts[0] > 0 else 0),
                    fontsize=8, color=INK_SECONDARY, va='center')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8, color=INK)
        ax.set_xlim(0, counts[0] * 1.35)
        ax.set_title(name, fontsize=9, color=INK)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: compact(v)))
        ax.grid(True, axis='x', color=GRID_GRAY, linewidth=0.5, alpha=0.6, zorder=0)
        style_axes(ax)
    axes[1].set_yticklabels([])
    fig.suptitle('%s cells: prediction coverage' % cell_type, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)

    summary = pd.DataFrame(stages, columns=['stage', 'n_pairs', 'n_genes'])
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary


def fig_variance_diagnostics(df, cell_type, output_file):
    # Sanity check on the delta-method variance. Left: se_combined^2 (conservative, includes the cross term) vs the
    # unbiased variance (subtracts it), on log axes; points below zero on the unbiased side are shown on the log axis
    # as |var| with a separate colour. Middle: fraction of pairs with a negative unbiased variance, by n_peaks.
    # Right: ratio of unbiased to conservative variance, by n_peaks (median with 5-95% band)
    var_c = (df['se_combined'] ** 2).to_numpy()
    var_u = df['var_combined_unbiased'].to_numpy()
    n_peaks = df['n_peaks'].to_numpy()
    neg = var_u < 0
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))

    ax = axes[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), size=min(len(df), 200000), replace=False)
    ax.plot(var_c[idx][~neg[idx]], var_u[idx][~neg[idx]], 'o', color=SERIES[0], markersize=1.5, markeredgewidth=0, alpha=0.3,
            label='unbiased ≥ 0', zorder=3)
    if neg[idx].any():
        ax.plot(var_c[idx][neg[idx]], -var_u[idx][neg[idx]], 'o', color=SERIES[1], markersize=1.5, markeredgewidth=0, alpha=0.3,
                label='unbiased < 0 (|value| shown)', zorder=3)
    lim = (np.nanmin(var_c[var_c > 0]), np.nanmax(var_c))
    ax.plot(lim, lim, color=GRID_GRAY, linewidth=1, zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('se_combined² (conservative)', color=INK, fontsize=9)
    ax.set_ylabel('var_combined_unbiased', color=INK, fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc='upper left', markerscale=4)
    ax.set_title('Negative unbiased variance: %.2f%% of pairs' % (100 * neg.mean()), fontsize=9, color=INK)
    style_axes(ax)

    peak_group = np.digitize(n_peaks, N_PEAK_EDGES[1:-1])
    peak_labels = range_labels(N_PEAK_EDGES, integer=True)
    rows = []
    ax = axes[1]
    fracs, cis, ns = [], [], []
    for g in range(len(peak_labels)):
        m = peak_group == g
        k, n = int(neg[m].sum()), int(m.sum())
        _, lo, hi = wilson_ci(k, n)
        fracs.append(k / n if n > 0 else np.nan)
        cis.append((lo, hi))
        ns.append(n)
    fracs = np.array(fracs)
    cis = np.array(cis)
    x = np.arange(len(peak_labels))
    ax.errorbar(x, fracs, yerr=[fracs - cis[:, 0], cis[:, 1] - fracs], fmt='o-', color=SERIES[0], capsize=2, markersize=5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(['%s\nn=%s' % (l, compact(n)) for l, n in zip(peak_labels, ns)], fontsize=8)
    ax.set_xlabel('Number of peaks contributing to the prediction', color=INK, fontsize=9)
    ax.set_ylabel('Fraction with negative unbiased variance', color=INK, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis='y', color=GRID_GRAY, linewidth=0.5, alpha=0.6, zorder=0)
    style_axes(ax)

    ax = axes[2]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = var_u / var_c
    med, q05, q95 = [], [], []
    for g in range(len(peak_labels)):
        r = ratio[(peak_group == g) & np.isfinite(ratio)]
        if len(r) > 0:
            med.append(np.median(r)); q05.append(np.percentile(r, 5)); q95.append(np.percentile(r, 95))
        else:
            med.append(np.nan); q05.append(np.nan); q95.append(np.nan)
        rows.append((peak_labels[g], ns[g], fracs[g], cis[g, 0], cis[g, 1], med[-1], q05[-1], q95[-1]))
    ax.fill_between(x, q05, q95, color=SERIES[0], alpha=0.15, linewidth=0, zorder=2)
    ax.plot(x, med, 'o-', color=SERIES[0], markersize=5, zorder=3)
    ax.axhline(1, color=GRID_GRAY, linewidth=1, zorder=1)
    ax.axhline(0, color=GRID_GRAY, linewidth=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(peak_labels, fontsize=8)
    ax.set_xlabel('Number of peaks contributing to the prediction', color=INK, fontsize=9)
    ax.set_ylabel('var_unbiased / se_combined²  (median, 5–95%)', color=INK, fontsize=9)
    ax.grid(True, axis='y', color=GRID_GRAY, linewidth=0.5, alpha=0.6, zorder=0)
    style_axes(ax)

    fig.suptitle('%s cells: delta-method variance diagnostics' % cell_type, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)

    summary = pd.DataFrame(rows, columns=['n_peaks', 'n', 'frac_negative', 'ci_lower', 'ci_upper', 'ratio_median', 'ratio_q05', 'ratio_q95'])
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary


def fig_shrinkage_check(df, cell_type, output_file, max_se_pred=0.1):
    # Is the delta-method SE the right amount of noise? Under the measurement-error model x_hat = x + e with known
    # var(e), and y = slope_true * x + noise, the regression of y on x_hat has slope
    #     slope_true * reliability,   reliability = var(x) / (var(x) + var(e))
    # var(x) and slope_true come from the global moment estimate (figure 19). Because var(e) differs across pairs, the
    # expected slope differs across |z_pred| bins: it uses the mean var(e) within the bin. The empirical within-bin OLS
    # slope of beta_eQTL on beta_pred is compared with that expectation. Agreement means the stated SE is the right
    # amount of shrinkage to apply when using beta_pred as a prior; an empirical slope below expectation means the SE
    # is too small (under-shrinkage), above means too large
    keep = (df['se_combined'] <= max_se_pred).to_numpy()
    x = df.loc[keep, 'beta_combined'].to_numpy()
    y = df.loc[keep, 'beta_eqtl_hat'].to_numpy()
    vx = df.loc[keep, 'var_combined_unbiased'].to_numpy()
    vy = df.loc[keep, 'beta_eqtl_se'].to_numpy() ** 2
    z = np.abs(df.loc[keep, 'z_pred'].to_numpy())
    m = moment_correlation(x, y, vx, vy)
    var_true = np.var(x) - np.mean(vx)
    group = np.digitize(z, Z_EDGES[1:-1])
    labels = range_labels(Z_EDGES)
    rows = []
    for g in range(len(labels)):
        k = group == g
        if k.sum() < 100:
            rows.append((labels[g], int(k.sum()), np.nan, np.nan, np.nan, np.nan))
            continue
        slope, se = ols_slope(x[k], y[k])
        # Expected within-bin OLS slope: regress the model's conditional mean, slope_true * r_i * x_hat_i with
        # per-pair reliability r_i = var(x) / (var(x) + var(e_i)), on x_hat within the bin. Pairs with a large SE have
        # a large |x_hat| at a given |z|, so they dominate the within-bin OLS while having the lowest reliability;
        # a single mean reliability per bin would overstate the expected slope
        if var_true > 0:
            r_i = var_true / (var_true + vx[k])
            expected, _ = ols_slope(x[k], m['slope_true'] * r_i * x[k])
            rel = np.mean(r_i)
        else:
            expected, rel = np.nan, np.nan
        rows.append((labels[g], int(k.sum()), slope, se, rel, expected))
    summary = pd.DataFrame(rows, columns=['abs_z_pred', 'n', 'slope_empirical', 'slope_se', 'mean_reliability', 'slope_expected'])

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    xs = np.arange(len(labels))
    ax.axhline(m['slope_naive'], color=GRID_GRAY, linewidth=1, zorder=1)
    ax.text(len(labels) - 0.6, m['slope_naive'], 'global naive slope', fontsize=7, color=INK_SECONDARY, va='bottom', ha='right')
    ax.axhline(m['slope_true'], color=GRID_GRAY, linewidth=1, linestyle='--', zorder=1)
    ax.text(len(labels) - 0.6, m['slope_true'], 'global noise-corrected slope', fontsize=7, color=INK_SECONDARY, va='bottom', ha='right')
    ax.plot(xs, summary['slope_expected'], 's-', color=SERIES[1], markersize=5, markeredgecolor='white',
            label='Expected: slope_true × per-pair reliability', zorder=3)
    ax.errorbar(xs, summary['slope_empirical'], yerr=1.96 * summary['slope_se'], fmt='o', color=SERIES[0], ecolor=SERIES[0],
                elinewidth=1, capsize=2, markersize=6, markeredgecolor='white', label='Empirical within-bin OLS slope', zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(['%s\nn=%s' % (l, compact(n)) for l, n in zip(labels, summary['n'])], fontsize=8)
    ax.set_xlabel('|z_pred|', color=INK, fontsize=9)
    ax.set_ylabel('Slope of β_eQTL on β_pred', color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc='upper left')
    ax.grid(True, axis='y', color=GRID_GRAY, linewidth=0.5, alpha=0.6, zorder=0)
    style_axes(ax)
    ax.set_title('%s cells: shrinkage calibration (se_combined ≤ %g; var(x_true) = %.2e, slope_true = %.2f ± %.2f)'
                 % (cell_type, max_se_pred, var_true, m['slope_true'], m['slope_true_se']), fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary


def fig_heteroskedasticity(df, cell_type, output_file, n_bins=20, max_se_pred=0.1):
    # Residual variance of beta_eQTL around the global calibration line (naive OLS of beta_eQTL on beta_pred), by
    # equal-count bin of |beta_pred|, against the mean eQTL sampling variance in the bin. The excess (residual minus
    # sampling) is the variance of the true eQTL effect given the prediction. Under the model it is
    #     slope_true² · var(x | x_hat) + var(y_true | x)   with   var(x | x_hat) = var(x) · (1 - reliability_bin)
    # which is roughly flat in |beta_pred| apart from the reliability term. Excess that grows with |beta_pred| beyond
    # that means the link / caQTL error scales with the effect size in a way the delta-method SE does not capture
    keep = (df['se_combined'] <= max_se_pred).to_numpy()
    x = df.loc[keep, 'beta_combined'].to_numpy()
    y = df.loc[keep, 'beta_eqtl_hat'].to_numpy()
    vx = df.loc[keep, 'var_combined_unbiased'].to_numpy()
    vy = df.loc[keep, 'beta_eqtl_se'].to_numpy() ** 2
    slope, _ = ols_slope(x, y)
    intercept = np.mean(y) - slope * np.mean(x)
    resid2 = (y - intercept - slope * x) ** 2
    m = moment_correlation(x, y, vx, vy)
    var_true = np.var(x) - np.mean(vx)
    # Variance of true y around the true-effect line: var(y_true) - slope_true² var(x_true)
    var_y_true = np.var(y) - np.mean(vy)
    resid_true = var_y_true - m['slope_true'] ** 2 * var_true if np.isfinite(m['slope_true']) else np.nan
    bins = equal_count_bins(np.abs(x), n_bins)
    rows = []
    for b in range(n_bins):
        k = bins == b
        n = int(k.sum())
        rv, rv_se = np.mean(resid2[k]), np.std(resid2[k]) / np.sqrt(n)
        sv = np.mean(vy[k])
        one_minus_r = np.mean(vx[k] / (var_true + vx[k])) if var_true > 0 else np.nan
        expected = m['slope_true'] ** 2 * var_true * one_minus_r + resid_true if np.isfinite(one_minus_r) else np.nan
        rows.append((b + 1, n, np.mean(np.abs(x[k])), rv, rv_se, sv, rv - sv, expected))
    summary = pd.DataFrame(rows, columns=['bin', 'n', 'mean_abs_beta_pred', 'resid_var', 'resid_var_se', 'mean_se_eqtl_sq',
                                          'excess_var', 'expected_excess_var'])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4))
    xs = summary['mean_abs_beta_pred']
    ax = axes[0]
    ax.errorbar(xs, summary['resid_var'], yerr=1.96 * summary['resid_var_se'], fmt='o-', color=SERIES[0], ecolor=SERIES[0],
                elinewidth=0.8, capsize=2, markersize=5, markeredgecolor='white', label='Residual variance around calibration line', zorder=4)
    ax.plot(xs, summary['mean_se_eqtl_sq'], 's-', color=SERIES[1], markersize=4, markeredgecolor='white', label='Mean SE_eQTL²', zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Variance of β_eQTL', color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc='upper left')
    ax = axes[1]
    ax.axhline(0, color=GRID_GRAY, linewidth=1, zorder=1)
    ax.errorbar(xs, summary['excess_var'], yerr=1.96 * summary['resid_var_se'], fmt='o-', color=SERIES[0], ecolor=SERIES[0],
                elinewidth=0.8, capsize=2, markersize=5, markeredgecolor='white', label='Excess: residual − sampling', zorder=4)
    ax.plot(xs, summary['expected_excess_var'], 's-', color=SERIES[1], markersize=4, markeredgecolor='white',
            label='Expected under model', zorder=3)
    ax.set_xscale('log')
    ax.set_ylabel('Excess variance of β_eQTL given β_pred', color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc='upper left')
    for ax in axes:
        ax.set_xlabel('Mean |β_pred| in bin (%d equal-count bins)' % n_bins, color=INK, fontsize=9)
        ax.grid(True, color=GRID_GRAY, linewidth=0.5, alpha=0.6, zorder=0)
        style_axes(ax)
    fig.suptitle('%s cells: heteroskedasticity of the observed eQTL effect (calibration slope = %.3f; se_combined ≤ %g)'
                 % (cell_type, slope, max_se_pred), fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    summary.to_csv(summary_path(output_file), sep='\t', index=False)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta_combined_file', type=str, required=True)
    parser.add_argument('--cell_type', type=str, required=True)
    parser.add_argument('--output_prefix', type=str, required=True)
    parser.add_argument('--n_bins', type=int, default=100)
    parser.add_argument('--n_bins_stratified', type=int, default=20)
    parser.add_argument('--format', type=str, default='pdf')
    parser.add_argument('--loeuf_file', type=str, default=None, help='gnomAD constraint metrics TSV; enables the LOEUF-decile figure')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    df, gene_all, funnel = load_pairs_with_prediction(args.beta_combined_file)
    gene_max_abs_z_eqtl_all = gene_all['max_abs_z_eqtl_all']
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
    # 1b. Same, with both axes on a symlog scale (log-distance from zero, original units)
    fig_bin_scatter(beta_pred, beta_eqtl, args.n_bins, args.cell_type,
                    'Mean predicted eQTL effect (Σ β_caQTL × β_link)', 'Mean observed eQTL effect (β_eQTL)',
                    '%s_%s_scatter_symlog.%s' % (args.output_prefix, label, fmt), symlog=True)

    # 2. Same thing on z-scores: observed z vs predicted z, by bin of predicted z
    fig_bin_scatter(z_pred, z_eqtl, args.n_bins, args.cell_type,
                    'Mean predicted z (β_combined / SE_combined)', 'Mean observed eQTL z (β_eQTL / SE_eQTL)',
                    '%s_z_%s_scatter.%s' % (args.output_prefix, label, fmt), fit_line=False)

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

    # 10. Same scatter restricted to confident predictions only, with no restriction on the eQTL z
    for t in [5, 8]:
        confident_pred = np.abs(z_pred) > t
        if confident_pred.sum() > 2:
            fig_effect_size_density(beta_pred[confident_pred], beta_eqtl[confident_pred], args.cell_type,
                                    '%s_effect_size_scatter_pred_z%g.png' % (args.output_prefix, t),
                                    note='|z_pred| > %g (no restriction on z_eQTL), n = %s pairs' % (t, format(int(confident_pred.sum()), ',')))
        else:
            print("No pairs with |z_pred| > %g; skipping that scatter" % t, flush=True)

    # 10b. Density scatter of z-scores (unit-free): all pairs, then restricted to confident predictions only
    z_labels_xy = dict(xlabel='Predicted z (β_combined / SE_combined)', ylabel='Observed eQTL z (β_eQTL / SE_eQTL)')
    fig_effect_size_density(z_pred, z_eqtl, args.cell_type, '%s_z_scatter.png' % args.output_prefix,
                            note='All pairs with a prediction, n = %s' % format(len(z_pred), ','), **z_labels_xy)
    for t in [5, 8]:
        confident_pred = np.abs(z_pred) > t
        if confident_pred.sum() > 2:
            fig_effect_size_density(z_pred[confident_pred], z_eqtl[confident_pred], args.cell_type,
                                    '%s_z_scatter_pred_z%g.png' % (args.output_prefix, t),
                                    note='|z_pred| > %g (no restriction on z_eQTL), n = %s pairs' % (t, format(int(confident_pred.sum()), ',')),
                                    **z_labels_xy)

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
    genes = per_gene_table(df, gene_all)
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
    loeuf = load_loeuf_deciles(args.loeuf_file, gene_all.index) if args.loeuf_file is not None else None
    summary = fig_moment_correlation(df, args.cell_type, '%s_moment_correlation.%s' % (args.output_prefix, fmt), loeuf=loeuf)
    print(summary[['n', 'r_naive', 'r_true', 'r_true_se', 'slope_true', 'slope_true_se', 'reliability_x', 'reliability_y']].to_string(), flush=True)

    # 25. QQ plot of z_pred among pairs with no observed eQTL signal: should be N(0,1) if the SE is right and nothing leaks
    summary = fig_null_qq(z_pred, z_eqtl, args.cell_type, '%s_null_qq.%s' % (args.output_prefix, fmt))
    print(summary.to_string(index=False), flush=True)

    # 26. Prediction coverage funnel: pairs and genes surviving each stage
    n_genes_conf = lambda m: df.loc[m, 'gene_id'].nunique()
    stages = [('Tested by eQTL study', funnel['tested'][0], funnel['tested'][1]),
              ('Has a prediction (≥ 1 linked peak with caQTL)', funnel['with_prediction'][0], funnel['with_prediction'][1]),
              ('Finite SE and variance', funnel['finite_se'][0], funnel['finite_se'][1]),
              ('|z_pred| > 2', int((np.abs(z_pred) > 2).sum()), n_genes_conf(np.abs(z_pred) > 2)),
              ('|z_pred| > 4', int((np.abs(z_pred) > 4).sum()), n_genes_conf(np.abs(z_pred) > 4))]
    summary = fig_coverage_funnel(stages, args.cell_type, '%s_coverage_funnel.%s' % (args.output_prefix, fmt))
    print(summary.to_string(index=False), flush=True)

    # 27. Delta-method variance diagnostics: conservative vs unbiased variance, negative-variance rate by n_peaks
    summary = fig_variance_diagnostics(df, args.cell_type, '%s_variance_diagnostics.png' % args.output_prefix)
    print(summary.to_string(index=False), flush=True)

    # 28. Shrinkage calibration: within-|z_pred|-bin slope of beta_eQTL on beta_pred vs slope_true x bin reliability
    summary = fig_shrinkage_check(df, args.cell_type, '%s_shrinkage_check.%s' % (args.output_prefix, fmt))
    print(summary.to_string(index=False), flush=True)

    # 29. Heteroskedasticity: residual variance around the calibration line vs eQTL sampling variance, by |beta_pred|
    summary = fig_heteroskedasticity(df, args.cell_type, '%s_heteroskedasticity.%s' % (args.output_prefix, fmt))
    print(summary.to_string(index=False), flush=True)

    # 20-24. Gene constraint (LOEUF decile) analyses; all use per-gene Bonferroni calls rather than a fixed |z| cutoff
    if args.loeuf_file is not None:
        alpha = BONF_ALPHAS[0]
        calls = gene_calls(genes, gene_all, alpha)
        genes = genes.join(loeuf).join(calls[['eqtl_p_bonf', 'eqtl_call', 'pred_p_bonf', 'pred_call']])
        genes.to_csv('%s_per_gene.tsv' % args.output_prefix, sep='\t')
        print("Bonferroni p < %g: eQTL calls %d of %d tested genes; prediction calls %d of %d genes with a prediction"
              % (alpha, int(calls['eqtl_call'].sum()), len(calls), int(calls['pred_call'].sum()), int(calls['has_pred'].sum())), flush=True)

        # 20. Fraction of genes with a QTL call (observed eQTL vs chromatin-predicted), by LOEUF decile
        summary = fig_qtl_fraction_by_loeuf_decile(genes, gene_all, loeuf, BONF_ALPHAS, args.cell_type,
                                                   '%s_qtl_fraction_by_loeuf_decile.%s' % (args.output_prefix, fmt))
        print(summary.to_string(index=False), flush=True)

        # 21. Rescue: chromatin-predicted calls among genes with no eQTL call, by LOEUF decile
        summary = fig_rescue_by_loeuf_decile(genes, gene_all, loeuf, BONF_ALPHAS, args.cell_type,
                                             '%s_rescue_by_loeuf_decile.%s' % (args.output_prefix, fmt))
        print(summary.to_string(index=False), flush=True)

        # 22. Effect size at the strongest variant by LOEUF decile (Kanai et al. Fig. 5 analogue), split into layers when available
        summary = fig_effect_size_by_loeuf_decile(genes, calls, loeuf, alpha, args.cell_type,
                                                  '%s_effect_size_by_loeuf_decile.%s' % (args.output_prefix, fmt))
        print(summary.to_string(index=False), flush=True)

        # 23. Localization by constraint: fraction of genes whose eQTL lead is in the top 5% by |z_pred|, by LOEUF decile
        group20 = decile_index(loeuf, genes20.index)
        summary = fig_fraction_by_group(group20, DECILE_LABELS,
                                        [('Lead |z_eQTL| > 4', is_egene & (group20 >= 0), lead_top5),
                                         ('Lead |z_eQTL| ≤ 4', ~is_egene & (group20 >= 0), lead_top5)],
                                        args.cell_type, 'LOEUF decile (1 = most constrained); genes with ≥ 20 predicted variants',
                                        'Fraction of genes with eQTL lead in top 5% by |z_pred|',
                                        '%s_lead_variant_top5_by_loeuf_decile.%s' % (args.output_prefix, fmt),
                                        ref_line=0.05, ref_label='uniform')
        print(summary.to_string(index=False), flush=True)

        # 24. Sign concordance among pairs confident on both sides, by LOEUF decile of the gene
        cat_decile = loeuf['loeuf_decile'].reindex(df['gene_id'].cat.categories).to_numpy(dtype=float)
        pair_group = np.where(np.isnan(cat_decile), -1, np.nan_to_num(cat_decile, nan=0) - 1).astype(int)[df['gene_id'].cat.codes.to_numpy()]
        confident_eqtl = (np.abs(z_eqtl) > 4) & (pair_group >= 0)
        summary = fig_fraction_by_group(pair_group, DECILE_LABELS,
                                        [('|z_eQTL| > 4, |z_pred| > 3', confident_eqtl & (np.abs(z_pred) > 3), same_sign),
                                         ('|z_eQTL| > 4, |z_pred| > 5', confident_eqtl & (np.abs(z_pred) > 5), same_sign)],
                                        args.cell_type, 'LOEUF decile of the gene (1 = most constrained); n pairs below',
                                        'Fraction of pairs with concordant sign',
                                        '%s_sign_concordance_by_loeuf_decile.%s' % (args.output_prefix, fmt),
                                        ref_line=0.5, ref_label='null')
        print(summary.to_string(index=False), flush=True)
    else:
        print("No --loeuf_file given; skipping the LOEUF-decile figures", flush=True)
