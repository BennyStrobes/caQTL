import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

# Two-panel grant figure built from the summary TSVs written by visualize_beta_combined.py:
#   a. fraction of genes with a significant caQTL2G by LOEUF decile (Wilson CIs), one series, one call rule
#   b. observed vs predicted eQTL effect in equal-count bins of the prediction (symlog axes, block-jackknife CIs)

BLUE = '#0f8b8d'  # default point colour (deep teal, grant figure only); overridden by --color
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
GRID_GRAY = '#d9d8d4'

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,  # keep text editable in Illustrator
    'ps.fonttype': 42,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'xtick.major.pad': 1.5,
    'ytick.major.pad': 1.5,
})
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

TICK_FS = 5.5
LABEL_FS = 6.5
PANEL_FS = 8


def style_axes(ax):
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=TICK_FS, length=2, width=0.5)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color(INK)


def decade_label(v, _pos=None):
    # Tick label for a symlog axis: 0, or +/-10^k
    if v == 0:
        return '0'
    k = int(round(np.log10(abs(v))))
    return ('$-10^{%d}$' if v < 0 else '$10^{%d}$') % k


def draw_scatter(ax, summary, linthresh=(5e-5, 5e-4), x_label_min=1e-3):
    ax.axhline(0, color=GRID_GRAY, linewidth=0.5, zorder=0)
    ax.axvline(0, color=GRID_GRAY, linewidth=0.5, zorder=0)
    ax.errorbar(summary['x_mean'], summary['y_mean'], xerr=summary['x_ci95'], yerr=summary['y_ci95'],
                fmt='o', color=BLUE, ecolor=BLUE, elinewidth=0.5, capsize=0, markersize=2.2,
                markeredgecolor='white', markeredgewidth=0.3, zorder=3)
    ax.set_xscale('symlog', linthresh=linthresh[0], linscale=0.3)
    ax.set_yscale('symlog', linthresh=linthresh[1], linscale=0.3)
    # Decade ticks outside the linear window, plus zero. On x the panel is too narrow to label the decades next to
    # zero without collisions, so decades below x_label_min are drawn as unlabeled minor ticks.
    for axis, thresh, vmax, label_min in [(ax.xaxis, linthresh[0], summary['x_mean'].abs().max(), x_label_min),
                                          (ax.yaxis, linthresh[1], summary['y_mean'].abs().max(), 0)]:
        decades = 10.0 ** np.arange(np.ceil(np.log10(thresh)), np.floor(np.log10(vmax)) + 1)
        major = decades[decades >= label_min]
        minor = decades[decades < label_min]
        axis.set_ticks(np.concatenate([-major[::-1], [0], major]))
        axis.set_ticks(np.concatenate([-minor[::-1], minor]), minor=True)
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(decade_label))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis='x', which='minor', length=1.5, width=0.5, colors=INK_SECONDARY)
    ax.margins(0.08)
    ax.set_xlabel('Average caQTL2G effect', color=INK, fontsize=LABEL_FS, labelpad=2)
    ax.set_ylabel('Average eQTL effect', color=INK, fontsize=LABEL_FS, labelpad=2)
    style_axes(ax)


def draw_loeuf(ax, summary):
    d = summary['loeuf_decile'].to_numpy()
    p = summary['fraction'].to_numpy()
    lo = np.maximum(p - summary['ci95_lower'].to_numpy(), 0)
    hi = np.maximum(summary['ci95_upper'].to_numpy() - p, 0)
    ax.errorbar(d, p, yerr=[lo, hi], fmt='o', color=BLUE, ecolor=BLUE, elinewidth=0.6, capsize=1.2, capthick=0.6,
                markersize=2.8, markeredgecolor='white', markeredgewidth=0.3, zorder=3)
    ax.set_xticks(d)
    ax.set_xticklabels([str(int(v)) for v in d])
    ax.set_xlim(0.4, 10.6)
    ax.set_ylim(0, max(summary['ci95_upper'].max() * 1.12, 0.05))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xlabel('LOEUF decile\n(1 = most constrained)', color=INK, fontsize=LABEL_FS, labelpad=2)
    ax.set_ylabel('Fraction of genes with a\nsignificant caQTL2G', color=INK, fontsize=LABEL_FS, labelpad=2)
    style_axes(ax)


def bh_qvalues(p):
    # Benjamini-Hochberg q-values (monotone step-up) for a vector of p-values, NaN preserved
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    tmp = np.empty(m)
    tmp[order] = ranked
    q[ok] = tmp
    return q


def wilson_ci(k, n, z=1.96):
    # Wilson score interval for k successes in n trials (vectorized)
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = k / n
        denom = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return p, center - half, center + half


def main():
    global BLUE
    parser = argparse.ArgumentParser()
    parser.add_argument('--scatter_summary', required=True, help='*_scatter_symlog[_centered]_summary.tsv from visualize_beta_combined.py')
    parser.add_argument('--loeuf_summary', required=True, help='*_qtl_fraction_by_loeuf_decile_summary.tsv from visualize_beta_combined.py')
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--alpha', type=float, default=0.001, help='per-gene Bonferroni level to show (one of the alphas in the summary)')
    parser.add_argument('--series', default='Chromatin-predicted (all genes)', help='series name in the LOEUF summary to show')
    parser.add_argument('--per_gene', default=None, help='*_per_gene.tsv from visualize_beta_combined.py; required for --call bh, '
                        'and used to count genes with a significant caQTL2G by Benjamini-Hochberg across genes at --fdr')
    parser.add_argument('--fdr', type=float, default=0.05)
    parser.add_argument('--call', choices=['bh', 'bonferroni'], default='bh',
                        help='gene call shown in panel a: BH across genes at --fdr on the per-gene Bonferroni p (needs --per_gene), '
                             'or the fixed per-gene Bonferroni p < --alpha from the LOEUF summary')
    parser.add_argument('--color', default=BLUE, help='point colour (hex) for both panels')
    parser.add_argument('--width', type=float, default=3.8, help='figure width in inches')
    parser.add_argument('--height', type=float, default=2.3, help='figure height in inches')
    args = parser.parse_args()

    scatter = pd.read_csv(args.scatter_summary, sep='\t')
    loeuf = pd.read_csv(args.loeuf_summary, sep='\t')
    sel = loeuf[(np.isclose(loeuf['alpha'], args.alpha)) & (loeuf['series'] == args.series)].sort_values('loeuf_decile')
    if len(sel) == 0:
        raise SystemExit('No rows in %s with alpha = %g and series = %r; available: %s'
                         % (args.loeuf_summary, args.alpha, args.series,
                            loeuf[['alpha', 'series']].drop_duplicates().to_string(index=False)))

    BLUE = args.color
    per_gene = pd.read_csv(args.per_gene, sep='\t', index_col=0) if args.per_gene is not None else None
    if args.call == 'bh':
        # Numerator: genes with a prediction whose BH q (across genes with a prediction) is below --fdr, by LOEUF decile.
        # Denominator: every tested gene with a LOEUF decile (from the summary; genes without a prediction count as not called).
        if per_gene is None:
            raise SystemExit('--call bh needs --per_gene')
        per_gene['pred_q'] = bh_qvalues(per_gene['pred_p_bonf'].to_numpy())
        dec = per_gene['loeuf_decile'].to_numpy(dtype=float)
        called = (per_gene['pred_q'].to_numpy() < args.fdr)
        n_called = np.array([int((called & (dec == d)).sum()) for d in sel['loeuf_decile']])
        frac, lo, hi = wilson_ci(n_called, sel['n_genes'].to_numpy())
        sel = pd.DataFrame({'loeuf_decile': sel['loeuf_decile'].to_numpy(), 'n_genes': sel['n_genes'].to_numpy(),
                            'n_called': n_called, 'fraction': frac, 'ci95_lower': lo, 'ci95_upper': hi})
        call_label = 'BH FDR < %g across genes with a prediction' % args.fdr
    else:
        call_label = 'per-gene Bonferroni p < %g' % args.alpha

    # Panel a: LOEUF decile fractions; panel b: effect-size scatter
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(args.width, args.height))
    draw_loeuf(ax_a, sel)
    draw_scatter(ax_b, scatter)
    fig.tight_layout(pad=0.3, w_pad=1.2, rect=(0, 0, 1, 0.955))
    # Panel labels in the headroom above each panel, left-aligned with the panel's y-axis label
    for ax, letter in [(ax_a, 'a'), (ax_b, 'b')]:
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        fig.text(bbox.x0, 0.995, letter, fontsize=PANEL_FS, fontweight='bold', color=INK, va='top', ha='left')
    fig.savefig(args.output_file, dpi=300)
    plt.close(fig)
    print('wrote %s' % args.output_file)
    # Gene counts behind panel a (genes with a LOEUF decile only)
    n_called, n_genes = int(sel['n_called'].sum()), int(sel['n_genes'].sum())
    print('panel a: %s, %s: %d of %d genes with a LOEUF decile (%.1f%%)'
          % (args.series, call_label, n_called, n_genes, 100.0 * n_called / max(n_genes, 1)))
    print(sel[['loeuf_decile', 'n_genes', 'n_called', 'fraction']].to_string(index=False))
    if per_gene is not None:
        # Gene-level count the standard way: BH across genes on the per-gene Bonferroni p (one hypothesis per gene with a
        # prediction; genes without a prediction cannot be called and are not hypotheses)
        g = per_gene
        for name, col in [('caQTL2G (pred_p_bonf)', 'pred_p_bonf'), ('eQTL (eqtl_p_bonf)', 'eqtl_p_bonf')]:
            if col not in g.columns:
                print('%s: no %s column in %s' % (name, col, args.per_gene))
                continue
            q = bh_qvalues(g[col].to_numpy())
            tested = np.isfinite(q)
            sig = tested & (q < args.fdr)
            line = '%s: BH FDR < %g: %d of %d genes with a prediction' % (name, args.fdr, int(sig.sum()), int(tested.sum()))
            if 'loeuf' in g.columns:
                has_loeuf = g['loeuf'].notna().to_numpy()
                line += '; %d of %d among genes with a LOEUF decile' % (int((sig & has_loeuf).sum()), int((tested & has_loeuf).sum()))
            line += '; Bonferroni p < %g for comparison: %d' % (args.alpha, int((g[col].to_numpy() < args.alpha).sum()))
            print(line)


if __name__ == '__main__':
    main()
