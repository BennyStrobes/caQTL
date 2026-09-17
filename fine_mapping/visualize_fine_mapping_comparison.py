import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Plots comparing eQTL-only SuSiE fine-mapping with caQTL-mediated fine-mapping on the same genes.
# Inputs are the two results files (gene_id, variant_ids_file, pips, ...). Genes are joined by gene_id.
# Genes with / without linked peaks are separated using the K column of the mediated run's gene convergence file
# (<mediated stem>_gene_convergence.txt); genes without peaks should sit on the diagonal (eQTL-only SuSiE).
# If the mediated run's components file (<mediated stem>_components.txt) exists, direct vs mediated PIPs and
# link posteriors are plotted too. Output: <output_stem>_<plot>.pdf and <output_stem>_summary.txt

parser = argparse.ArgumentParser()
parser.add_argument('--eqtl_only_results_file', type=str)
parser.add_argument('--mediated_results_file', type=str)
parser.add_argument('--output_stem', type=str)
parser.add_argument('--pip_thresholds', type=str, default='0.5,0.9,0.95')
parser.add_argument('--n_bootstrap', type=int, default=1000)  # gene-level bootstrap replicates for the 95% CIs on variant counts
args = parser.parse_args()
pip_thresholds = [float(x) for x in args.pip_thresholds.split(',')]


def load_pips(results_file):
    pips = {}
    f = open(results_file)
    head_count = 0
    for line in f:
        data = line.rstrip().split('\t')
        if head_count == 0:
            head_count += 1
            continue
        pips[data[0]] = np.array(data[2].split(','), dtype=float)
    f.close()
    return pips


def load_gene_num_peaks(gene_convergence_file):
    num_peaks = {}
    if not os.path.exists(gene_convergence_file):
        return num_peaks
    f = open(gene_convergence_file)
    head_count = 0
    for line in f:
        data = line.rstrip().split('\t')
        if head_count == 0:
            head_count += 1
            continue
        num_peaks[data[0]] = int(data[2])
    f.close()
    return num_peaks


def load_components(components_file):
    # Returns dict gene -> (direct_pip, mediated_pip, link_probs)
    comps = {}
    if not os.path.exists(components_file):
        return comps
    f = open(components_file)
    head_count = 0
    for line in f:
        data = line.rstrip().split('\t')
        if head_count == 0:
            head_count += 1
            header = data
            continue
        row = dict(zip(header, data))
        link_probs = np.array(row['link_probs'].split(','), dtype=float) if row['link_probs'] != '' else np.zeros(0)
        # P(link non-zero and mediates): Model D / E write it separately; in Model A it equals link_probs
        link_mediates = np.array(row['link_inclusion_probs'].split(','), dtype=float) if row.get('link_inclusion_probs', '') != '' else link_probs.copy()
        comps[row['gene_id']] = (np.load(row['direct_pip_file']), np.load(row['mediated_pip_file']), link_probs, link_mediates)
    f.close()
    return comps


def num_variants_for_coverage(pips, coverage=0.95):
    # Number of top variants needed for their PIPs to sum to `coverage` (fine-mapping resolution; nan if total PIP < coverage)
    if np.sum(pips) < coverage:
        return np.nan
    cum = np.cumsum(np.sort(pips)[::-1])
    return int(np.searchsorted(cum, coverage) + 1)


#######################
# Load and join
#######################
eqtl_pips = load_pips(args.eqtl_only_results_file)
med_pips = load_pips(args.mediated_results_file)
mediated_stem = os.path.splitext(args.mediated_results_file)[0]
num_peaks = load_gene_num_peaks(mediated_stem + '_gene_convergence.txt')
comps = load_components(mediated_stem + '_components.txt')

genes = [g for g in med_pips if g in eqtl_pips]
if len(genes) == 0:
    print('assumption error: no genes shared between the two results files')
    sys.exit(1)
for g in genes:
    if len(eqtl_pips[g]) != len(med_pips[g]):
        print('assumption error: different numbers of variants for gene ' + g)
        sys.exit(1)
has_peaks = np.array([num_peaks.get(g, -1) > 0 for g in genes])
peaks_known = len(num_peaks) > 0

# Stacked per-variant arrays
v_eqtl = np.concatenate([eqtl_pips[g] for g in genes])
v_med = np.concatenate([med_pips[g] for g in genes])
v_has_peaks = np.concatenate([np.ones(len(eqtl_pips[g]), dtype=bool) * has_peaks[i] for i, g in enumerate(genes)])
# Per-gene arrays
g_max_eqtl = np.array([np.max(eqtl_pips[g]) for g in genes])
g_max_med = np.array([np.max(med_pips[g]) for g in genes])
g_cov_eqtl = np.array([num_variants_for_coverage(eqtl_pips[g]) for g in genes])
g_cov_med = np.array([num_variants_for_coverage(med_pips[g]) for g in genes])

group_masks = [('genes with linked peaks', has_peaks), ('genes without linked peaks', ~has_peaks)] if peaks_known else [('all genes', np.ones(len(genes), dtype=bool))]
vgroup_masks = [('genes with linked peaks', v_has_peaks), ('genes without linked peaks', ~v_has_peaks)] if peaks_known else [('all genes', np.ones(len(v_eqtl), dtype=bool))]


#######################
# Plot 1: per-variant PIP scatter, eQTL-only vs mediated
#######################
fig, axes = plt.subplots(1, len(vgroup_masks), figsize=(5.0 * len(vgroup_masks), 4.8), squeeze=False)
for ax, (name, mask) in zip(axes[0], vgroup_masks):
    ax.scatter(v_eqtl[mask], v_med[mask], s=4, alpha=0.3, color='steelblue', rasterized=True)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=0.8)
    ax.set_xlabel('eQTL-only PIP')
    ax.set_ylabel('caQTL-mediated PIP')
    ax.set_title(name + ' (' + str(int(np.sum(mask))) + ' variants)', fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
fig.savefig(args.output_stem + '_variant_pip_scatter.pdf')
plt.close(fig)


#######################
# Plot 2: numbers of variants above PIP thresholds, per method, with 95% CIs from a gene-level bootstrap
# (genes are the sampling unit; variants within a gene are not independent)
#######################
def bootstrap_count_ci(per_gene_counts, n_bootstrap, rng):
    # per_gene_counts: (n_genes, n_thresholds). Returns (lower, upper) of the summed counts, each length n_thresholds
    n_genes_here = per_gene_counts.shape[0]
    if n_genes_here == 0:
        return np.zeros(per_gene_counts.shape[1]), np.zeros(per_gene_counts.shape[1])
    idx = rng.integers(0, n_genes_here, size=(n_bootstrap, n_genes_here))
    sums = np.array([np.sum(per_gene_counts[idx[b]], axis=0) for b in range(n_bootstrap)])
    return np.percentile(sums, 2.5, axis=0), np.percentile(sums, 97.5, axis=0)

rng = np.random.default_rng(0)
gene_counts_eqtl = np.array([[np.sum(eqtl_pips[g] > t) for t in pip_thresholds] for g in genes])   # genes x thresholds
gene_counts_med = np.array([[np.sum(med_pips[g] > t) for t in pip_thresholds] for g in genes])
fig, axes = plt.subplots(1, len(group_masks), figsize=(5.0 * len(group_masks), 4.2), squeeze=False)
summary_lines = ['group\tpip_threshold\tn_variants_eqtl_only\tci95_eqtl_only\tn_variants_caqtl_mediated\tci95_caqtl_mediated']
for ax, (name, mask) in zip(axes[0], group_masks):
    n_eqtl = np.sum(gene_counts_eqtl[mask], axis=0)
    n_med = np.sum(gene_counts_med[mask], axis=0)
    lo_eqtl, hi_eqtl = bootstrap_count_ci(gene_counts_eqtl[mask], args.n_bootstrap, rng)
    lo_med, hi_med = bootstrap_count_ci(gene_counts_med[mask], args.n_bootstrap, rng)
    x = np.arange(len(pip_thresholds))
    ax.bar(x - 0.2, n_eqtl, width=0.4, label='eQTL-only', color='grey', yerr=[n_eqtl - lo_eqtl, hi_eqtl - n_eqtl], capsize=3, error_kw={'linewidth': 1.0})
    ax.bar(x + 0.2, n_med, width=0.4, label='caQTL-mediated', color='steelblue', yerr=[n_med - lo_med, hi_med - n_med], capsize=3, error_kw={'linewidth': 1.0})
    ax.set_xticks(x)
    ax.set_xticklabels(['PIP > ' + str(t) for t in pip_thresholds])
    ax.set_ylabel('number of variants (95% CI, gene bootstrap)')
    ax.set_title(name + ' (' + str(int(np.sum(mask))) + ' genes)', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for i, t in enumerate(pip_thresholds):
        summary_lines.append(name + '\t' + str(t) + '\t' + str(int(n_eqtl[i])) + '\t' + '%.0f-%.0f' % (lo_eqtl[i], hi_eqtl[i]) + '\t' + str(int(n_med[i])) + '\t' + '%.0f-%.0f' % (lo_med[i], hi_med[i]))
fig.tight_layout()
fig.savefig(args.output_stem + '_variant_pip_threshold_counts.pdf')
plt.close(fig)


#######################
# Plot 3: per-gene maximum PIP and number of genes with a confidently fine-mapped variant
#######################
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
for name, mask in group_masks:
    axes[0].scatter(g_max_eqtl[mask], g_max_med[mask], s=8, alpha=0.5, label=name, rasterized=True)
axes[0].plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=0.8)
axes[0].set_xlabel('eQTL-only: max PIP in gene')
axes[0].set_ylabel('caQTL-mediated: max PIP in gene')
axes[0].legend(frameon=False, fontsize=8)
summary_lines.append('')
summary_lines.append('group\tpip_threshold\tn_genes_max_pip_above_eqtl_only\tn_genes_max_pip_above_caqtl_mediated\tn_genes')
labels = []
vals_eqtl = []
vals_med = []
for name, mask in group_masks:
    for t in pip_thresholds:
        a = int(np.sum(g_max_eqtl[mask] > t))
        b = int(np.sum(g_max_med[mask] > t))
        labels.append(name.replace('genes ', '') + '\nmax PIP > ' + str(t))
        vals_eqtl.append(a)
        vals_med.append(b)
        summary_lines.append(name + '\t' + str(t) + '\t' + str(a) + '\t' + str(b) + '\t' + str(int(np.sum(mask))))
x = np.arange(len(labels))
axes[1].bar(x - 0.2, vals_eqtl, width=0.4, label='eQTL-only', color='grey')
axes[1].bar(x + 0.2, vals_med, width=0.4, label='caQTL-mediated', color='steelblue')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=6)
axes[1].set_ylabel('number of genes')
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(args.output_stem + '_gene_max_pip.pdf')
plt.close(fig)


#######################
# Plot 4: PIP changes (mediated - eQTL-only) for variants with signal in either method, genes with peaks only
#######################
fig, ax = plt.subplots(figsize=(5.5, 4.2))
mask = v_has_peaks & ((v_eqtl > 0.05) | (v_med > 0.05))
diff = v_med[mask] - v_eqtl[mask]
ax.hist(diff, bins=np.linspace(-1.0, 1.0, 61), color='steelblue')   # fixed edges: robust to degenerate (zero-range) data
ax.set_yscale('log')
ax.set_xlabel('caQTL-mediated PIP - eQTL-only PIP')
ax.set_ylabel('number of variants')
ax.set_title('variants with PIP > 0.05 in either method (' + str(int(np.sum(mask))) + ')' + (', genes with peaks' if peaks_known else ''), fontsize=9)
fig.tight_layout()
fig.savefig(args.output_stem + '_variant_pip_change_hist.pdf')
plt.close(fig)
summary_lines.append('')
summary_lines.append('variants with PIP > 0.05 in either method (genes with peaks): ' + str(int(np.sum(mask))))
summary_lines.append('  PIP increased by > 0.1: ' + str(int(np.sum(diff > 0.1))) + '; decreased by > 0.1: ' + str(int(np.sum(diff < -0.1))))


#######################
# Plot 5: fine-mapping resolution: number of variants to reach 95% cumulative PIP, per gene
#######################
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
ok = np.isfinite(g_cov_eqtl) & np.isfinite(g_cov_med)
# log2 counts on linear axes (jittered), ticks labelled with the counts
for name, mask in group_masks:
    m = mask & ok
    if np.sum(m) > 0:
        jitter = np.random.default_rng(0).uniform(-0.1, 0.1, size=(2, int(np.sum(m))))
        axes[0].scatter(np.log2(g_cov_eqtl[m]) + jitter[0], np.log2(g_cov_med[m]) + jitter[1], s=8, alpha=0.5, label=name + ' (' + str(int(np.sum(m))) + ')', rasterized=True)
lim = int(np.ceil(np.log2(np.nanmax(np.concatenate([g_cov_eqtl[ok], g_cov_med[ok]]))))) if np.sum(ok) > 0 else 1
lim = max(lim, 1)
axes[0].plot([0, lim], [0, lim], color='grey', linestyle='--', linewidth=0.8)
ticks = np.arange(0, lim + 1)
axes[0].set_xticks(ticks)
axes[0].set_xticklabels([str(2**k) for k in ticks])
axes[0].set_yticks(ticks)
axes[0].set_yticklabels([str(2**k) for k in ticks])
axes[0].set_xlabel('eQTL-only: variants for 95% cumulative PIP')
axes[0].set_ylabel('caQTL-mediated: variants for 95% cumulative PIP')
axes[0].legend(frameon=False, fontsize=8)
m = has_peaks & ok if peaks_known else ok
if np.sum(m) > 0:
    ratio = np.log2(g_cov_med[m] / g_cov_eqtl[m])
    lim_r = max(float(np.max(np.abs(ratio))), 0.5)
    axes[1].hist(ratio, bins=np.linspace(-lim_r, lim_r, 41), color='steelblue')
    axes[1].set_yscale('log')
    summary_lines.append('')
    summary_lines.append('genes (with peaks) where both methods reach 95% cumulative PIP: ' + str(int(np.sum(m))))
    summary_lines.append('  resolution improved (fewer variants): ' + str(int(np.sum(ratio < 0))) + '; worsened: ' + str(int(np.sum(ratio > 0))) + '; unchanged: ' + str(int(np.sum(ratio == 0))))
axes[1].set_xlabel('log2( variants for 95% PIP: caQTL-mediated / eQTL-only )')
axes[1].set_ylabel('number of genes')
axes[1].set_title('genes with peaks' if peaks_known else 'all genes', fontsize=10)
fig.tight_layout()
fig.savefig(args.output_stem + '_gene_resolution.pdf')
plt.close(fig)


#######################
# Plot 6 (if components exist): direct vs mediated PIPs, mediated PIP vs eQTL-only PIP, link posteriors
#######################
if len(comps) > 0:
    cgenes = [g for g in genes if g in comps and num_peaks.get(g, 1) > 0]
    c_direct = np.concatenate([comps[g][0] for g in cgenes])
    c_med = np.concatenate([comps[g][1] for g in cgenes])
    c_eqtl = np.concatenate([eqtl_pips[g] for g in cgenes])
    link_probs = np.concatenate([comps[g][2] for g in cgenes])
    link_mediates = np.concatenate([comps[g][3] for g in cgenes])
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    axes[0].scatter(c_direct, c_med, s=4, alpha=0.3, color='steelblue', rasterized=True)
    axes[0].set_xlabel('direct PIP')
    axes[0].set_ylabel('mediated PIP')
    axes[0].set_title('caQTL-mediated model components (' + str(len(cgenes)) + ' genes with peaks)', fontsize=9)
    axes[1].scatter(c_eqtl, c_med, s=4, alpha=0.3, color='steelblue', rasterized=True)
    axes[1].set_xlabel('eQTL-only PIP')
    axes[1].set_ylabel('mediated PIP')
    axes[1].set_title('is mediation supported by the eQTL alone?', fontsize=9)
    axes[2].hist(link_probs, bins=np.linspace(0.0, 1.0, 41), color='lightgrey', label='P(link non-zero)')
    axes[2].hist(link_mediates, bins=np.linspace(0.0, 1.0, 41), color='steelblue', alpha=0.8, label='P(link non-zero and mediates)')
    axes[2].set_yscale('log')
    axes[2].set_xlabel('posterior probability')
    axes[2].set_ylabel('number of peak-gene links')
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].set_title(str(len(link_probs)) + ' links; non-zero > 0.9: ' + str(int(np.sum(link_probs > 0.9))) + '; mediates > 0.9: ' + str(int(np.sum(link_mediates > 0.9))), fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output_stem + '_components.pdf')
    plt.close(fig)
    summary_lines.append('')
    summary_lines.append('links: ' + str(len(link_probs)) + '; P(non-zero) > 0.5: ' + str(int(np.sum(link_probs > 0.5))) + '; > 0.9: ' + str(int(np.sum(link_probs > 0.9))) + ' | P(non-zero and mediates) > 0.5: ' + str(int(np.sum(link_mediates > 0.5))) + '; > 0.9: ' + str(int(np.sum(link_mediates > 0.9))))
    summary_lines.append('variants with mediated PIP > 0.5: ' + str(int(np.sum(c_med > 0.5))) + '; of these with eQTL-only PIP < 0.1: ' + str(int(np.sum((c_med > 0.5) & (c_eqtl < 0.1)))))

open(args.output_stem + '_summary.txt', 'w').write('\n'.join(summary_lines) + '\n')
print('plots written to ' + args.output_stem + '_*.pdf (' + str(len(genes)) + ' genes, ' + str(int(np.sum(has_peaks))) + ' with linked peaks)')
