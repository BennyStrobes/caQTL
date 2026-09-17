import sys
import os
import argparse
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Stacked Manhattan plots (eQTL on top, the relevant peak's caQTL below) for example loci where the caQTL-mediated
# fine-mapping gives a variant a high PIP but eQTL-only fine-mapping does not. The variant is highlighted in both panels,
# points are coloured by LD r^2 with it, and its PIPs / link posterior are annotated.
# The relevant peak is the one contributing most to the variant's mediated PIP (link prob x caQTL PIP, from the mediated
# run's components files); without components it is the linked peak with the largest |caQTL z| at the variant.
# Output: <output_stem>_<gene>_<variant>.pdf per example and <output_stem>_examples.txt listing them (the output directory is created if needed).

parser = argparse.ArgumentParser()
parser.add_argument('--eqtl_only_results_file', type=str)
parser.add_argument('--mediated_results_file', type=str)
parser.add_argument('--fm_input_summary_file', type=str)  # fine-mapping input summary (z-scores, caQTL sumstats, LD per gene)
parser.add_argument('--output_stem', type=str)
parser.add_argument('--caqtl_sumstat_summary_file', type=str, default=None)  # optional: <cell>_caqtl_mediated_sumstats_summary.txt, for peak IDs
parser.add_argument('--min_eqtl_only_pip', type=float, default=0.0)  # e.g. 0.1 with --max_eqtl_only_pip 0.5 selects modest -> high changes rather than extremes
parser.add_argument('--max_eqtl_only_pip', type=float, default=0.1)
parser.add_argument('--min_mediated_pip', type=float, default=0.9)
parser.add_argument('--n_examples', type=int, default=4)  # one variant per gene
parser.add_argument('--rank_by', type=str, default='pip_change', choices=['pip_change', 'eqtl_only_pip', 'random'])  # how genes (and the variant within a gene) are chosen: largest mediated - eQTL-only PIP, largest eQTL-only PIP, or a random sample of qualifying genes
parser.add_argument('--gene_ids', type=str, default=None)  # optional comma-separated genes to plot instead of the automatic selection (top variant per gene)
parser.add_argument('--all_qualifying', action='store_true', default=False)  # plot every variant meeting the PIP criteria (all genes, possibly several per gene); ignores n_examples and rank_by
args = parser.parse_args()


def load_pips(results_file):
    pips = {}
    variant_files = {}
    f = open(results_file)
    head_count = 0
    for line in f:
        data = line.rstrip().split('\t')
        if head_count == 0:
            head_count += 1
            continue
        pips[data[0]] = np.array(data[2].split(','), dtype=float)
        variant_files[data[0]] = data[1]
    f.close()
    return pips, variant_files


def load_fm_input_summary(summary_file):
    rows = {}
    f = open(summary_file)
    head_count = 0
    for line in f:
        data = line.rstrip().split('\t')
        if head_count == 0:
            head_count += 1
            header = data
            continue
        rows[data[0]] = dict(zip(header, data))
    f.close()
    return rows


def load_components(mediated_results_file):
    comps = {}
    components_file = os.path.splitext(mediated_results_file)[0] + '_components.txt'
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
        link_probs = np.array(row['link_probs'].split(','), dtype=float) if row['link_probs'] != '' else np.zeros(0)   # P(link non-zero)
        # P(link non-zero AND mediates) = what the mediated PIP uses; Model D writes it separately, in Model A it equals link_probs
        link_mediates = np.array(row['link_inclusion_probs'].split(','), dtype=float) if row.get('link_inclusion_probs', '') != '' else link_probs.copy()
        kept = np.array(row['kept_link_indices'].split(','), dtype=int) if row.get('kept_link_indices', '') != '' else np.arange(len(link_probs))
        comps[row['gene_id']] = {'link_probs': link_probs, 'link_mediates': link_mediates, 'kept': kept, 'caqtl_pip': np.load(row['caqtl_pip_file']), 'direct_pip': np.load(row['direct_pip_file']), 'mediated_pip': np.load(row['mediated_pip_file'])}
    f.close()
    return comps


def load_peak_ids(caqtl_summary_file):
    peak_ids = {}
    if caqtl_summary_file is None or not os.path.exists(caqtl_summary_file):
        return peak_ids
    f = open(caqtl_summary_file)
    head_count = 0
    for line in f:
        data = line.rstrip().split('\t')
        if head_count == 0:
            head_count += 1
            header = data
            continue
        row = dict(zip(header, data))
        pf = row['Peak_IDs_File']
        if os.path.exists(pf) and os.path.getsize(pf) > 0:
            peak_ids[row['Gene_Name']] = np.loadtxt(pf, dtype=str, ndmin=1)
    f.close()
    return peak_ids


def variant_positions(variant_ids):
    # variant ids are chr_pos_ref_alt; fall back to the index if they cannot be parsed
    try:
        return np.array([int(v.split('_')[1]) for v in variant_ids], dtype=float), True
    except (IndexError, ValueError):
        return np.arange(len(variant_ids), dtype=float), False


def neglog10p(z):
    # -log10 of the two-sided normal p-value, via the log survival function so very large |z| do not underflow to a cap
    return -(norm.logsf(np.abs(z)) + np.log(2.0)) / np.log(10.0)


#######################
# Load and pick examples
#######################
eqtl_pips, variant_files = load_pips(args.eqtl_only_results_file)
med_pips, _ = load_pips(args.mediated_results_file)
fm_rows = load_fm_input_summary(args.fm_input_summary_file)
comps = load_components(args.mediated_results_file)
peak_ids = load_peak_ids(args.caqtl_sumstat_summary_file)

output_dir = os.path.dirname(args.output_stem)
if output_dir != '' and not os.path.exists(output_dir):
    os.makedirs(output_dir)

examples = []   # (gene, variant index, eqtl-only pip, mediated pip)
if args.all_qualifying:
    for g in med_pips:
        if g not in eqtl_pips or g not in fm_rows:
            continue
        ok = (eqtl_pips[g] >= args.min_eqtl_only_pip) & (eqtl_pips[g] <= args.max_eqtl_only_pip) & (med_pips[g] >= args.min_mediated_pip)
        for j in np.where(ok)[0]:
            examples.append((g, int(j), eqtl_pips[g][j], med_pips[g][j]))
elif args.gene_ids is not None:
    for g in args.gene_ids.split(','):
        if g not in med_pips or g not in eqtl_pips:
            print('assumption error: gene ' + g + ' not in both results files')
            sys.exit(1)
        j = int(np.argmax(med_pips[g] - eqtl_pips[g]))
        examples.append((g, j, eqtl_pips[g][j], med_pips[g][j]))
else:
    candidates = []
    for g in med_pips:
        if g not in eqtl_pips or g not in fm_rows:
            continue
        ok = (eqtl_pips[g] >= args.min_eqtl_only_pip) & (eqtl_pips[g] <= args.max_eqtl_only_pip) & (med_pips[g] >= args.min_mediated_pip)
        if np.sum(ok) == 0:
            continue
        score = med_pips[g] - eqtl_pips[g] if args.rank_by == 'pip_change' else eqtl_pips[g]
        j = int(np.argmax(np.where(ok, score, -np.inf)))
        candidates.append((score[j], g, j))
    if args.rank_by == 'random':
        rng = np.random.default_rng(0)
        candidates = [candidates[i] for i in rng.permutation(len(candidates))]
    else:
        candidates.sort(reverse=True)
    for score, g, j in candidates[:args.n_examples]:
        examples.append((g, j, eqtl_pips[g][j], med_pips[g][j]))
print(str(len(examples)) + ' example loci selected', flush=True)

t = open(args.output_stem + '_examples.txt', 'w')
t.write('gene_id\tvariant_id\tvariant_index\teqtl_only_pip\tcaqtl_mediated_pip\tpeak\tlink_nonzero_prob\tlink_mediates_given_nonzero_prob\tlink_mediates_prob\tcaqtl_pip\teqtl_z\tcaqtl_z\tplot_file\n')

#######################
# Plot each example
#######################
for g, j, pip_e, pip_m in examples:
    row = fm_rows[g]
    variant_ids = np.loadtxt(variant_files[g], dtype=str, ndmin=1)
    pos, pos_parsed = variant_positions(variant_ids)
    z_e = np.load(row['eQTL_Effects_File']) / np.load(row['eQTL_SE_File'])
    bA = np.atleast_2d(np.load(row['caQTL_Effects_File']))
    sA = np.atleast_2d(np.load(row['caQTL_SE_File']))
    K = bA.shape[0]
    if K == 0:
        print('gene ' + g + ' has no linked peaks; skipping')
        continue
    zA = bA / sA
    imputed = np.atleast_2d(np.load(row['caQTL_Imputed_Mask_File'])).astype(bool) if 'caQTL_Imputed_Mask_File' in row and os.path.exists(row['caQTL_Imputed_Mask_File']) else np.zeros(bA.shape, dtype=bool)
    R = np.load(row['LD_File'])
    r2 = R[:, j]**2

    # The relevant peak: largest P(link mediates) x caQTL PIP at the variant (components), else largest |caQTL z| at the variant
    link_prob = np.nan
    link_mediates = np.nan
    link_mediates_given_nonzero = np.nan
    caqtl_pip_at_variant = np.nan
    if g in comps:
        contrib = comps[g]['link_mediates'] * comps[g]['caqtl_pip'][:, j]
        k_fit = int(np.argmax(contrib))
        k = int(comps[g]['kept'][k_fit])     # row in the input caQTL matrices (bad links were dropped by the runner)
        link_prob = comps[g]['link_probs'][k_fit]
        link_mediates = comps[g]['link_mediates'][k_fit]
        link_mediates_given_nonzero = link_mediates / link_prob if link_prob > 1e-12 else np.nan   # posterior P(u_k = 1 | z_k = 1)
        caqtl_pip_at_variant = comps[g]['caqtl_pip'][k_fit, j]
        caqtl_pip_track = comps[g]['caqtl_pip'][k_fit]
    else:
        k = int(np.argmax(np.abs(zA[:, j])))
        caqtl_pip_track = None
    k_orig = int(row['Kept_Peak_Indices'].split(',')[k]) if row.get('Kept_Peak_Indices', '') != '' else k   # original peak index if the LD screen dropped peaks
    peak_label = peak_ids[g][k_orig] if g in peak_ids and k_orig < len(peak_ids[g]) else 'peak ' + str(k_orig)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    x = pos / 1e6 if pos_parsed else pos
    order = np.argsort(r2)   # draw high-LD points on top
    for ax, z, imp, title in [(axes[0], z_e, np.zeros(len(z_e), dtype=bool), 'eQTL: ' + g), (axes[1], zA[k], imputed[k], 'caQTL: ' + peak_label)]:
        y = neglog10p(z)
        sc = ax.scatter(x[order], y[order], c=r2[order], cmap='viridis', vmin=0, vmax=1, s=14, edgecolors='none', rasterized=True)
        if np.any(imp):
            ax.scatter(x[imp], y[imp], facecolors='none', edgecolors='black', s=18, linewidths=0.5, label='imputed caQTL')
        ax.scatter([x[j]], [y[j]], marker='D', s=70, facecolors='none', edgecolors='red', linewidths=1.5, zorder=5)
        ax.set_ylabel('-log10 p')
        ax.set_title(title, fontsize=10, loc='left')
    cbar = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label('LD r^2 with highlighted variant')
    if np.any(imputed[k]):
        axes[1].legend(frameon=False, fontsize=8, loc='upper right')
    axes[1].set_xlabel('position (Mb)' if pos_parsed else 'variant index')
    note = variant_ids[j] + ':  eQTL-only PIP %.2f, caQTL-mediated PIP %.2f' % (pip_e, pip_m) + ' | eQTL z %.1f, caQTL z %.1f' % (z_e[j], zA[k, j])
    if np.isfinite(link_prob):
        note = note + '\nP(link non-zero) %.2f, P(mediates | non-zero) %.2f, P(link non-zero and mediates) %.2f, caQTL PIP %.2f' % (link_prob, link_mediates_given_nonzero, link_mediates, caqtl_pip_at_variant)
    fig.suptitle(note, fontsize=9)
    fig.subplots_adjust(top=0.88)
    plot_file = args.output_stem + '_' + g + '_' + variant_ids[j] + '.pdf'
    fig.savefig(plot_file)
    plt.close(fig)
    t.write('\t'.join(map(str, [g, variant_ids[j], j, pip_e, pip_m, peak_label, link_prob, link_mediates_given_nonzero, link_mediates, caqtl_pip_at_variant, z_e[j], zA[k, j], plot_file])) + '\n')
t.close()
print('done', flush=True)
