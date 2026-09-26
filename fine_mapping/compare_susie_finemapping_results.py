import sys
import numpy as np

# Compare two SuSiE fine-mapping results files (e.g. standard eQTL-only vs chromatin-informed) on the same genes
# Usage: python compare_susie_finemapping_results.py baseline_results_file informed_results_file
# Genes are stratified by whether any variant has a (gated) chromatin path in the informed results file


def load_in_results(results_file):
    gene_to_results = {}
    f = open(results_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            header = data
            continue
        pips = np.asarray(data[2].split(',')).astype(float)
        if data[3] == 'NA':
            cs_sizes = []
        else:
            cs_sizes = [len(cs.split(',')) for cs in data[3].split(';')]
        has_path = False
        if 'gated_path_probs' in header:
            has_path = np.sum(np.asarray(data[header.index('gated_path_probs')].split(',')).astype(float)) > 0.0
        gene_to_results[data[0]] = (pips, cs_sizes, has_path)
    f.close()
    return gene_to_results


def print_summary(gene_ids, gene_to_results, label):
    cs_sizes = np.hstack([gene_to_results[gene_id][1] for gene_id in gene_ids] + [[]])
    n_high_pip = np.sum([np.sum(gene_to_results[gene_id][0] > 0.9) for gene_id in gene_ids])
    n_mid_pip = np.sum([np.sum(gene_to_results[gene_id][0] > 0.5) for gene_id in gene_ids])
    if len(cs_sizes) > 0:
        cs_string = str(len(cs_sizes)) + ' credible sets (median size ' + str(np.median(cs_sizes)) + ', mean size ' + str(np.round(np.mean(cs_sizes), 2)) + ', ' + str(int(np.sum(cs_sizes == 1))) + ' of size 1, ' + str(int(np.sum(cs_sizes <= 5))) + ' of size <= 5)'
    else:
        cs_string = '0 credible sets'
    print('    ' + label + ': ' + cs_string + '; ' + str(n_high_pip) + ' variants with PIP > 0.9; ' + str(n_mid_pip) + ' variants with PIP > 0.5')


baseline = load_in_results(sys.argv[1])
informed = load_in_results(sys.argv[2])
gene_ids = np.asarray([gene_id for gene_id in informed if gene_id in baseline])
has_path = np.asarray([informed[gene_id][2] for gene_id in gene_ids])
for subset, subset_name in [(has_path, 'Genes with a variant with a chromatin path'), (has_path == False, 'Genes without')]:
    print(subset_name + ' (' + str(np.sum(subset)) + ' genes)')
    if np.sum(subset) == 0:
        continue
    print_summary(gene_ids[subset], baseline, 'baseline')
    print_summary(gene_ids[subset], informed, 'informed')
    pip_changes = np.hstack([informed[gene_id][0] - baseline[gene_id][0] for gene_id in gene_ids[subset]])
    print('    largest |PIP change| = ' + str(np.max(np.abs(pip_changes))) + '; variants with |PIP change| > 0.1: ' + str(np.sum(np.abs(pip_changes) > 0.1)))
