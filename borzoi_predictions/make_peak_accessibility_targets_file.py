import numpy as np
import os
import sys
import pdb
import re


# Blood accessibility panel: adult sorted mononuclear lineages (T, B, NK, monocyte).
# There is no whole-blood or PBMC DNase/ATAC track in Borzoi, and no neutrophil track,
# so this is the closest available composite of whole blood.
# Patterns are matched against the description text after the 'DNASE:' / 'ATAC:' prefix.
dnase_patterns = [
	r'CD4-positive, alpha-beta T cell',   # includes the naive thymus-derived CD4 track
	r'^T-helper \d+ cell',                # Th1, Th2, Th17
	r'^regulatory T cell',
	r'^T-cell',                           # pan T
	r'CD8-positive, alpha-beta T cell',
	r'^B cell',
	r'^natural killer cell',
	r'^CD14-positive monocyte',
]
atac_patterns = [
	r'^T lymphocyte 2 \(CD4\+\)$',
	r'^Naive T$',
	r'^T Lymphocyte 1 \(CD8\+\)$',
	r'^Natural Killer T$',
	r'^Memory B$',
]


def description_matches(description_text, patterns):
	for pattern in patterns:
		if re.search(pattern, description_text) is not None:
			return True
	return False


def make_peak_accessibility_targets_file(borzoi_target_file, output_file):
	f = open(borzoi_target_file)
	t = open(output_file, 'w')
	head_count = 0
	n_selected = 0
	for line in f:
		line = line.rstrip('\n')
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			t.write(line + '\n')
			continue
		# Columns: index, identifier, file, clip, clip_soft, scale, sum_stat, strand_pair, description
		track_index = data[0]
		strand_pair = data[7]
		description = data[8]

		if description.startswith('DNASE:'):
			patterns = dnase_patterns
		elif description.startswith('ATAC:'):
			patterns = atac_patterns
		else:
			continue
		description_text = description.split(':', 1)[1]

		if description_matches(description_text, patterns) == False:
			continue

		# Accessibility tracks are unstranded, so strand_pair must point at the track itself.
		# fast_borzoi_sed.py remaps strand_pair through the subset, so a dangling pair would crash it.
		if strand_pair != track_index:
			print('assumption error: track ' + track_index + ' has strand_pair ' + strand_pair)
			sys.exit(1)

		# Keep the row exactly as is: the original index column is what fast_borzoi_sed.py uses to slice model heads
		t.write(line + '\n')
		n_selected = n_selected + 1
		print(track_index + '\t' + description)
	f.close()
	t.close()
	return n_selected


#####################
# Command line args
#####################
borzoi_target_file = sys.argv[1]
output_file = sys.argv[2]


n_selected = make_peak_accessibility_targets_file(borzoi_target_file, output_file)
print(str(n_selected) + ' tracks written to ' + output_file)
