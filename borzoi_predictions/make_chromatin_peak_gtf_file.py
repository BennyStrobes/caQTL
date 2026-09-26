import numpy as np
import os
import sys
import pdb


def extract_unique_peak_ids_from_pair_file(variant_peak_pair_file):
	f = open(variant_peak_pair_file)
	peak_dicti = {}
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		peak_dicti[data[1]] = 1
	f.close()
	return peak_dicti


def make_chromatin_peak_gtf_file(peak_dicti, peak_gtf_file):
	# Peak ids look like chr21-42403636-42404390
	# Coordinates are treated as 1-based inclusive (Signac style), so they go into the GTF as is.
	# If they are 0-based BED, start should be +1; this is 1bp against a 32bp model bin, so immaterial.
	tupler = []
	for peak_id in [*peak_dicti]:
		data = peak_id.split('-')
		if len(data) != 3:
			print('assumption error: unexpected peak id ' + peak_id)
			sys.exit(1)
		chrom = data[0]
		start = int(data[1])
		end = int(data[2])
		if start >= end:
			print('assumption error: peak start >= end for ' + peak_id)
			sys.exit(1)
		tupler.append((chrom, start, end, peak_id))

	# Sort by chromosome, then position (not required by baskerville, just tidy)
	tupler.sort(key=lambda x: (x[0], x[1]))

	t = open(peak_gtf_file, 'w')
	for chrom, start, end, peak_id in tupler:
		# baskerville's Transcriptome only reads rows whose feature column is 'exon'
		# and keys them by gene_id, so one exon line per peak makes each peak its own "gene".
		attributes = 'gene_id "' + peak_id + '"; gene_name "' + peak_id + '";'
		t.write(chrom + '\t' + 'peaks' + '\t' + 'exon' + '\t' + str(start) + '\t' + str(end) + '\t.\t.\t.\t' + attributes + '\n')
	t.close()

	return len(tupler)


#####################
# Command line args
#####################
variant_peak_pair_file = sys.argv[1]
peak_gtf_file = sys.argv[2]


# Unique peaks in the pair file (so the gtf holds exactly the peaks we test)
peak_dicti = extract_unique_peak_ids_from_pair_file(variant_peak_pair_file)

# Write one exon line per peak
n_peaks = make_chromatin_peak_gtf_file(peak_dicti, peak_gtf_file)

print(str(n_peaks) + ' peaks written to ' + peak_gtf_file)
