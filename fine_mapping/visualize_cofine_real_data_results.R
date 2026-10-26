args = commandArgs(trailingOnly=TRUE)
library(cowplot)
library(ggplot2)
options(warn=1)

# Plots comparing the real-data CoFine fit with CoFine without caQTL and peak-gene link data, from organize_cofine_real_data_results.py
#   args[1]: organized results stem (the --output_stem of the organizer); args[2]: output directory for the pdfs; args[3]: cell type
# A variant is "on a chromatin path" if CoFine's P(H = 1) > 0.5. Variant-level plots use the variants with PIP >= 0.1 in either fit.
organized_stem <- args[1]
viz_dir <- args[2]
cell_type <- args[3]
dir.create(viz_dir, showWarnings=FALSE, recursive=TRUE)

method_colors <- c("CoFine"="#D06A4B", "CoFine without chromatin"="#4C78A8")
call_colors <- c("Both fits"="#7E8A97", "CoFine only"="#D06A4B", "Without chromatin only"="#4C78A8")

figure_theme <- function() {
	return(
		theme_minimal(base_size=11) +
		theme(
			plot.title=element_text(face="bold", size=12, hjust=.5),
			axis.title=element_text(size=11, color="#25313B"),
			axis.text=element_text(size=10, color="#33424F"),
			strip.text=element_text(size=10, face="bold"),
			panel.grid.minor=element_blank(),
			panel.grid.major.x=element_blank(),
			axis.line.x=element_line(colour="#7E8A97", linewidth=.35),
			axis.line.y=element_line(colour="#7E8A97", linewidth=.35),
			legend.position="bottom",
			legend.title=element_blank()
		)
	)
}

gene_df <- read.table(paste0(organized_stem, "_genes.txt"), header=TRUE, sep="\t", stringsAsFactors=FALSE, check.names=FALSE)
variant_df <- read.table(paste0(organized_stem, "_variants.txt"), header=TRUE, sep="\t", stringsAsFactors=FALSE)
hyperparameter_df <- read.table(paste0(organized_stem, "_hyperparameter_samples.txt"), header=TRUE, sep="\t", stringsAsFactors=FALSE)
variant_df$on_path <- ifelse(variant_df$path_prob > .5, "On a chromatin path", "Off a chromatin path")
variant_df$abs_z <- abs(variant_df$eqtl_z)
n_genes <- nrow(gene_df)


#####################
# Number of eQTLs: totals and per gene
#####################
totals_df <- data.frame(
	method=rep(names(method_colors), 3),
	metric=rep(c("Expected causal eQTLs\n(sum of PIPs)", "PIP > 0.5", "PIP > 0.9"), each=2),
	value=c(sum(gene_df$sum_pip_cofine), sum(gene_df$sum_pip_no_chromatin), sum(gene_df[["n_pip_0.5_cofine"]]), sum(gene_df[["n_pip_0.5_no_chromatin"]]), sum(gene_df[["n_pip_0.9_cofine"]]), sum(gene_df[["n_pip_0.9_no_chromatin"]]))
)
totals_df$metric <- factor(totals_df$metric, levels=unique(totals_df$metric))
totals_plot <- ggplot(totals_df, aes(x=metric, y=value, fill=method)) +
	geom_col(position=position_dodge(width=.8), width=.75) +
	geom_text(aes(label=round(value)), position=position_dodge(width=.8), vjust=-.4, size=3) +
	scale_fill_manual(values=method_colors) +
	labs(x="", y="Number of eQTLs", title=paste0(cell_type, ": eQTLs across ", n_genes, " genes")) +
	figure_theme()

max_sum <- max(c(gene_df$sum_pip_cofine, gene_df$sum_pip_no_chromatin))
gene_sum_plot <- ggplot(gene_df, aes(x=sum_pip_no_chromatin, y=sum_pip_cofine)) +
	geom_abline(slope=1, intercept=0, linetype="dashed", color="grey50") +
	geom_point(size=1, alpha=.5, color="#25313B") +
	coord_cartesian(xlim=c(0, max_sum), ylim=c(0, max_sum)) +
	labs(x="Sum of PIPs, CoFine without chromatin", y="Sum of PIPs, CoFine", title="Expected causal eQTLs per gene") +
	figure_theme()


#####################
# Calls of the two fits: shared or specific, on or off a chromatin path
#####################
call_rows <- list()
for (threshold in c(.5, .9)) {
	called <- variant_df[variant_df$pip_cofine > threshold | variant_df$pip_no_chromatin > threshold, ]
	called$call <- ifelse(called$pip_cofine > threshold & called$pip_no_chromatin > threshold, "Both fits", ifelse(called$pip_cofine > threshold, "CoFine only", "Without chromatin only"))
	called$threshold_label <- paste0("PIP > ", threshold)
	call_rows[[length(call_rows) + 1]] <- called
}
call_df <- do.call(rbind, call_rows)
call_df$call <- factor(call_df$call, levels=names(call_colors))
call_plot <- ggplot(call_df, aes(x=on_path, fill=call)) +
	geom_bar(width=.7) +
	facet_wrap(~threshold_label, scales="free_y") +
	scale_fill_manual(values=call_colors) +
	labs(x="CoFine P(H = 1) > 0.5", y="Number of variants called", title="Variants called by each fit") +
	figure_theme()

pip_scatter_plot <- ggplot(variant_df[order(variant_df$path_prob), ], aes(x=pip_no_chromatin, y=pip_cofine, color=path_prob)) +
	geom_abline(slope=1, intercept=0, linetype="dashed", color="grey50") +
	geom_point(size=1, alpha=.7) +
	scale_color_gradient(low="#C9D3DC", high="#D06A4B", limits=c(0, 1), name="CoFine P(H = 1)") +
	coord_cartesian(xlim=c(0, 1), ylim=c(0, 1)) +
	labs(x="PIP, CoFine without chromatin", y="PIP, CoFine", title="Variant PIPs (PIP >= 0.1 in either fit)") +
	figure_theme() +
	theme(legend.title=element_text(size=9), legend.key.width=unit(1, "cm"))


#####################
# eQTL evidence behind the calls at PIP > 0.5
#####################
evidence_df <- call_df[call_df$threshold_label == "PIP > 0.5", ]
evidence_df$group <- ifelse(evidence_df$call == "CoFine only", paste0("CoFine only,\n", tolower(evidence_df$on_path)), as.character(evidence_df$call))
group_levels <- c("Both fits", "CoFine only,\non a chromatin path", "CoFine only,\noff a chromatin path", "Without chromatin only")
evidence_df$group <- factor(evidence_df$group, levels=group_levels)
evidence_summary <- aggregate(abs_z ~ group, data=evidence_df, FUN=function(x) c(n=length(x), weak=mean(x < 1.96)))
evidence_summary <- do.call(data.frame, evidence_summary)
evidence_summary$label <- paste0("n = ", evidence_summary$abs_z.n, "\n", round(100 * evidence_summary$abs_z.weak), "% |z| < 1.96")
evidence_plot <- ggplot(evidence_df, aes(x=group, y=abs_z)) +
	geom_hline(yintercept=1.96, linetype="dashed", color="grey50") +
	geom_boxplot(outlier.shape=NA, fill="#F3D9CF", color="#D06A4B", width=.6) +
	geom_jitter(width=.12, height=0, size=.6, alpha=.4, color="#25313B") +
	geom_text(data=evidence_summary, aes(x=group, y=Inf, label=label), vjust=1.3, size=3, color="#33424F") +
	scale_y_sqrt() +
	labs(x="", y="Marginal eQTL |z| (square-root scale)", title="eQTL evidence for variants with PIP > 0.5") +
	figure_theme()


#####################
# CoFine hyper-parameter traces (retained sweeps)
#####################
hyperparameter_df$hyperparameter <- factor(hyperparameter_df$hyperparameter, levels=c("pi0", "pi1", "pi_delta", "pi_gamma", "lambda", "tau2", "sigma2_0", "sigma2_delta", "sigma2_gamma"))
trace_plot <- ggplot(hyperparameter_df, aes(x=sweep, y=value)) +
	geom_line(linewidth=.3, color="#D06A4B") +
	facet_wrap(~hyperparameter, scales="free_y", ncol=3) +
	labs(x="Retained sweep", y="", title="CoFine hyper-parameters after burn-in") +
	figure_theme() +
	theme(panel.grid.major.x=element_line(colour="grey92"))


ggsave(paste0(viz_dir, cell_type, "_cofine_eqtl_counts.pdf"), plot_grid(totals_plot, gene_sum_plot, ncol=2), width=10, height=4.5, units="in")
ggsave(paste0(viz_dir, cell_type, "_cofine_calls.pdf"), plot_grid(call_plot, pip_scatter_plot, ncol=2), width=11, height=4.8, units="in")
ggsave(paste0(viz_dir, cell_type, "_cofine_call_evidence.pdf"), evidence_plot, width=8, height=4.5, units="in")
ggsave(paste0(viz_dir, cell_type, "_cofine_hyperparameter_traces.pdf"), trace_plot, width=9, height=6.5, units="in")
combined <- plot_grid(plot_grid(totals_plot, gene_sum_plot, ncol=2), plot_grid(call_plot, pip_scatter_plot, ncol=2), evidence_plot, trace_plot, ncol=1, rel_heights=c(1, 1.05, 1, 1.4))
ggsave(paste0(viz_dir, cell_type, "_cofine_real_data_summary.pdf"), combined, width=11, height=20, units="in")
print(paste0("Plots of ", n_genes, " genes written to ", viz_dir))
