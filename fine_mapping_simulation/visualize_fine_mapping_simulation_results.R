args = commandArgs(trailingOnly=TRUE)
library(cowplot)
library(ggplot2)
library(RColorBrewer)
options(warn=1)

# Organized results hold one line per (simulation, gene, variant) with the variant's simulated causal class
# (none / direct / mediated) and the eQTL PIP from each method. Calibration and power are computed within
# each simulation replicate and then averaged across replicates, with 95% CIs from the across-replicate
# standard error of the mean.
method_columns <- c("eqtl_only_pip", "ca_qtl_mediated_pip")
method_labels <- c("eQTL only", "caQTL-mediated")
method_colors <- c("eQTL only"="#4C78A8", "caQTL-mediated"="#D06A4B")

figure_theme <- function() {
	return(
		theme_minimal(base_size=11) +
		theme(
			plot.title=element_text(face="bold", size=12, hjust=.5),
			axis.title=element_text(size=11, color="#25313B"),
			axis.text=element_text(size=10, color="#33424F"),
			panel.grid.minor=element_blank(),
			panel.grid.major.x=element_blank(),
			panel.grid.major.y=element_blank(),
			panel.background=element_rect(fill="#FBFCFD", color=NA),
			plot.background=element_rect(fill="#FBFCFD", color=NA),
			axis.line.x=element_line(colour="#7E8A97", linewidth=.35),
			axis.line.y=element_line(colour="#7E8A97", linewidth=.35),
			legend.position="none"
		)
	)
}

# Mean and 95% CI across simulation replicates of a per-replicate statistic
summarize_across_simulations <- function(per_sim_df, value_column, group_columns) {
	summary_df <- aggregate(per_sim_df[[value_column]], by=per_sim_df[group_columns], FUN=function(x) c(mean=mean(x), se=sd(x)/sqrt(length(x)), n=length(x)))
	summary_df <- do.call(data.frame, summary_df)
	colnames(summary_df)[(ncol(summary_df)-2):ncol(summary_df)] <- c("mean", "se", "n_simulations")
	summary_df$se[is.na(summary_df$se)] <- 0
	# All summarized statistics are proportions, so the CIs are clipped to [0, 1]
	summary_df$ci_lower <- pmax(summary_df$mean - 1.96*summary_df$se, 0)
	summary_df$ci_upper <- pmin(summary_df$mean + 1.96*summary_df$se, 1)
	return(summary_df)
}

# Calibration: within each replicate, the fraction of variants in a PIP bin that are truly causal
compute_calibration_per_simulation <- function(df, pip_bin_lower, pip_bin_upper) {
	rows <- list()
	for (method_iter in seq_along(method_columns)) {
		pips <- df[[method_columns[method_iter]]]
		for (bin_iter in seq_along(pip_bin_lower)) {
			in_bin <- pips >= pip_bin_lower[bin_iter] & pips < pip_bin_upper[bin_iter]
			for (sim in unique(df$simulation_number[in_bin])) {
				sel <- in_bin & df$simulation_number == sim
				rows[[length(rows)+1]] <- data.frame(
					method=method_labels[method_iter],
					bin_label=paste0(pip_bin_lower[bin_iter], "-", min(pip_bin_upper[bin_iter], 1)),
					simulation_number=sim,
					mean_pip=mean(pips[sel]),
					fraction_causal=mean(df$causal_class[sel] != "none"),
					n_variants=sum(sel)
				)
			}
		}
	}
	return(do.call(rbind, rows))
}

# Power: within each replicate, the fraction of truly causal variants with PIP above a threshold,
# for all causal variants and stratified by causal class (direct / mediated) and by gene constraint
compute_power_per_simulation <- function(df, pip_thresholds) {
	causal_df <- df[df$causal_class != "none", ]
	causal_df$gene_class <- ifelse(causal_df$is_constrained == 1, "constrained", "non-constrained")
	rows <- list()
	for (method_iter in seq_along(method_columns)) {
		for (threshold in pip_thresholds) {
			for (sim in unique(causal_df$simulation_number)) {
				sel <- causal_df$simulation_number == sim
				for (causal_class in c("all", "direct", "mediated")) {
					for (gene_class in c("all", "constrained", "non-constrained")) {
						class_sel <- sel & (causal_class == "all" | causal_df$causal_class == causal_class) & (gene_class == "all" | causal_df$gene_class == gene_class)
						if (sum(class_sel) == 0) next
						rows[[length(rows)+1]] <- data.frame(
							method=method_labels[method_iter],
							pip_threshold=threshold,
							causal_class=causal_class,
							gene_class=gene_class,
							simulation_number=sim,
							power=mean(causal_df[[method_columns[method_iter]]][class_sel] > threshold),
							n_causal=sum(class_sel)
						)
					}
				}
			}
		}
	}
	return(do.call(rbind, rows))
}

# Calibration at thresholds: within each replicate, the fraction of SNPs with PIP above a threshold that are truly causal,
# for all genes and stratified by gene constraint
compute_calibration_at_thresholds_per_simulation <- function(df, pip_thresholds) {
	gene_class_all <- ifelse(df$is_constrained == 1, "constrained", "non-constrained")
	rows <- list()
	for (method_iter in seq_along(method_columns)) {
		pips <- df[[method_columns[method_iter]]]
		for (threshold in pip_thresholds) {
			above <- pips > threshold
			for (sim in unique(df$simulation_number[above])) {
				for (gene_class in c("all", "constrained", "non-constrained")) {
					sel <- above & df$simulation_number == sim & (gene_class == "all" | gene_class_all == gene_class)
					if (sum(sel) == 0) next
					rows[[length(rows)+1]] <- data.frame(
						method=method_labels[method_iter],
						pip_threshold=threshold,
						gene_class=gene_class,
						simulation_number=sim,
						fraction_causal=mean(df$causal_class[sel] != "none"),
						mean_pip=mean(pips[sel]),
						n_variants=sum(sel)
					)
				}
			}
		}
	}
	return(do.call(rbind, rows))
}

make_calibration_plot <- function(calibration_df, plot_title) {
	calibration_df$method <- factor(calibration_df$method, levels=method_labels)
	return(
		ggplot(calibration_df, aes(x=mean_pip, y=fraction_causal, color=method)) +
			geom_abline(intercept=0, slope=1, linetype="dashed", color="#7E8A97", linewidth=.5) +
			geom_errorbar(aes(ymin=ci_lower, ymax=ci_upper), width=.02, linewidth=.45) +
			geom_point(size=2) +
			scale_color_manual(values=method_colors, name="") +
			coord_cartesian(xlim=c(0, 1)) +
			xlab("Mean PIP in bin") +
			ylab("Fraction of variants that are causal") +
			ggtitle(plot_title) +
			figure_theme() +
			theme(legend.position="right")
	)
}

make_power_plot <- function(power_df, plot_title) {
	power_df$method <- factor(power_df$method, levels=method_labels)
	power_df$pip_threshold <- factor(power_df$pip_threshold, levels=sort(unique(power_df$pip_threshold)))
	dodge_width <- 0.35
	return(
		ggplot(power_df, aes(x=pip_threshold, y=power, color=method, group=method)) +
			geom_errorbar(aes(ymin=ci_lower, ymax=ci_upper), width=.14, linewidth=.45, position=position_dodge(width=dodge_width)) +
			geom_line(linewidth=.7, position=position_dodge(width=dodge_width)) +
			geom_point(size=2, position=position_dodge(width=dodge_width)) +
			scale_color_manual(values=method_colors, name="") +
			xlab("PIP threshold") +
			ylab("Power (fraction of causal variants above threshold)") +
			ggtitle(plot_title) +
			figure_theme() +
			theme(legend.position="right")
	)
}

# Barplot of a per-threshold statistic (mean with 95% CI) for both methods, one bar group per PIP threshold
# expected_column (optional): a per-bar expected value drawn as a dashed line across the bar
make_threshold_barplot <- function(summary_df, y_column, y_label, plot_title, expected_column=NULL) {
	summary_df$method <- factor(summary_df$method, levels=method_labels)
	summary_df$pip_threshold <- factor(summary_df$pip_threshold, levels=sort(unique(summary_df$pip_threshold)), labels=paste0("> ", sort(unique(summary_df$pip_threshold))))
	dodge_width <- 0.75
	plot_obj <- ggplot(summary_df, aes(x=pip_threshold, y=.data[[y_column]], fill=method)) +
			geom_col(color="#22313B", width=.68, linewidth=.35, position=position_dodge(width=dodge_width)) +
			geom_errorbar(aes(ymin=ci_lower, ymax=ci_upper), width=.16, linewidth=.45, color="#22313B", position=position_dodge(width=dodge_width))
	if (!is.null(expected_column)) {
		# Zero-height errorbar spanning the bar width: a dashed line at the expected value of each bar
		plot_obj <- plot_obj + geom_errorbar(aes(ymin=.data[[expected_column]], ymax=.data[[expected_column]]), width=.34, linewidth=.8, linetype="dashed", color="#22313B", position=position_dodge(width=dodge_width))
	}
	return(
		plot_obj +
			scale_fill_manual(values=method_colors, name="") +
			xlab("PIP threshold") +
			ylab(y_label) +
			ggtitle(plot_title) +
			figure_theme() +
			theme(legend.position="right")
	)
}

make_faceted_threshold_barplot <- function(summary_df, y_column, y_label, facet_column, facet_labels, plot_title, expected_column=NULL) {
	summary_df$facet_label <- factor(facet_labels[summary_df[[facet_column]]], levels=facet_labels)
	return(
		make_threshold_barplot(summary_df, y_column, y_label, plot_title, expected_column) +
			facet_wrap(~facet_label) +
			theme(strip.text=element_text(size=10, color="#25313B"))
	)
}

# Power plot faceted by a stratification column (facet_labels: named vector mapping column values to panel titles, in display order)
make_faceted_power_plot <- function(power_df, facet_column, facet_labels, plot_title) {
	power_df$facet_label <- factor(facet_labels[power_df[[facet_column]]], levels=facet_labels)
	return(
		make_power_plot(power_df, plot_title) +
			facet_wrap(~facet_label) +
			theme(strip.text=element_text(size=10, color="#25313B"))
	)
}


#################
# Command line args
#################
organized_results_file <- args[1]
viz_dir <- args[2]

pip_bin_lower <- c(0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
pip_bin_upper <- c(0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)
pip_thresholds <- c(0.1, 0.3, 0.5, 0.7, 0.9, 0.95)
barplot_pip_thresholds <- c(0.5, 0.7, 0.9, 0.95)


#################
# Load in organized results
#################
df <- read.table(organized_results_file, header=TRUE, sep="\t", colClasses=c("integer", "character", "integer", "integer", "character", "numeric", "numeric"))
n_simulations <- length(unique(df$simulation_number))
print(paste0("Loaded ", nrow(df), " variant-gene pairs from ", n_simulations, " simulations"))


#################
# Calibration plot
#################
calibration_per_sim_df <- compute_calibration_per_simulation(df, pip_bin_lower, pip_bin_upper)
calibration_df <- summarize_across_simulations(calibration_per_sim_df, "fraction_causal", c("method", "bin_label"))
calibration_df$fraction_causal <- calibration_df$mean
calibration_df$mean_pip <- summarize_across_simulations(calibration_per_sim_df, "mean_pip", c("method", "bin_label"))$mean
calibration_plot <- make_calibration_plot(calibration_df, paste0("eQTL PIP calibration (", n_simulations, " simulations)"))
calibration_plot_file <- file.path(viz_dir, "fine_mapping_simulation_calibration_plot.pdf")
ggsave(filename=calibration_plot_file, plot=calibration_plot, width=5.2, height=3.8)


#################
# Power plots
#################
power_per_sim_df <- compute_power_per_simulation(df, pip_thresholds)
power_df <- summarize_across_simulations(power_per_sim_df, "power", c("method", "pip_threshold", "causal_class", "gene_class"))
power_df$power <- power_df$mean

# All causal variants, all genes
power_plot <- make_power_plot(power_df[power_df$causal_class == "all" & power_df$gene_class == "all", ], paste0("eQTL fine-mapping power (", n_simulations, " simulations)"))
power_plot_file <- file.path(viz_dir, "fine_mapping_simulation_power_plot.pdf")
ggsave(filename=power_plot_file, plot=power_plot, width=5.2, height=3.8)

# Stratified by causal variant class (direct / mediated)
power_by_class_plot <- make_faceted_power_plot(power_df[power_df$causal_class != "all" & power_df$gene_class == "all", ], "causal_class", c(direct="Direct causal variants", mediated="Mediated causal variants"), paste0("eQTL fine-mapping power by causal variant class (", n_simulations, " simulations)"))
power_by_class_plot_file <- file.path(viz_dir, "fine_mapping_simulation_power_by_causal_class_plot.pdf")
ggsave(filename=power_by_class_plot_file, plot=power_by_class_plot, width=8.0, height=3.8)

# Stratified by gene constraint
power_by_constraint_plot <- make_faceted_power_plot(power_df[power_df$causal_class == "all" & power_df$gene_class != "all", ], "gene_class", c("non-constrained"="Non-constrained genes", "constrained"="Constrained genes"), paste0("eQTL fine-mapping power by gene constraint (", n_simulations, " simulations)"))
power_by_constraint_plot_file <- file.path(viz_dir, "fine_mapping_simulation_power_by_gene_constraint_plot.pdf")
ggsave(filename=power_by_constraint_plot_file, plot=power_by_constraint_plot, width=8.0, height=3.8)


#################
# Barplots at PIP thresholds: calibration (fraction of SNPs above threshold that are causal) and power
#################
calibration_threshold_per_sim_df <- compute_calibration_at_thresholds_per_simulation(df, barplot_pip_thresholds)
calibration_threshold_df <- summarize_across_simulations(calibration_threshold_per_sim_df, "fraction_causal", c("method", "pip_threshold", "gene_class"))
calibration_threshold_df$fraction_causal <- calibration_threshold_df$mean
# Expected fraction causal under calibration: the mean PIP of the SNPs above the threshold (dashed line on each bar)
calibration_threshold_df$expected_fraction_causal <- summarize_across_simulations(calibration_threshold_per_sim_df, "mean_pip", c("method", "pip_threshold", "gene_class"))$mean
calibration_y_label <- "Fraction of SNPs above threshold that are causal"

calibration_barplot <- make_threshold_barplot(calibration_threshold_df[calibration_threshold_df$gene_class == "all", ], "fraction_causal", calibration_y_label, paste0("Calibration at PIP thresholds (", n_simulations, " simulations)"), expected_column="expected_fraction_causal")
ggsave(filename=file.path(viz_dir, "fine_mapping_simulation_calibration_threshold_barplot.pdf"), plot=calibration_barplot, width=6.0, height=3.8)

calibration_barplot_by_constraint <- make_faceted_threshold_barplot(calibration_threshold_df[calibration_threshold_df$gene_class != "all", ], "fraction_causal", calibration_y_label, "gene_class", c("non-constrained"="Non-constrained genes", "constrained"="Constrained genes"), paste0("Calibration at PIP thresholds by gene constraint (", n_simulations, " simulations)"), expected_column="expected_fraction_causal")
ggsave(filename=file.path(viz_dir, "fine_mapping_simulation_calibration_threshold_barplot_by_gene_constraint.pdf"), plot=calibration_barplot_by_constraint, width=8.0, height=3.8)

power_threshold_df <- power_df[power_df$causal_class == "all" & power_df$pip_threshold %in% barplot_pip_thresholds, ]
power_y_label <- "Power (fraction of causal variants above threshold)"

power_barplot <- make_threshold_barplot(power_threshold_df[power_threshold_df$gene_class == "all", ], "power", power_y_label, paste0("Power at PIP thresholds (", n_simulations, " simulations)"))
ggsave(filename=file.path(viz_dir, "fine_mapping_simulation_power_threshold_barplot.pdf"), plot=power_barplot, width=6.0, height=3.8)

power_barplot_by_constraint <- make_faceted_threshold_barplot(power_threshold_df[power_threshold_df$gene_class != "all", ], "power", power_y_label, "gene_class", c("non-constrained"="Non-constrained genes", "constrained"="Constrained genes"), paste0("Power at PIP thresholds by gene constraint (", n_simulations, " simulations)"))
ggsave(filename=file.path(viz_dir, "fine_mapping_simulation_power_threshold_barplot_by_gene_constraint.pdf"), plot=power_barplot_by_constraint, width=8.0, height=3.8)
