library(hcocena)
library(org.Hs.eg.db)
library(AnnotationDbi)
library(readr)
library(dplyr)
library(ggplot2)

# Adapted from:
# https://github.com/MarieOestreich/PRO-GENE-GEN/blob/main/eval/bio_eval/hcocena_main.Rmd
# Towards Biologically Plausible and Private Gene Expression Data Generation
# Chen, Oestreich, & Afonja et al. 2024

args <- commandArgs(trailingOnly <- TRUE)
home_dir <- args[1]
split_no <- as.integer(args[2])
dataset_name <- args[3]
generator_name <- args[4]
param_dir <- args[5]
hcocena_dir <- args[6]
cutoff <- as.double(args[7])
### this is co-expression cutoff

bio_res_dir <- file.path(home_dir, "results/bio", dataset_name, generator_name)

real_data_dir <- file.path(home_dir, "data_splits", dataset_name, "real")
synthetic_data_dir <- file.path(
  home_dir, "data_splits", dataset_name,
  "synthetic", generator_name, param_dir
)

if (!dir.exists(bio_res_dir)) {
  dir.create(bio_res_dir, recursive = TRUE)
}

# Initialize hCoCena Object
init_object()

# initialize working directory
init_wd(
  dir_count_data = FALSE,
  dir_annotation = FALSE,
  dir_reference_files = file.path(hcocena_dir, "reference_files/"),
  dir_output = bio_res_dir
)

check_dirs()
init_save_folder(name = param_dir)

load_data <- function(data_dir,
                      file_prefix,
                      split_no,
                      scaling_params = NULL) {
  data <- read.csv(file.path(
    data_dir,
    paste0(file_prefix, split_no, ".csv")
  ))
  rownames(data) <- paste0("P", seq_len(nrow(data)))
  # Transpose data to switch gene names to rows
  data <- t(data)
  # reverse standardization if scaling parameters are provided
  if (!is.null(scaling_params) && all(rownames(data) == rownames(scaling_params))) {
    means <- scaling_params$mean
    stds <- scaling_params$std
    data <- sweep(data, 1, stds, `*`)
    data <- sweep(data, 1, means, `+`)
  }
  data <- as.data.frame(data)
  ## retrieve gene symbols...
  gene_alias <- mapIds(org.Hs.eg.db,
    keys = as.character(rownames(data)),
    column = "SYMBOL",
    keytype = "ENSEMBL",
    multiVals = "first"
  )
  gene_alias <- as.vector(gene_alias)
  if (anyNA(rownames(gene_alias))) {
    print("NA in gene names")
  } else {
    rownames(data) <- gene_alias
  }
  return(data)
}

# Function to load annotations
load_annotations <- function(data_dir, file_prefix, split_no) {
  annots <- read.csv(file.path(
    data_dir,
    paste(file_prefix, split_no, ".csv", sep = "")
  ))
  rownames(annots) <- paste0("P", seq_len(nrow(annots)))
  colnames(annots) <- gsub("Subtype_Selected", "Subtype", colnames(annots))
  colnames(annots) <- gsub("cancer_type", "Subtype", colnames(annots))
  annots <- as.data.frame(annots)
  annots$SampleID <- rownames(annots)
  return(annots)
}

# scaling_params <- read.csv(file.path(real_data_dir,
#  paste("X_train_scale_params_split_", split_no, ".csv", sep = "")), row.names = 1)

real_data <- load_data(real_data_dir, "X_train_real_split_", split_no)
synthetic_data <- load_data(synthetic_data_dir, "synthetic_data_split_", split_no)

real_annots <- load_annotations(real_data_dir, "y_train_real_split_", split_no)
synthetic_annots <- load_annotations(synthetic_data_dir, "synthetic_labels_split_", split_no)
synthetic_annots <- synthetic_annots %>% mutate(Subtype = paste0(Subtype, "_syn"))

# Define hococena layers
define_layers(list(
  synthetic = c("synthetic_data", "synthetic_annots"),
  real = c("real_data", "real_annots")
))

# Set supplementary files - defined above under reference_files
set_supp_files(
  Tf = "TFcat.txt",
  Go = "c5.go.bp.v2023.1.Hs.symbols.gmt",
  Hallmark = "h.all.v2023.1.Hs.symbols.gmt",
  Kegg = "c2.cp.kegg.v2023.1.Hs.symbols.gmt"
)
# Read supplementary data
read_supplementary()

# import data without annotation files
read_data(
  sep_counts = "\t",
  sep_anno = "\t",
  gene_symbol_col = NULL,
  sample_col = NULL,
  count_has_rn = TRUE,
  anno_has_rn = TRUE
)

# Set global settings
set_global_settings(
  organism = "human",
  control_keyword = "none",
  variable_of_interest = "Subtype",
  min_nodes_number_for_network = 30,
  min_nodes_number_for_cluster = 15,
  range_GFC = 2.0,
  layout_algorithm = "layout_with_fr",
  data_in_log = FALSE
)

# Set layer-specific settings
set_layer_settings(
  top_var = c("all", "all"),
  min_corr = rep(cutoff, length(hcobject[["layers"]])),
  range_cutoff_length = rep(100, length(hcobject[["layers"]])),
  # NOTE: changed to 1000 from 100 on Nov 26, 2025
  print_distribution_plots = rep(FALSE, length(hcobject[["layers"]]))
)

# Run initial expression analysis
run_expression_analysis_1(corr_method = "pearson")

# Plot cut-offs and set cut-off value
# plot_cutoffs(interactive = TRUE)
set_cutoff(cutoff_vector = c(cutoff, cutoff))

# Plot degree distribution
# plot_deg_dist()

# run final expression analysis
run_expression_analysis_2()

# Integrate the networks
build_integrated_network(mode = "u", multi_edges = "min")

# Cluster the integrated network and plot heatmap
cluster_calculation(
  cluster_algo = "cluster_leiden",
  no_of_iterations = 2,
  resolution = 0.1,
  partition_type = "ModularityVertexPartition",
  max_cluster_count_per_gene = 1
)

plot_cluster_heatmap(
  cluster_rows = FALSE,
  file_name = paste0("Heatmap_modules", str(split_no), ".pdf")
)

# Plot the integrated network colored by clusters
plot_integrated_network()

# Set edges for real data
edges_real <- hcobject[["layer_specific_outputs"]][[
  paste0("set", length(hcobject$layers))
]][["part2"]][["heatmap_out"]][["filt_cutoff_data"]]
edges_real <- apply(edges_real, 1, function(x) {
  if (x[1] > x[2]) {
    paste0(x[1], "_", x[2])
  } else {
    paste0(x[2], "_", x[1])
  }
})

coex_rec <- data.frame()
for (i in 1:length(hcobject$layers)) {
  edges <- hcobject[["layer_specific_outputs"]][[
    paste0("set", i)
  ]][["part2"]][["heatmap_out"]][["filt_cutoff_data"]]
  edges <- apply(edges, 1, function(x) {
    if (x[1] > x[2]) {
      paste0(x[1], "_", x[2])
    } else {
      paste0(x[2], "_", x[1])
    }
  })
  intersec <- length(intersect(edges, edges_real))
  false_rec <- length(edges) - length(intersect(edges, edges_real))
  coex_rec <- rbind(coex_rec, data.frame(
    set = hcobject$layers_names[i],
    correctly_rec = intersec,
    falsely_rec = false_rec,
    score = intersec / false_rec * false_rec / length(edges_real)
  ))
}

print(length(edges_real))

coex_rec <- coex_rec[order(coex_rec$correctly_rec, decreasing = T), ]
coex_rec$set <- factor(coex_rec$set, levels = rev(coex_rec$set))
reference <- dplyr::filter(coex_rec, set == "real")$correctly_rec

coex_rec_summary <- data.frame(
  rec = c(
    mean(dplyr::filter(coex_rec, !set == "real")$correctly_rec),
    mean(dplyr::filter(coex_rec, !set == "real")$falsely_rec)
  ),
  sd = c(
    sd(dplyr::filter(coex_rec, !set == "real")$correctly_rec),
    sd(dplyr::filter(coex_rec, !set == "real")$falsely_rec)
  ),
  dir = c("correct", "false"),
  set = "synthetic"
  # edges_real = length(edges_real) # NOTE: added this Nov 26, 2025
)
coex_rec_summary$set <- factor(coex_rec_summary$set, levels = c("synthetic", "real"))

output_dir <- file.path(bio_res_dir, param_dir)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}


## save coexpression file
cutoff_str <- sprintf("%.1f", cutoff)
output_file <- file.path(
  output_dir,
  paste0("coexpr_cutoff=", cutoff_str, "_split_", split_no, ".csv")
)
write.csv(coex_rec_summary, paste0(output_file))


p <- ggplot(coex_rec_summary, aes(x = set, y = rec, fill = dir)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_errorbar(aes(ymin = rec - sd, ymax = rec + sd),
    width = .2,
    position = position_dodge(.9)
  ) +
  scale_fill_manual(values = c("#50c1de", "#ba3061")) +
  coord_flip() +
  theme_bw() +
  geom_hline(
    yintercept = reference,
    linewidth = .5, linetype = "dashed"
  ) +
  theme( # remove the vertical grid lines
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.title = element_blank()
  ) +
  ylab(paste0("coexpressions with r > ", hcobject$cutoff_vec[1])) +
  xlab("") +
  ggtitle("Co-Expression Preservation")



# ggsave(
#  file = file.path(
#    output_dir,
#    paste0("coexpr-cutoff=", cutoff, "_split_", split_no, ".png")
#  ),
#  bg = "transparent", plot = p, width = 10, height = 6, dpi = 300
# )

# Perform functional enrichment analysis
# functional_enrichment(
#  gene_sets = c("Hallmark", "Go", "Kegg"),
#  clusters = "all",
#  top = 5,
#  padj = "BH",
# qval = 0.1
# )

# Transcription factor enrichment analysis
# TF_overrep_module(clusters = "all", topTF = 5, topTarget = 5)
