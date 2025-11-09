library(scran)
library(dplyr)
library(stringr)
library(ggplot2)

# Adapted from:
# https://github.com/MarieOestreich/PRO-GENE-GEN/blob/main/eval/bio_eval/hcocena_main.Rmd
# Towards Biologically Plausible and Private Gene Expression Data Generation
# Chen, Oestreich, & Afonja et al. 2024

args <- commandArgs(trailingOnly = TRUE)
home_dir <- args[1]
split_no <- as.integer(args[2])
dataset_name <- args[3]
generator_name <- args[4]
param_dir <- args[5]
p_value_th <- as.double(args[6])
lfc_th <- as.double(args[7])

real_data_dir <- file.path(home_dir, "data_splits", dataset_name, "real")
synthetic_data_dir <- file.path(
  home_dir, "data_splits", dataset_name,
  "synthetic", generator_name, param_dir
)
meta_dir <- file.path(home_dir, "data/meta")
bio_res_dir <- file.path(home_dir, "results/bio", dataset_name, generator_name)
anno_colname_class <- "Subtype"


# Load synthetic and real data
real_data <- read.csv(file.path(
  real_data_dir,
  paste("X_train_real_split_", split_no, ".csv", sep = "")
))
rownames(real_data) <- paste0("P", seq_len(nrow(real_data)))

real_annots <- read.csv(file.path(
  real_data_dir,
  paste("y_train_real_split_", split_no, ".csv", sep = "")
))
rownames(real_annots) <- paste0("P", seq_len(nrow(real_annots)))
colnames(real_annots) <- gsub(
  "Subtype_Selected",
  "Subtype", colnames(real_annots)
)
colnames(real_annots) <- gsub(
  "cancer_type",
  "Subtype", colnames(real_annots)
)


synthetic_data <- read.csv(file.path(
  synthetic_data_dir,
  paste("synthetic_data_split_", split_no, ".csv", sep = "")
))
rownames(synthetic_data) <- paste0("P", seq_len(nrow(synthetic_data)))

synthetic_annots <- read.csv(file.path(
  synthetic_data_dir,
  paste("synthetic_labels_split_", split_no, ".csv", sep = "")
))
### this was missing...
colnames(real_annots) <- gsub(
  "Subtype_Selected",
  "Subtype", colnames(real_annots)
)
colnames(synthetic_annots) <- gsub(
  "cancer_type",
  "Subtype", colnames(real_annots)
)
rownames(synthetic_annots) <- paste0("P", seq_len(nrow(synthetic_annots)))

# Load means and standard deviations for reverse standardization
# scaling_params <- read.csv(file.path(real_data_dir,
#    paste("X_train_scale_params_split_", split_no, ".csv", sep = "")), row.names =1)


# transpose to switch gene names to rows
synthetic_data <- t(synthetic_data)
synthetic_data <- as.data.frame(synthetic_data)

real_data <- t(real_data)
real_data <- as.data.frame(real_data)

# reverse standardization to get original counts
# means <- scaling_params$mean
# stds <- scaling_params$std
# real_data <- sweep(real_data, 1, stds, `*`)
# real_data <- sweep(real_data, 1, means, `+`)

# synthetic_data <- sweep(synthetic_data, 1, stds, `*`)
# synthetic_data <- sweep(synthetic_data, 1, means, `+`)

# Step 0: Use pairwise Wilcoxon test for DE
DE_genes <- list()
datasets <- list(real_data, synthetic_data)
annotations <- list(real_annots, synthetic_annots)
dataset_names <- c("real_data", "synthetic_data")
print(dim(real_data))

# head(real_annots)
# colnames(real_annots)
# rownames(real_annots)

# head(synthetic_annots)
# colnames(synthetic_annots)
# rownames(synthetic_annots)

for (i in 1:length(datasets)) {
  current_counts <- datasets[[i]]
  current_counts[is.na(current_counts)] <- 0
  current_groups <- annotations[[i]][[anno_colname_class]]

  # TODO: might add this to snakemake with lfc=0 and 0.5
  up <- scran::pairwiseWilcox(current_counts,
    groups = current_groups,
    direction = "up",
    lfc = lfc_th
  )

  down <- scran::pairwiseWilcox(current_counts,
    groups = current_groups,
    direction = "down",
    lfc = lfc_th
  )

  # Process the pairs and filter significant genes
  pairs <- paste0(up$pairs$first, "_", up$pairs$second)

  res <- list()
  for (j in 1:length(pairs)) {
    signif_up <- rownames(dplyr::filter(
      as.data.frame(up[["statistics"]][[j]]), p.value < p_value_th
    ))
    signif_down <- rownames(dplyr::filter(
      as.data.frame(down[["statistics"]][[j]]), p.value < p_value_th
    ))
    res[[pairs[j]]][["up"]] <- signif_up
    res[[pairs[j]]][["down"]] <- signif_down
  }

  DE_genes[[dataset_names[i]]] <- res
}


# Step 1: Create Unique Comparisons for "real" and "synthetic"
unique_comparisons <- c()
for (comp in names(DE_genes$real_data)) {
  spl <- strsplit(comp, split = "_") %>% unlist()

  # Ensure lexicographical order for comparisons
  if (spl[1] < spl[2]) {
    unique_comparisons <- c(unique_comparisons, paste0(spl[1], "_", spl[2]))
  } else {
    unique_comparisons <- c(unique_comparisons, paste0(spl[2], "_", spl[1]))
  }
}
unique_comparisons <- unique(unique_comparisons)

print("Step 1 completed.")

# Step 2: Calculate TPR and FPR for each comparison
DE_correct <- list()

# Iterate over all datasets
for (i in 1:length(dataset_names)) {
  set1 <- dataset_names[i]
  set2 <- "real_data" # Assuming "real_data" is your baseline comparison
  comp <- paste0(set1, "_vs_", set2)

  DE_correct[[comp]] <- list()
  DE_correct[[comp]][["up"]] <- c()
  DE_correct[[comp]][["down"]] <- c()

  # Iterate over each unique comparison
  for (j in unique_comparisons) {
    # Upregulated genes
    inter_up <- length(intersect(
      DE_genes[[set1]][[j]][["up"]],
      DE_genes[[set2]][[j]][["up"]]
    )) /
      length(DE_genes[[set1]][[j]][["up"]])
    inter_down <- length(intersect(
      DE_genes[[set1]][[j]][["down"]],
      DE_genes[[set2]][[j]][["down"]]
    )) /
      length(DE_genes[[set1]][[j]][["down"]])

    DE_correct[[comp]][["up"]] <- append(
      DE_correct[[comp]][["up"]],
      c(inter_up)
    )
    DE_correct[[comp]][["down"]] <- append(
      DE_correct[[comp]][["down"]],
      c(inter_down)
    )
  }
}

print("Step 2 completed.")

# Step 3: Calculate TPR and FPR for each comparison
DE_TP <- list()
DE_FP <- list()

for (i in 1:length(dataset_names)) {
  set1 <- dataset_names[i]
  set2 <- "real_data"
  comp <- paste0(set1, "_vs_", set2)

  DE_TP[[comp]] <- list()
  DE_TP[[comp]][["up"]] <- c()
  DE_TP[[comp]][["down"]] <- c()

  DE_FP[[comp]] <- list()
  DE_FP[[comp]][["up"]] <- c()
  DE_FP[[comp]][["down"]] <- c()

  # Iterate over each unique comparison
  for (j in unique_comparisons) {
    # Upregulated genes

    # 'set1' = synthetic, 'set2' = real (the ground truth)
    # ------------------------------------------------------

    # 1. True "positive" genes (real upregulated)
    P <- DE_genes[[set2]][[j]][["up"]]
    # All genes real says are up

    # 2. True "negative" genes (real NOT upregulated)
    N <- base::setdiff(rownames(real_data), P)
    # Everything else that real says is *not* up

    # 3. Predicted positives (synthetic upregulated)
    pred_up <- DE_genes[[set1]][[j]][["up"]]
    # Synthetic’s guesses.

    # 4. True Positives (TP): real up + predicted up
    TP <- intersect(pred_up, P)

    # 5. False Positives (FP): synthetic said up, real said not up
    FP <- base::setdiff(pred_up, P)

    # 6. False Negatives (FN): real up, synthetic missed them
    FN <- base::setdiff(P, pred_up)

    # 7. True Negatives (TN): not up in real, not up in synthetic
    TN <- base::setdiff(N, pred_up)

    # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
    TPR_up <- length(TP) / (length(TP) + length(FN))
    FPR_up <- length(FP) / (length(FP) + length(TN))


    # Downregulated genes (similarly)
    # 1. True "positive" genes (real downregulated)
    P <- DE_genes[[set2]][[j]][["down"]]
    # All genes real says are down.

    # 2. True "negative" genes (real NOT dowregulated)
    N <- base::setdiff(rownames(real_data), P)
    # Everything else that real says is *not* down

    # 3. Predicted positives (synthetic downregulated)
    pred_down <- DE_genes[[set1]][[j]][["down"]]
    # Synthetic’s guesses.

    # 4. True Positives (TP): real down + predicted down
    TP <- intersect(pred_down, P)

    # 5. False Positives (FP): synthetic said down, real said not down
    FP <- base::setdiff(pred_down, P)

    # 6. False Negatives (FN): real down, synthetic missed them
    FN <- base::setdiff(P, pred_down)

    # 7. True Negatives (TN): not down in real, not down in synthetic
    TN <- base::setdiff(N, pred_down)

    # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
    TPR_down <- length(TP) / (length(TP) + length(FN))
    FPR_down <- length(FP) / (length(FP) + length(TN))

    ####

    DE_TP[[comp]][["up"]] <- append(DE_TP[[comp]][["up"]], c(TPR_up))
    DE_TP[[comp]][["down"]] <- append(DE_TP[[comp]][["down"]], c(TPR_down))

    DE_FP[[comp]][["up"]] <- append(DE_FP[[comp]][["up"]], c(FPR_up))
    DE_FP[[comp]][["down"]] <- append(DE_FP[[comp]][["down"]], c(FPR_down))
  }
}

## functionize repetitions
create_plot_data <- function(DE_data, unique_comparisons) {
  plot_data <- data.frame()
  for (n in names(DE_data)) {
    n_clean <- strsplit(n, "_vs_")[[1]][1] %>% str_replace(., "_", " ")
    tmp <- data.frame(
      correct = DE_data[[n]] %>% unlist(),
      direction = rep(c("up", "down"),
        each = length(unique_comparisons)
      ),
      comparison = paste0(n_clean, " ", rep(c("up", "down"),
        each = length(unique_comparisons)
      ))
    )
    plot_data <- rbind(plot_data, tmp)
  }
  plot_data <- dplyr::filter(
    plot_data,
    !comparison %in% c("real up", "real down")
  )
  plot_data$seed <- str_replace(plot_data$comparison, " up", "") %>%
    str_replace(., " down", "") %>%
    str_replace(., "seed ", "")
  plot_data$comparison <- unique_comparisons

  return(plot_data)
}

print("Step 3 completed.")

plot_tpr <- create_plot_data(DE_TP, unique_comparisons)
plot_fpr <- create_plot_data(DE_FP, unique_comparisons)

output_dir <- file.path(bio_res_dir, param_dir)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

output_file <- file.path(output_dir, paste0("DE_lfc=", lfc_th, "_split_", split_no))
write.csv(plot_fpr, paste0(output_file, "_fpr.csv"))
write.csv(plot_tpr, paste0(output_file, "_tpr.csv"))


plot_df <- data.frame()
for (n in names(DE_correct)) {
  n_clean <- strsplit(n, "_vs_")[[1]][1] %>% str_replace(., "_", " ")

  tmp <- data.frame(
    correct = DE_correct[[n]] %>% unlist(),
    direction = rep(c("up", "down"),
      each = length(unique_comparisons)
    ),
    comparison = paste0(n_clean, " ", rep(c("up", "down"),
      each = length(unique_comparisons)
    ))
  )
  plot_df <- rbind(plot_df, tmp)
}
plot_df <- dplyr::filter(plot_df, !comparison %in% c("real up", "real down"))


# p <- ggplot() +
#  geom_boxplot(data = plot_df, aes(x = comparison, y = correct, fill = direction)) +
#  theme_bw() +
#  ylim(c(0, 1)) +
#  scale_fill_manual(values = rep(
#    c("#999999", "#e3e3e3"),
#    2 * length(datasets)
#  )) + # hcobject$layers_names
#  ggtitle(paste0("DE-Gene Preservation, split=", split_no)) +
#  theme(axis.text.x = element_text(angle = 90)) +
#  ylab("Correctly reconstructed DE genes \n across class comparisons [%]") +
#  xlab("")

# ggsave(
#  file = paste0(output_file, "_DE-genes.png"), bg = "transparent",
#  plot = p, width = 10, height = 6, dpi = 300
# )
