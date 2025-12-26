# Datasets 

## Track I: bulk RNA-seq datasets

We re-distribute two open access TCGA RNA-seq datasets, which can be accessed from the [GDC portal](https://gdc.cancer.gov), in the pre-processed form:

- **TCGA-BRCA RNASeq** 

    **Dimensions:** <1089 x 978> <individuals x landmark genes>
    **Details:** Suitable for cancer subtype prediction (5 subtypes)
    - LumA
    - LumB
    - Her2
    - Normal
    - Basal

- **TCGA-COMBINED RNASeq** (with 10 different tissues )

    **Dimensions:** <4323 x 978> <individuals x landmark genes>
    **Details:** Suitable for cancer tissue-of-origin prediction (10 tissues)
    - "TCGA-BRCA" = "Breast",
    - "TCGA-COAD" = "Colorectal",
    - "TCGA-ESCA" = "Esophagus",
    - "TCGA-KIRC" = "Kidney",
    - "TCGA-KIRP" = "Kidney",
    - "TCGA-LIHC" = "Liver",
    - "TCGA-LUSC" = "Lung",
    - "TCGA-LUAD" = "Lung",
    - "TCGA-OV" = "Ovarian",
    - "TCGA-PAAD" = "Pancreatic",
    - "TCGA-PRAD" = "Prostate",
    - "TCGA-SKCM" = "Skin"


You can download the pre-processed datasets from [ELSA Benchmarks Competition platform](https://benchmarks.elsa-ai.eu/?ch=4&com=introduction).


## Preparing the datasets

### Download
The datasets are downloaded using `TCGABiolinks` R package. 

```r
library(TCGAbiolinks)
raw_dir <- "data/raw/"
download_dir <- "data/raw/downloads"
cancer_types <- c("TCGA-BRCA", "TCGA-COAD", "TCGA-ESCA", "TCGA-KIRC",
                  "TCGA-KIRP", "TCGA-LIHC", "TCGA-LUSC", "TCGA-LUAD",
                  "TCGA-OV", "TCGA-PAAD", "TCGA-PRAD", "TCGA-SKCM")

for (proj in cancer_types) {
  print(paste("Downloading data for:", proj))
  filename <- paste(proj, "_primary_tumor_star_counts.rda", sep = "")
  if (!file.exists(file.path(getwd(), raw_dir, filename))) {
    query <- GDCquery(
      project = proj,
      data.category = "Transcriptome Profiling",
      data.type = "Gene Expression Quantification",
      workflow.type = "STAR - Counts",
      access = "open",
      sample.type = c("Primary Tumor")
    )

    if (!dir.exists(file.path(getwd(), download_dir, proj))) {
      if (!dir.exists(file.path(getwd(), download_dir))) {
        dir.create(file.path(getwd(), download_dir), recursive = TRUE)
      }
      GDCdownload(query, directory = file.path(getwd(), download_dir))
    }

    data <- GDCprepare(query,
                      save = TRUE,
                      save.filename = file.path(getwd(), raw_dir, filename),
                      directory = download_dir)
  }


}
```

### Pre-processing
The raw count data is preprocessed as follows: 
- Genes with low counts are removed. 
- The counts are normalized using DeSeq2's VST function.
- The normalized data is further filtered to landmark genes (lmgenes_filename = f1000_lm_ensembl.tsv). The landmark genes (n=978) are  identified as representative genes that allow the inference of around 20K other genes by the LINCS L1000 (Subramanian et al., 2017).

```r
# Load necessary libraries
library(DESeq2)

# Define the normalize_counts_filter_to_lm_genes function
normalize_counts_and_filter2LM_genes <- function(selected_data,
                                                type_meta,
                                                type_col,
                                                meta_dir,
                                                lmgenes_filename = "",
                                                filter_to_lm = TRUE) {
  # load landmark F1000 genes
  if (!file.exists(file.path(meta_dir, lmgenes_filename))) {
    source(file.path(getwd(), "src/prepare_data", "_get_ensembl_id.r"))
  }
  lm_genes <- read.csv(file.path(meta_dir, lmgenes_filename),
                       header = TRUE,
                       row.names = NULL)

  # Create DESeq2 dataset
  design_formula <- as.formula(paste("~", type_col))
  # Check if the design variable exists in colData
  if (!type_col %in% colnames(type_meta)) {
    stop(paste("Error: The design variable", type_col, "is not a column in colData"))
  }

  dds <- DESeqDataSetFromMatrix(countData = selected_data,
                                colData = type_meta,
                                design = design_formula)

  # filtering low count genes
  count_th <- 1
  sample_th <- dim(dds)[2] * 0.1
  keep <- rowSums(counts(dds) >= count_th) >= sample_th
  dds <- dds[keep, ]
  print(dim(dds))
  
  # Run DESeq to perform normalization
  dds <- DESeq(dds)
  vst_counts <- vst(dds, blind = FALSE)
  normalized_counts <- assay(vst_counts)


  rownames(normalized_counts) <- sub("\\..*$", "", rownames(normalized_counts))

  # if filter_to_lm is TRUE, filter to landmark F1000 genes
  if (filter_to_lm) {
    filtered_norm_counts <- normalized_counts[
      rownames(normalized_counts) %in% lm_genes$ENSEMBL.ID,
    ]
  } else {
    filtered_norm_counts <- normalized_counts
  }

  return(filtered_norm_counts) 
}
```