# CAMDA 2026 - ELSA Health Privacy Challenge
The Health Privacy Challenge is back in 2026, featuring expanded baseline methods and evaluation metrics and an extended submission period across all tasks.

This repository is a "starter package" for the [Health Privacy Competition](https://benchmarks.elsa-ai.eu/?ch=8) that runs within [CAMDA Conference 2026](). The  Health Privacy Challenge is organized in the context of the European Lighthouse on Safe and Secure AI (ELSA, https://elsa-ai.eu). 

The Health Privacy Challenge consists of two tracks: 

### [Track I: Featuring Bulk RNA-seq](/experiments/track_i/) 
Track I runs in a “Blue Team (🫐)  vs Red Team (🍅)” scheme. 
-  The **blue teams** develop **novel privacy preserving generative methods** that can mitigate privacy risks while preserving biological insights for bulk gene expression datasets,
- The **red teams**  act as adversaries by deploying state‑of‑the‑art, realistic membership inference attacks (MIAs) **against a set of the baseline and Blue Teams’ solutions from the CAMDA 2025 Challenge**, in order to  assess the privacy robustness of the proposed generative methods.


### [Track II: Featuring Single-cell RNA-seq](/experiments/track_ii/) 
Track II invites participants to explore the privacy and utility of synthetic single-cell gene expression (scRNA-seq) data. Participants are encouraged to:
- **investigate and reveal potential privacy risks** linked to generating synthetic **scRNA-seq** datasets.
- develop **privacy-preserving generative methods** that balances data privacy and utility.
- propose **novel evaluation metrics and strategies** to assess both utility and privacy preservation in a **multi-sample donor** setting.


We are looking forward to engaging with you and working together to deepen our understanding of privacy in healthcare. :hugs: 

## Introduction  

**This repository contains:**

- :woman_technologist: **Baseline code** for generative methods (**Blue Teams**) and  Membership Inference Attack algorithms  (**Red teams**).
- :memo: **Documentation** that details setup and submission instructions for the competition. 
- :paperclip: **Submission templates** to base your submissions on. 

**Other resources:**

- :speech_balloon: **[CAMDA Health Privacy Challenge Google Groups:](https://groups.google.com/g/camda-health-privacy-challenge)** Join us for questions, discussions and further announcements. 
- :globe_with_meridians: **[CAMDA Challenge website:](https://bipress.boku.ac.at/camda2025/)** Follow CAMDA 2026 for conference announcements. **Will be updated soon**
- :globe_with_meridians: **[ELSA Benchmark method submission platform:](https://benchmarks.elsa-ai.eu/?ch=8)** The platform to register, to download datasets, and to submit your benchmark methods. 
- :books: **Relevant papers:** https://arxiv.org/abs/2402.04912 

## :roller_coaster: Get started!  
All teams, please check out your home pages to set up and use the starter package!

- [Track I Blue Teams Home Page](/experiments/track_i/blue_team/)
- [Track I Red Teams Home Page](/experiments/track_i/red_team/)
- [Track II Single-cell Home Page](/experiments/track_ii/)


## Datasets 

Datasets are available for download in [ELSA Benchmarks Competition platform](https://benchmarks.elsa-ai.eu/?ch=8&com=introduction) after registration and signing the data download agreement. 

### Track I: Featuring bulk RNA-seq
We re-distribute pre-processed versions of two open-access TCGA RNA-seq datasets, available through the  [GDC portal](https://gdc.cancer.gov):

- **TCGA-BRCA RNASeq** 

    **Dimensions:** <1089 x 978> <individuals x landmark genes>
    **Details:** Suitable for cancer subtype prediction (5 subtypes)

- **TCGA COMBINED RNASeq** (with 10 different cancer tissues )

    **Dimensions:** <4323 x 978> <individuals x landmark genes>
    **Details:** Suitable for cancer tissue of origin prediction (10 tissues)

Navigate [here](/data/) for details about the pre-processing steps. 

### Track II: Featuring single-cell RNA-seq 
We re-distribute raw counts of **OneK1K single-cell RNA-seq** dataset (https://onek1k.org/), a cohort containing 1.26 million peripheral blood mononuclear cells (PBMCs) of 981 donors, generously provided by [Joseph Powell](https://www.garvan.org.au/people/researchers/joseph-powell) and the authors (Yazar et al., 2022) in Garvan Institute of Medical Research. 

- **Train dataset:** <633711 cells from 490 donors x 25834 genes > 
- **Test dataset:**  <634022 cells from 491 donors x 25834 genes > 

Navigate [Track II homepage](/experiments/track_ii/) for details about the pre-processing steps. 


##  :date: Schedule 

#### Exact submission dates will be updated soon!

![Timeline](timeline-2026.png)


## :busts_in_silhouette: Organization Team  
This competition is designed as a collaborative effort between [European Molecular Biology Laboratory  (EMBL)](https://www.embl.org), [CISPA Helmholtz Center for Information Security](https://cispa.de/en), and the [University of Helsinki](https://www.helsinki.fi/en) with the support of [Barcelona Computer Vision Center (CVC)](https://www.cvc.uab.es) within the context of ELSA Project.  

- **EMBL:** [Hakime Öztürk](https://github.com/hkmztrk), [Julio Saez-Rodriguez](https://saezlab.org) and [Oliver Stegle](https://steglelab.org)
- **CISPA:** [Tejumade Afonja](https://github.com/tejuafonja), [Ruta Binkyte](https://github.com/RuSaBin) and [Mario Fritz](https://cispa.de/en/research/groups/fritz)
- **University of Helsinki:**  [Joonas Jälkö](https://researchportal.helsinki.fi/en/persons/joonas-jälkö/) and [Antti Honkela](https://www.cs.helsinki.fi/u/ahonkela/)

and in collaboration with **Saez-Rodriguez** group in Track II and the review process:
- **University of Heidelberg:** [Sebastian Lobentanzer](https://github.com/slobentanzer), and [Pablo R. Mier](https://github.com/pablormier).


We also thank [Katharina Mikulik (DKFZ)](https://steglelab.org/katharina-mikulik/),  [Kevin Domanegg (DKFZ)](https://steglelab.org/kevin-domanegg/), and [Danai Vaigaki (EMBL)](https://steglelab.org/danai-vagiaki/) for helpful feedback. 

<!-- 
## :pushpin: Statement

Membership inference attacks (MIA) aim to re-identify the training data points used to generate synthetic datasets from the original dataset. This re-identification process pertains only to identifying the pseudo-identities within the dataset and **does not, in any way, attempt to re-identify the original donors.**
-->