# SpaEF: Spatially Resolved Transcriptomics Data Element-Wise Denoising Framework Powered by Large Models

A new SRT data denoising method powered by pre-trained large models. 

### Usage 
We utilize OmiCLIP[1] and GenePT[2] as feature encoders or prior knowledge injectors. 

You should download the checkpoints and other files from their tutorials first.

checkpoint.pt of OmiCLIP: https://guangyuwanglab2021.github.io/Loki/pretrain.html

GenePT_gene_embedding_ada_text.pickle of GenePT: https://github.com/yiqunchen/GenePT

And then you can start easily by using the following script after downloading the source code.

```
cd SpaEF
conda env create -f environment.yml
conda activate SpaEF
python main.py --root_path YOUR_INPUT_DATA_PATH 
```
Expression.csv (spots * genes) and coordinate.csv (spots * coordinates) should be in YOUR_INPUT_DATA_PATH. 

### Benchmark
For benchmarks, you can download them by following the links:

HOCWT: https://www.10xgenomics.com/datasets/human-ovarian-cancer-whole-transcriptome-analysis-stains-dapi-anti-pan-ck-anti-cd-45-1-standard-1-2-0

HGBM: https://www.10xgenomics.com/datasets/gene-and-protein-expression-library-of-human-glioblastoma-cytassist-ffpe-2-standard

HDHBC: https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-human-breast-cancer-ffpe-if

DLPFC: https://research.libd.org/globus/

We supply a pre_process.py to handle the raw dataset.
```
python pre_process.py
```

# Acknowledgments 
We utilize OmiCLIP[1] and GenePT[2] as feature encoders or prior knowledge injectors. The housekeeping_genes.csv we supplied is is obtained from the Molecular Signatures Database (MSigDB): https://www.gsea-msigdb.org/gsea/msigdb/.

[1] Chen, W., Zhang, P., Tran, T.N. et al. A visual–omics foundation model to bridge histopathology with spatial transcriptomics. Nat Methods 22, 1568–1582 (2025). https://doi.org/10.1038/s41592-025-02707-1

[2] Chen, Y., Zou, J. Simple and effective embedding model for single-cell biology built from ChatGPT. Nat. Biomed. Eng 9, 483–493 (2025). https://doi.org/10.1038/s41551-024-01284-6


