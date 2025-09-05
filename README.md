# DNA Sequence Classification using CNN-GRU Hybrid Model

**Authors:** Vijay B. Vishwakarma, Sankalp Gupta, Vinay Kumar, Pranshu Yadav, Ujjwal Mishra

## Abstract
DNA sequence classification into functional categories like promoters and enhancers is a key challenge in genomics. This study proposes a hybrid CNN-GRU model, combining CNNs for local motif detection with GRUs for long-range dependencies, achieving higher accuracy on benchmark datasets compared to traditional CNN models.

## Introduction
DNA, composed of nucleotides A, T, G, and C, encodes life’s blueprint. Classifying sequences into promoters, enhancers, and other regulatory elements is vital for gene expression and health. Traditional methods struggle with complex patterns, while deep learning, especially the CNN-GRU hybrid, offers a promising solution.

## Methodology
### Dataset
- **Genomic Benchmark Dataset:** Used 3 datasets:
  - `human_nontata_promoters`: 36,131 sequences, 2 classes
  - `human_enhancers_ensembl`: 154,842 sequences, 2 classes
  - `drosophila_enhancers_stark`: 6,914 sequences, 2 classes

### Preprocessing & Encoding
- K-mer extraction (k=6) with optional jumping k-mer strategy for efficiency.
- Sequences padded with "_" and zeros for uniform length.
- Embeddings generated as 32-dimensional vectors.

### Model Architecture
- **Embedding Layer:** Converts k-mers to dense vectors.
- **1D CNN Layer:** Extracts motifs (kernel_size=7, filters=64).
- **Max-Pooling Layer:** Reduces sequence length.
- **Bidirectional GRU Layer:** Captures dependencies (hidden_dim=128).
- **Dropout Layer:** Prevents overfitting (rate=0.5).
- **Fully Connected Layer:** Outputs class probabilities.

### Training
- Loss: Categorical cross-entropy.
- Optimizer: Adam (learning rate=0.001).
- Early stopping on validation loss.

### Evaluation
- Metrics: Accuracy, precision, recall, F1-score.
- Compared traditional vs. jumping k-mer encodings.

## Results
| Model              | Dataset                  | Approach    | Accuracy | F1 Score |
|---------------------|--------------------------|-------------|----------|----------|
| Baseline CNN        | human_nontata_promoters  | -           | 84.6     | 83.7     |
| Baseline CNN        | human_enhancers_ensembl  | -           | 68.9     | 56.5     |
| Baseline CNN        | drosophila_enhancers_stark | -       | 58.6     | 44.5     |
| Hybrid CNN + GRU    | human_nontata_promoters  | -           | 92.55    | 92.56    |
| Hybrid CNN + GRU    | human_enhancers_ensembl  | *           | 86.13    | 86.13    |
| Hybrid CNN + GRU    | drosophila_enhancers_stark | *       | 50.00    | 33.33    |

(-) Traditional k-mers, (*) Jumping k-mers

## Conclusion
The CNN-GRU hybrid outperforms the baseline CNN, especially on human datasets (92.5% and 86.13% accuracy), though performance stagnates on *drosophila_enhancers_stark* at 50%.

## References
- Grešová et al. (2023). *Genomic Benchmarks*. BMC.
- Quang & Xie (2016). *DanQ*. Nucleic Acids Res.
- Ji et al. (2021). *DNABERT*. Bioinformatics.
- Shen et al. (2018). *GRU for TF Binding*. Sci Rep.
- And more (see PDF for full list).

## Usage
Clone the repo and run the model using the provided scripts. Adjust hyperparameters as needed.

## Contributing
Feel free to fork, improve, and submit pull requests!
