# CS 4412: Data Mining - Discovering Music Patterns in Spotify Data
## M2: Initial Implementation

**Author:** Vu Le | **Institution:** Kennesaw State University | **Email:** vle28@student.ksu.edu

---

## Reproducibility Statement

All results reported in this repository are reproducible using `random_state=42` applied globally and to all scikit-learn estimators. The notebook is designed for Google Colab. Upload `dataset.csv` to `/content/` before running. All outputs (CSV files, figures) are saved to the Colab working directory. No subdirectory setup is required.


## Discovery Questions

| # | Question | Technique | Milestone |
|---|----------|-----------|-----------| 
| **Q1** | What natural groups of songs exist based on audio features, independent of genre labels? | K-Means Clustering | M2 (Complete) |
| **Q2** | Which audio features frequently co-occur across songs? | Apriori Association Rules | M3 (Planned) |
| **Q3** | What anomalous songs exist that do not conform to any discovered cluster? | Anomaly Detection | M3 (Planned) |

### Discovery vs. Prediction

`track_genre` is **excluded from all clustering inputs**. Genre labels appear only as post-hoc interpretive context after cluster assignments are determined from audio features alone. The original Q3 (decision trees on genre labels) was supervised classification and has been corrected to anomaly detection, which is a valid unsupervised discovery objective.


## Actual Experimental Results (M2)

All values are verified against notebook execution output.

| Metric | Value |
|--------|-------|
| Dataset size | 114,000 rows x 21 columns |
| Fully duplicate rows | 0 |
| Track+artist duplicate pairs | 32,656 (28.6%) |
| Unique genres | 114 (1,000 records each; balance ratio = 1.0x) |
| Missing values (audio features) | 0 |
| Selected k | **2** |
| Silhouette Score (full dataset) | **0.2902** (moderate structure, not overclaimed) |
| Davies-Bouldin Index | **1.4663** |
| Final Inertia | 790,507.9 |
| Anomalous songs (99th percentile) | 1,140 (1.00%; threshold = 4.534) |
| Cluster 0 modal genre | reggaeton (1.2%) |
| Cluster 1 modal genre | sleep (3.5%) |
| PCA variance captured | PC1 = 36.0%, PC2 = 21.0% (total = 57.0%) |

### Cluster Descriptions

**Cluster 0: Electronic/Energetic.** This cluster contains songs with high energy, high loudness, and low acousticness. It is cross-genre in composition. The modal genre concentration of 1.2% confirms that the grouping is not genre-specific but reflects a shared audio profile across many genre labels.

**Cluster 1: Acoustic/Calm.** This cluster contains songs with high acousticness, low energy, and elevated instrumentalness. It captures acoustic, ambient, and classical music. The modal genre concentration of 3.5% confirms a similarly cross-genre composition.

### k = 2 Selection Justification

The selection of k = 2 is based on convergence across four criteria.

First, the silhouette score is maximized at k = 2 (0.2989) and declines monotonically from k = 4 onward. The silhouette score at k = 3 is 0.2931, which is only marginally lower (difference = 0.0058). This marginal difference is insufficient to justify the added interpretive complexity of a third cluster.

Second, the Davies-Bouldin index reaches its minimum at k = 3 (1.0161), nominally favoring three clusters. However, the improvement in Davies-Bouldin does not compensate for the loss in silhouette score or the additional interpretive burden of a third cluster.

Third, the elbow method shows the steepest inertia decrease in the k = 2 to k = 5 range, with no single sharp inflection point. The elbow method alone does not distinguish clearly between k = 2 and k = 3.

Fourth, the two-cluster solution aligns directly with the dominant structural axis in the data: the energy-acousticness correlation (r = -0.734). This produces interpretable cluster profiles consistent with known musical production categories. A three-cluster solution would require an additional grouping that is harder to motivate from the EDA evidence.

The silhouette score of 0.2902 on the full dataset indicates moderate structure. Music similarity is a continuous property, not a sharply categorical one, and this result is accurate and domain-consistent.

---

## KDD Pipeline

```
+-----------+   +-----------+   +-----------+   +--------------+   +-------------+   +---------------+
| SELECTION |-->|    EDA    |-->| CLEANING  |-->|TRANSFORMATION|-->|  CLUSTERING |-->|INTERPRETATION |
| Load CSV  |   | Univariat |   | 1 row drop|   | Z-score std  |   | k=2 to 12   |   | Cluster names |
| 114K songs|   | Bivariate |   | Constraint|   | mood_index   |   | k=2 selected|   | Anomaly flag  |
| 9 features|   | Corr mtx  |   | check     |   | ac_el_ratio  |   | Sil=0.2902  |   | Q1/Q3 answers |
|           |   | Outlier   |   |           |   | PCA (viz)    |   | DB=1.4663   |   |               |
+-----------+   +-----------+   +-----------+   +--------------+   +-------------+   +---------------+
```

---

## Repository Structure

```
cs4412-project/
+-- dataset.csv                           # Kaggle dataset (upload to /content/ in Colab)
+-- notebooks/
|   +-- Project_spotify_analysis.ipynb     # M2 notebook (corrected)
+-- docs/
|   +-- CS4412_M2.docx        # Summary paper
+-- README.md
```

**Notebook outputs** (saved to Colab working directory):
`fig1_univariate_distributions.png` | `fig2_correlation_matrix.png` | `fig3_bivariate_scatter.png` | `fig4_genre_boxplots.png` | `fig5_cluster_evaluation_metrics.png` | `fig6_cluster_pca_visualization.png` | `fig7_radar_cluster_profiles.png` | `spotify_clustered.csv` | `cluster_profiles.csv` | `anomalous_songs.csv`

---

## Dataset

**URL:** https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

```python
# Google Colab upload
from google.colab import files; files.upload()
df_raw = pd.read_csv('/content/dataset.csv')
```

### Clustering Features

| Feature | Type | Rationale |
|---------|------|-----------|
| danceability | Raw [0-1] | Rhythmic suitability |
| energy | Raw [0-1] | Perceptual intensity |
| loudness | Raw [dBFS] | Production amplitude |
| valence | Raw [0-1] | Emotional tone |
| acousticness | Raw [0-1] | Acoustic vs. electronic character |
| instrumentalness | Raw [0-1] | Vocal vs. instrumental content |
| tempo | Raw [BPM] | Rhythmic speed |
| mood_index | Derived | valence multiplied by danceability |
| acoustic_electronic_ratio | Derived | acousticness divided by (energy + epsilon) |

**Excluded from clustering:** `track_genre` (label: excluded to preserve unsupervised discovery), `speechiness`, `liveness`, `popularity`, `duration_ms`, `explicit`, `key`, `mode`, `time_signature`

---

## Data Quality Notes

**Missing values:** Zero missing values across all nine audio features. Three metadata columns each have one missing record (less than 0.001%).

**Duplicate records:** Zero fully duplicate rows. However, 32,656 track+artist pairs appear more than once (28.6%). This is caused by the dataset structure, where each song is replicated once per genre. Multi-genre songs are therefore overrepresented in the feature space, which may bias K-Means centroids toward their audio profiles. Deduplication by track_id is planned for M3.

**Feature skewness:** Instrumentalness and speechiness exhibit strong right skew. The majority of values are concentrated near zero, which compresses inter-song distances in these dimensions. Z-score standardization addresses scale differences but does not correct for non-normality.

**Correlation and Euclidean distance:** Energy and loudness are strongly correlated (r = 0.762). Energy and acousticness are strongly negatively correlated (r = -0.734). Because K-Means uses Euclidean distance, correlated features contribute redundant directional signal, effectively up-weighting the energy-acoustic axis. The derived feature acoustic_electronic_ratio partially addresses this by encoding the dominant axis explicitly.

**Outlier retention:** IQR-based outlier counts are reported for each feature, but outliers are retained in the analysis. Extreme values in Spotify audio features represent legitimate musical properties. For example, instrumentalness equal to 1.0 indicates a fully instrumental composition. Removing outliers would eliminate a meaningful portion of the musical spectrum.

---

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

All packages are pre-installed in Google Colab.



## Feedback Compliance Checklist

| Requirement | Status | Evidence |
|---|---|---|
| Sample data table with example rows | Complete | Table 1 (Section 2) with interpretation |
| Quantified data quality | Complete | 0 full duplicates, 32,656 track+artist pairs, 1.0x balance ratio |
| Corrected Q3 (anomaly detection) | Complete | No supervised classification; centroid distance flagging |
| Explicit evaluation metrics | Complete | Silhouette = 0.2902, Davies-Bouldin = 1.4663, Inertia = 790,507.9 |
| KDD pipeline diagram | Complete | ASCII pipeline in notebook header and README |
| Granular methodology | Complete | k = 2 to 12 full sweep; all three metrics; k = 2 formally justified |
| Expanded challenges with mitigation | Complete | Five limitations with explanation |
| Academic writing, no first person | Complete | All narrative in third person |
| No em dashes | Complete | All em dashes removed from notebook and README |
| No prediction as discovery | Complete | Genre excluded from inputs; Q3 corrected |
| Silhouette interpreted correctly | Complete | 0.2902 interpreted as moderate, not strong |


## Inconsistencies Corrected from M1 Proposal

| Item | M1 Claim (Incorrect) | Notebook Actual | Correction Applied |
|------|-------------------|-----------------|-------------|
| Fully duplicate rows | "1,000+ duplicate rows" | 0 duplicate rows | Corrected to 0 in all documents |
| Genre imbalance ratio | "greater than 10x" | 1.0x (perfectly balanced) | Corrected to 1.0x |
| Dataset size | "170,000 songs" | 114,000 songs | Corrected throughout |
| Cluster quality | Overclaimed strong clusters | Silhouette = 0.2902 (moderate) | Explicitly stated as moderate |
| Q3 technique | Decision trees on genre labels | Anomaly detection | Q3 corrected to unsupervised |


## References

1. M. Pandya, "Spotify Tracks Dataset," Kaggle, 2023.
2. J. Hartigan and M. Wong, "Algorithm AS 136: A K-Means Clustering Algorithm," Applied Statistics, 1979.
3. P. Rousseeuw, "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis," Journal of Computational and Applied Mathematics, 1987.
4. D. Davies and D. Bouldin, "A Cluster Separation Measure," IEEE Transactions on Pattern Analysis and Machine Intelligence, 1979.
5. M. Zaki and W. Meira Jr., Data Mining and Machine Learning, Cambridge University Press, 2020.
6. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, 2011.
