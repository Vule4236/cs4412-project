# CS 4412: Data Mining - Discovering Music Patterns in Spotify Data
## M2: Initial Implementation

**Author:** Vu Le | **Institution:** Kennesaw State University | **Email:** vle28@students.ksu.edu

---

## Reproducibility Statement

All results reported in this repository are reproducible using `random_state=42` applied globally and to all scikit-learn estimators. The notebook is designed for Google Colab. Upload `dataset.csv` to `/content/` before running. All outputs (CSV files, figures) are saved to the Colab working directory. No subdirectory setup is required.

---

## Discovery Questions

| # | Question | Technique | Milestone |
|---|----------|-----------|-----------|
| **Q1** | What natural groups of songs exist based on audio features, independent of genre labels? | K-Means Clustering | M2 (Complete) |
| **Q2** | Which audio features frequently co-occur across songs? | Apriori Association Rules | M3 (Planned) |
| **Q3** | What anomalous songs exist that do not conform to any discovered cluster? | Anomaly Detection | M3 (Planned) |

`track_genre` is **excluded from all clustering inputs**. Genre labels appear only as post-hoc interpretive context. The original Q3 (decision trees on genre labels) was supervised classification and has been corrected to anomaly detection.

---

## Inconsistencies Corrected from M1 Proposal

| Item | M1 Claim (Incorrect) | Notebook Actual | Correction Applied |
|------|-------------------|-----------------|-------------|
| Fully duplicate rows | "1,000+ duplicate rows" | 0 duplicate rows | Corrected to 0 in all documents |
| Genre imbalance ratio | "greater than 10x" | 1.0x (perfectly balanced) | Corrected to 1.0x |
| Dataset size | "170,000 songs" | 114,000 songs | Corrected throughout |
| Cluster quality | Overclaimed strong clusters | Silhouette = 0.2902 (moderate) | Explicitly stated as moderate |
| Q3 technique | Decision trees on genre labels | Anomaly detection | Q3 corrected to unsupervised |

## M2 Feedback Compliance

| Feedback Item | Status | Result |
|---|---|---|
| Log-transform `acoustic_electronic_ratio` (max Z = +87.4) | **Resolved** | `log1p` applied in Section 4; max Z reduced +87.4 → 13.61; Silhouette improved 0.2902 → 0.3312 (+14.1%); DB metric conflict resolved — all three metrics now agree on k=2 |
| Deduplication experiment to measure centroid bias | **Resolved** | Section 9: 89,740 deduplicated songs; max centroid shift = **16.679** (far exceeds 0.05 threshold — material bias confirmed); M3 will use deduplicated dataset |

---

## Actual Experimental Results (Verified Against Notebook HTML Output)

| Metric | Value |
|--------|-------|
| Dataset size | 114,000 rows × 21 columns |
| Fully duplicate rows | 0 |
| Track+artist duplicate pairs | 32,656 (28.65% of dataset) |
| Unique genres | 114 (1,000 records each; balance ratio = 1.0×) |
| Missing values (audio features) | 0 |
| Selected k | **2** |
| Silhouette Score (20K sample, k=2) | **0.3373** |
| Silhouette Score (full dataset, n=113,999) | **0.3312** (moderate-to-good; +14.1% from log1p) |
| Davies-Bouldin Index | **1.3086** (improved from 1.4663; now minimized at k=2) |
| Final Inertia | **732,826.5** |
| Anomaly threshold (99th percentile) | **4.9680** |
| Anomalous songs | **1,140** (1.00%) |
| Cluster 0 — Acoustic/Calm | **25,677 songs (22.5%)**; modal genre = sleep (3.8%) |
| Cluster 1 — Electronic/Energetic | **88,322 songs (77.5%)**; modal genre = happy (1.1%) |
| PCA variance captured | **PC1 = 43.0%, PC2 = 23.0% (total = 66.0%)** |
| Deduplication: songs retained | **89,740** (removed 24,259 = 21.3%) |
| Deduplication: max centroid shift | **16.679** — material bias confirmed |
| Deduplication: mean centroid shift | **3.281** |

### Pre vs. Post Log1p Transformation

| Metric | Pre-transformation | Post-transformation | Change |
|---|---|---|---|
| Silhouette (full dataset) | 0.2902 | **0.3312** | +0.0410 (+14.1%) |
| Davies-Bouldin | 1.4663 | **1.3086** | −0.1577 (improved) |
| Inertia | 790,507.9 | **732,826.5** | −57,681 (tighter) |
| PCA variance (PC1+PC2) | 57.0% | **66.0%** | +9.0% |
| DB metric agreement | Conflicted (favored k=3) | **All 3 agree on k=2** | Conflict resolved |
| Silhouette gap k=2 vs k=3 | 0.0058 | **0.0977** | 17× larger — decisive |

---

## Cluster Descriptions

**Cluster 0: Acoustic/Calm — 25,677 songs (22.5%); modal genre: sleep (3.8%)**

High acousticness (0.772), low energy (0.293), elevated instrumentalness (0.300), loudness = −14.45 dBFS, mood_index = 0.155. Captures the minority acoustic segment — acoustic, ambient, instrumental, and classical music. The 3.8% modal concentration confirms cross-genre composition.

**Cluster 1: Electronic/Energetic — 88,322 songs (77.5%); modal genre: happy (1.1%)**

High energy (0.743), high loudness (−6.46 dBFS), low acousticness (0.182), danceability = 0.598, mood_index = 0.330. Represents the dominant production mode of the Spotify corpus. The 1.1% modal concentration confirms cross-genre composition.

**Low modal concentrations (3.8% and 1.1%) confirm that both clusters transcend official genre taxonomies.**

---

## k = 2 Selection Justification — All Three Metrics Agree

**Silhouette:** Maximized at k = 2 (0.3373). Drops sharply at k = 3 (0.2396) — gap of **0.0977**, nearly 17× larger than the pre-transformation gap (0.0058). Decisive evidence for k = 2.

**Davies-Bouldin:** Minimized at k = 2 (1.3153). Resolves the prior conflict where DB had favored k = 3. All three metrics now converge.

**Elbow:** Steepest drop between k = 2 (129,016) and k = 3 (102,195) — **20.8% reduction**.

**Domain:** Two clusters align with the dominant energy-acousticness axis (r = −0.734), producing maximally interpretable profiles.

---

## Anomaly Detection Results

Threshold = **4.9680** (99th percentile). **1,140 songs (1.00%)** flagged. Top anomalies: white noise and ambient sleep recordings with energy as low as **0.00002** and acousticness up to **0.994** — non-musical audio content occupying extreme feature space positions. Prior concern about log-transform artifacts driving anomaly designations is resolved.

---

## Deduplication Experiment Results (Section 9)

| | Full Dataset | Deduplicated |
|---|---|---|
| Songs | 113,999 | **89,740** |
| Silhouette | 0.3312 | 0.3321 (+0.0009 — negligible) |
| Davies-Bouldin | 1.3086 | 1.3106 (+0.0020 — negligible) |
| Max centroid shift | — | **16.679** (tempo ~16 BPM; loudness ~8 dBFS) |

**Conclusion: Material bias confirmed. M3 will use the 89,740-song deduplicated dataset.**

---

## Feature Engineering Notes

### acoustic_electronic_ratio_log (M2 Revised)

Raw ratio reached max Z = **+87.4**. Log1p applied:

```python
df['acoustic_electronic_ratio_log'] = np.log1p(df['acoustic_electronic_ratio'])
```

Max Z reduced to **13.61** (6.4× reduction). All clustering metrics improved measurably.

---

## KDD Pipeline

```
+-----------+   +-----------+   +-----------+   +--------------+   +-------------+   +---------------+
| SELECTION |-->|    EDA    |-->| CLEANING  |-->|TRANSFORMATION|-->|  CLUSTERING |-->|INTERPRETATION |
| Load CSV  |   | Univariat |   | 1 row drop|   | Z-score std  |   | k=2 to 12   |   | Cluster names |
| 114K songs|   | Bivariate |   | Constraint|   | log1p(ratio) |   | k=2 selected|   | Anomaly flag  |
| 9 features|   | Corr mtx  |   | check     |   | mood_index   |   | Sil=0.3312  |   | Dedup expt    |
|           |   | Outlier   |   |           |   | PCA (viz)    |   | DB=1.3086   |   | Q1/Q3 answers |
+-----------+   +-----------+   +-----------+   +--------------+   +-------------+   +---------------+
                                                        |
                                               +--------v--------+
                                               | SECTION 9       |
                                               | Dedup Expt      |
                                               | 89,740 songs    |
                                               | Max shift=16.68 |
                                               | M3: dedup only  |
                                               +-----------------+
```

---

## Repository Structure

```
cs4412-project/
+-- dataset.csv
+-- notebooks/
|   +-- Project_spotify_analysis.ipynb   
+-- docs/
|   +-- CS4412_M2.docx                   
+-- README.md
```

**Notebook outputs:** `fig1_univariate_distributions.png` | `fig2_correlation_matrix.png` | `fig3_bivariate_scatter.png` | `fig4_genre_boxplots.png` | `fig5_cluster_evaluation_metrics.png` | `fig6_cluster_pca_visualization.png` | `fig7_radar_cluster_profiles.png` | `spotify_clustered.csv` | `cluster_profiles.csv` | `anomalous_songs.csv`

---

## Dataset

**URL:** https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

```python
from google.colab import files; files.upload()
df_raw = pd.read_csv('/content/dataset.csv')
```

### Clustering Features

| Feature | Type | Rationale |
|---------|------|-----------|
| danceability | Raw [0–1] | Rhythmic suitability |
| energy | Raw [0–1] | Perceptual intensity |
| loudness | Raw [dBFS] | Production amplitude |
| valence | Raw [0–1] | Emotional tone |
| acousticness | Raw [0–1] | Acoustic vs. electronic character |
| instrumentalness | Raw [0–1] | Vocal vs. instrumental content |
| tempo | Raw [BPM] | Rhythmic speed |
| mood_index | Derived | valence × danceability |
| acoustic_electronic_ratio_log | Derived (revised) | log1p(acousticness / (energy + ε)); max Z reduced +87.4 → 13.61 |

**Excluded:** `track_genre`, `speechiness`, `liveness`, `popularity`, `duration_ms`, `explicit`, `key`, `mode`, `time_signature`

---

## Data Quality Notes

**Missing values:** Zero across all nine audio features. Three metadata columns: 1 missing each (< 0.001%).

**Duplicate records:** Zero fully duplicate rows. 32,656 track+artist pairs (28.65%) confirmed to introduce material centroid bias (max shift = 16.679). M3 uses deduplicated dataset (89,740 songs).

**Skewness:** speechiness = 4.648, liveness = 2.106, loudness = −2.007, instrumentalness = 1.734.

---

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

All packages are pre-installed in Google Colab.

---

## Feedback Compliance Checklist

| Requirement | Status | Evidence |
|---|---|---|
| Sample data table | Complete | Table 1 (Section 2) |
| Quantified data quality | Complete | 0 duplicates, 32,656 pairs, skewness per feature |
| Corrected Q3 | Complete | Anomaly detection; white noise anomalies identified |
| Explicit evaluation metrics | Complete | Silhouette = 0.3312, DB = 1.3086, Inertia = 732,826.5 |
| KDD pipeline diagram | Complete | ASCII pipeline in notebook and README |
| Granular methodology | Complete | k = 2 to 12 sweep; all metrics; formal justification |
| Expanded challenges with mitigation | Complete | 5 limitations; dedup resolves centroid bias |
| Academic writing, no first person | Complete | All narrative third person |
| No prediction as discovery | Complete | Genre excluded; Q3 unsupervised |
| Silhouette interpreted correctly | Complete | 0.3312 = moderate-to-good; improvement quantified |
| **Log-transform `acoustic_electronic_ratio`** | **Complete** | log1p applied; max Z +87.4 → 13.61; Sil +14.1%; DB conflict resolved |
| **Deduplication experiment** | **Complete** | Max shift = 16.679 (material); M3 uses 89,740-song dataset |

---

## Values  Stale vs. Actual

| Item | Stale Value | Actual Output |
|------|------------|---------------|
| Silhouette (full dataset) | 0.2902 | **0.3312** |
| Silhouette (sample k=2) | 0.2989 | **0.3373** |
| Davies-Bouldin | 1.4663 | **1.3086** |
| Inertia | 790,507.9 | **732,826.5** |
| PCA PC1 | 36.0% | **43.0%** |
| PCA PC2 | 21.0% | **23.0%** |
| PCA total | 57.0% | **66.0%** |
| Anomaly threshold | 4.534 | **4.9680** |
| Cluster 0 identity | Electronic/Energetic | **Acoustic/Calm (22.5%)** |
| Cluster 1 identity | Acoustic/Calm | **Electronic/Energetic (77.5%)** |
| Cluster 0 modal genre | reggaeton (1.2%) | **sleep (3.8%)** |
| Cluster 1 modal genre | sleep (3.5%) | **happy (1.1%)** |
| k=2 DB value | 1.4744 | **1.3153** |
| k=3 Silhouette | 0.2931 | **0.2396** |
| k=3 DB | 1.0161 | **1.3932** |
| Dedup songs | Planned | **89,740** |
| Dedup max shift | Planned | **16.679 (material)** |

---

## References

1. M. Pandya, "Spotify Tracks Dataset," Kaggle, 2023.
2. J. Hartigan and M. Wong, "Algorithm AS 136: A K-Means Clustering Algorithm," *Applied Statistics*, 1979.
3. P. Rousseeuw, "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis," *Journal of Computational and Applied Mathematics*, 1987.
4. D. Davies and D. Bouldin, "A Cluster Separation Measure," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 1979.
5. M. Zaki and W. Meira Jr., *Data Mining and Machine Learning*, Cambridge University Press, 2020.
6. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, 2011.

