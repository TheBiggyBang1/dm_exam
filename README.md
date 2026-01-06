# Customer Supermarket — Project README ✅

This README documents the notebook (`examen_dm.ipynb`) and the companion Streamlit app (`streamlit_app.py`). It explains every code cell in the notebook in a concise, actionable way so you can reproduce the analysis and understand the outputs.

---

## Quick start

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. To run the Streamlit app (interactive view):

```bash
streamlit run streamlit_app.py
```

4. To run the notebook: open `examen_dm.ipynb` in Jupyter / VS Code and run cells sequentially.

---

## Files

- `examen_dm.ipynb` — Main notebook (data load → cleaning → features → clustering → classification → results/tests).
- `streamlit_app.py` — Interactive reproduction of the notebook in a tabbed UI (Overview / Cleaning & Features / Clustering / Classification / Interpretation / Code).
- `data/customer_supermarket.csv` — Source dataset (transaction rows).
- `requirements.txt` — Python dependencies (use `python -m pip install -r requirements.txt`).

---

## Notebook — Cell-by-cell explanations (short) ✍️

1. **Imports & environment setup**
   - Imports core libs (pandas, numpy, matplotlib, seaborn, scipy, sklearn). Sets seed and plotting style.
   - Purpose: dependency checking and plotting defaults.

2. **Load dataset into `df`**
   - Reads `data/customer_supermarket.csv` with `sep='\t'`, `decimal=','` and `index_col=0`.
   - Outputs the shape (expected: `Rows, Columns: (471910, 8)`) and `df.head()` for quick sanity.

3. **Dataframe info & null counts**
   - Runs `df.info()` and prints null counts to highlight parsing problems or missing values.

4. **Cleaning: convert & filter**
   - Parses `BasketDate` (day-first), coerces `Sale` and `Qta` to numeric, computes `Amount = Sale * Qta`.
   - Removes negative `Qta` rows, zero `Sale` rows and rows with missing `CustomerID`.
   - Prints counts of rows dropped and the final shape / date range.

5. **Feature extraction (customer-level)**
   - Aggregates transaction rows to compute per-customer features:
     - `I`: total product rows (proxy for total items)
     - `Iu`: unique products purchased
     - `Imax`: max items in a single basket
     - `Entropy`: spending entropy across baskets (Entropy = -Σ p_b log p_b)
     - `BasketNum`: number of baskets (visits)
     - `SumExp`: total spend
     - `AvgExp`: average basket spend
   - Produces `df_customer` for downstream analyses.

6. **Outlier removal (z-score)**
   - Removes customers with |z| ≥ 3 on any feature (robustness step). Keeps result as `new_df`.

7. **Pairplot and correlation heatmap**
   - Visual diagnostics (sampled pairplot to inspect relationships and correlation heatmap to spot correlated features such as `BasketNum` vs `SumExp`).
   - Output: insight on feature relationships and multicollinearity signals.

8. **KMeans visuals (I vs BasketNum scatter + centers, cluster sizes)**
   - Plots clusters on `I` vs `BasketNum` with centers, and a bar chart with cluster sizes.
   - Use: identify customer segments (e.g., high-value customers: large `SumExp` and `BasketNum`).

9. **3D cluster scatter and centers**
   - 3D scatter (`I`, `BasketNum`, `SumExp`) to assess separation in three dimensions; prints cluster centers (numeric) and identifies top cluster by `SumExp`.

10. **Normalization and KMeans (actual run)**
    - Scales features with `MinMaxScaler` and runs KMeans (fixed n=4 in the original notebook). Prints cluster counts and plots cluster center parallel-coordinates.

11. **DBSCAN: Knee method & clustering**
    - Computes k-distance plot using `NearestNeighbors` and selects `eps` heuristically (e.g., 95th percentile).
    - Runs DBSCAN and prints labels and counts (including noise `-1`). Use: density-based segmentation and outlier detection.

12. **DBSCAN summary**
    - Reports labels counts and noise fraction (noise % indicates dispersion in customer behavior).

13. **Hierarchical clustering**
    - Builds distance linkage (Ward), draws a truncated dendrogram and cuts the tree at a chosen distance to extract clusters. Use: assess multi-level grouping and sub-cluster structure.

14. **Build classification dataset (tertiles)**
    - Creates class labels `low`/`medium`/`high` based on `SumExp` tertiles, drops unused columns and splits data to train/test (70/30, stratified).

15. **Model helper functions**
    - `grid_search` (RandomizedSearchCV) to tune hyperparameters (score by accuracy and f1_weighted); prints top parameter combos.
    - `report_scores` fits a classifier, prints `classification_report` and shows a confusion matrix.

16. **Gaussian Naive Bayes**
    - Trains and evaluates GNB; prints metrics and confusion matrix. Provides baseline performance.

17. **Decision Tree (RandomizedSearch)**
    - Randomized hyperparameter search for tree, trains final Decision Tree and reports metrics & confusion matrix.

18. **Random Forest (RandomizedSearch)**
    - Randomized hyperparameter search for RF, trains the selected model, shows classification report, confusion matrix, and top feature importances.
    - Use RF importances to guide practical actions (e.g., target by `BasketNum` or `Entropy`).

19. **KNN (k selection by error rate)**
    - Searches `k` (1..30) by test error rate, picks `k` with minimum error and evaluates KNN.

20. **SVM (Randomized Search over kernels)**
    - Randomized search over kernels, trains SVM with best params and evaluates.

21. **Results: distributions & cluster counts**
    - Plots histograms for features and prints KMeans cluster counts. Use as final summary visual.

22. **Assertions / Sanity checks**
    - Basic assertions to ensure no missing `CustomerID`, non-negative `Qta`, `Amount` column present, and `new_df` not empty.
    - If these pass, dataset is consistent for reported analyses.

---

## Streamlit app (`streamlit_app.py`) — short doc

- Reproduces the notebook pipeline in a **tabbed UI**:
  - **Overview** (head, rows/cols, basic info, feature descriptions)
  - **Cleaning & Features** (feature table, pairplot, correlation heatmap)
  - **Clustering** (silhouette-based K selection, KMeans scatter & centers, DBSCAN k-distance + parameter sweep, hierarchical dendrogram)
  - **Classification** (quick models, Random Forest limited tuning + cross-validated metrics, confusion matrices, feature importances)
  - **Interpretation** (final takeaways: RF CV accuracy/F1 and actionable insights)
  - **Code** (notebook code cells with short explanations)

- Notes:
  - Dendrogram and long tuning operations are gated (or run with modest sample sizes) to keep UI responsive.
  - DBSCAN parameter sweep table and recommended parameter combos are provided for diagnostics.

---

## Notes & tips

- **Entropy**: computed per customer as -Σ_b p_b log(p_b) with p_b the fraction of that customer's spend in basket b (natural log - units: nats). Higher entropy = more evenly distributed spending across baskets.
- **DBSCAN**: noise fraction is the proportion labeled `-1`; high noise suggests dispersion or too-strict `eps` / `min_samples` choices. Use the k-distance plot or the app's DBSCAN sweep to diagnose.
- **Modeling**: the notebook uses tertiles of `SumExp` (discrete labels). Consider a regression approach if continuous spend prediction is preferred.

---

If you want, I can also generate an expanded README that includes the full code from each cell (or export the notebook cells as ordered blocks), or add a short section mapping notebook cell indices to line numbers for quick reference. Which would you prefer next?