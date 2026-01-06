import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from io import StringIO

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, precision_recall_fscore_support
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

sns.set(style='whitegrid')

# Reduce overall figure footprint for better readability in the app
plt.rcParams['figure.figsize'] = (5,2.5)
plt.rcParams['figure.dpi'] = 100

st.set_page_config(layout='centered', page_title='Customer Supermarket Analysis')

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'customer_supermarket.csv')
NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), 'examen_dm.ipynb')

@st.cache_data
def load_raw_data(path=DATA_PATH):
    df = pd.read_csv(path, sep='\t', index_col=0, decimal=',')
    return df

@st.cache_data
def clean_data(df):
    df = df.copy()
    df['BasketDate'] = pd.to_datetime(df['BasketDate'], dayfirst=True, errors='coerce')
    df['Sale'] = pd.to_numeric(df['Sale'], errors='coerce')
    df['Qta'] = pd.to_numeric(df['Qta'], errors='coerce')
    df['Amount'] = df['Sale'] * df['Qta']

    # removal steps
    df = df[df['Qta'] >= 0]
    df = df[df['Sale'] != 0]
    df = df.dropna(subset=['CustomerID'])
    return df

@st.cache_data
def compute_customer_features(df):
    cust_group = df.groupby('CustomerID')
    I = cust_group['ProdID'].count().rename('I')
    Iu = cust_group['ProdID'].nunique().rename('Iu')
    Imax = df.groupby(['CustomerID', 'BasketID']).size().groupby(level=0).max().rename('Imax')

    pos_amounts = df[df['Amount'] >= 0].copy()
    p_sum = pos_amounts.groupby('CustomerID')['Amount'].transform('sum')
    probs = pos_amounts['Amount'] / p_sum
    pos_amounts['Entropy'] = -(probs * np.log(probs))
    Entropy = pos_amounts.groupby('CustomerID')['Entropy'].sum().rename('Entropy')

    BasketNum = df.groupby(['CustomerID', 'BasketID']).size().groupby(level=0).size().rename('BasketNum')
    BasketSum = df.groupby(['CustomerID', 'BasketID']).agg(BasketSum=('Amount', 'sum'))
    SumExp = BasketSum.groupby('CustomerID')['BasketSum'].sum().rename('SumExp')
    AvgExp = BasketSum.groupby('CustomerID')['BasketSum'].mean().rename('AvgExp')

    df_customer = pd.concat([I, Iu, Imax, Entropy, BasketNum, SumExp, AvgExp], axis=1)
    df_customer['Entropy'] = df_customer['Entropy'].fillna(0)
    return df_customer

@st.cache_data
def remove_outliers(df_customer):
    z_scores = zscore(df_customer.fillna(0))
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    new_df = df_customer[filtered_entries].copy()
    return new_df

def plot_pairplot(sample_df):
    fig = sns.pairplot(sample_df, corner=True, plot_kws={'alpha':0.4, 's':20})
    return fig

def run_kmeans(new_df, n_clusters=4):
    scaler = MinMaxScaler()
    df_transformed = scaler.fit_transform(new_df.values)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, max_iter=100, random_state=42)
    kmeans.fit(df_transformed)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    return kmeans, centers

def run_dbscan(new_df):
    scaler = MinMaxScaler()
    df_transformed = scaler.fit_transform(new_df.values)
    min_pts = new_df.shape[1] * 2
    neighbors = NearestNeighbors(n_neighbors=min_pts)
    neighbors_fit = neighbors.fit(df_transformed)
    distances, indices = neighbors_fit.kneighbors(df_transformed)
    distances = np.sort(distances[:,1])
    eps = np.percentile(distances, 95)
    dbscan = DBSCAN(eps=eps, min_samples=min_pts)
    dbscan.fit(df_transformed)
    return dbscan, distances

def hierarchical_clustering(new_df):
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage
    scaler = MinMaxScaler()
    df_transformed = scaler.fit_transform(new_df.values)
    data_dist = pdist(df_transformed, metric='euclidean')
    data_link = linkage(data_dist, method='ward')
    return data_link

# K selection via silhouette analysis
@st.cache_data
def run_kmeans_silhouette(new_df, k_range=range(2,9)):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(new_df.values)
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        try:
            s = silhouette_score(X, labels)
        except Exception:
            s = np.nan
        scores[k] = s
    best_k = max(scores, key=lambda k: scores[k] if not np.isnan(scores[k]) else -np.inf)
    return best_k, scores

# Grid sweep for DBSCAN (small, reasonable grid)
@st.cache_data
def run_dbscan_grid(new_df, eps_percentiles=[85,90,95,97.5], min_samples_list=[3,5,7]):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(new_df.values)
    neighbors = NearestNeighbors(n_neighbors=max(3, new_df.shape[1]*2))
    neighbors_fit = neighbors.fit(X)
    distances_full, _ = neighbors_fit.kneighbors(X)
    distances_full = np.sort(distances_full[:,1])

    results = []
    for p in eps_percentiles:
        eps = np.percentile(distances_full, p)
        for m in min_samples_list:
            db = DBSCAN(eps=eps, min_samples=m)
            db.fit(X)
            labels = db.labels_
            unique_labels = set(labels)
            n_clusters = len([l for l in unique_labels if l != -1])
            noise = int((labels == -1).sum())
            results.append({'eps_percentile': p, 'eps': float(eps), 'min_samples': m, 'n_clusters': int(n_clusters), 'noise_count': int(noise), 'noise_pct': float(noise/len(labels))})
    return pd.DataFrame(results)

# Random Forest quick tuning (limited iterations for speed)
@st.cache_data
def tune_random_forest(X, y, n_iter=20, cv=3):
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    rf = RandomForestClassifier(random_state=42)
    rsearch = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=n_iter, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42)
    rsearch.fit(X, y)
    best = rsearch.best_estimator_
    stats = rsearch.cv_results_
    return best, rsearch.best_params_, stats

# Simple helper to extract Python code cells from the notebook
@st.cache_data
def extract_notebook_code(nb_path=NOTEBOOK_PATH):
    """Return a list of dicts: {'code': str, 'explanation': str}. Tries to pair each code cell with the next nearby markdown cell."""
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        cells = nb.get('cells', [])
        results = []
        for idx, c in enumerate(cells):
            if c.get('cell_type') == 'code':
                code = ''.join(c.get('source', []))
                # find next markdown cell within the next 2 cells
                explanation = ''
                for j in range(idx+1, min(idx+4, len(cells))):
                    if cells[j].get('cell_type') == 'markdown':
                        explanation = ''.join(cells[j].get('source', []))
                        break
                if not explanation:
                    explanation = 'No notebook explanation cell found; this cell is taken from the analysis notebook.'
                results.append({'code': code, 'explanation': explanation})
        return results
    except Exception as e:
        return [{'code': f'Error reading notebook: {e}', 'explanation': ''}]

# Classification helpers

def build_classification_dataset(new_df):
    df_clf = new_df.copy()
    df_clf.sort_values(by='SumExp', inplace=True)
    length = len(df_clf)
    l1 = length // 3
    l2 = l1 * 2
    labels = pd.Series(index=df_clf.index, dtype='object')
    labels[:] = 'low'
    labels.iloc[l1:l2] = 'medium'
    labels.iloc[l2:] = 'high'
    X = df_clf.drop(columns=['Iu', 'Imax', 'SumExp', 'AvgExp'], errors='ignore')
    y = labels
    return train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

@st.cache_data
def fit_quick_models(X_train, y_train):
    models = {}
    # Quick default models
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    models['GaussianNB'] = gnb

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    models['DecisionTree'] = dt

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    models['KNN'] = knn

    svm = SVC(random_state=42)
    svm.fit(X_train, y_train)
    models['SVM'] = svm

    return models

# UI - TABS layout, polished presentation
st.title('Customer Supermarket — Analysis & Models')

# Minimal UI: always show notebook code and hide sidebar controls for a cleaner layout
show_code = True
st.markdown("**Note:** Sidebar controls hidden to keep the app focused. Notebook code is shown by default.")

# Create tabs for each main section
tab_overview, tab_features, tab_clustering, tab_classif, tab_interpret, tab_code = st.tabs([
    'Overview', 'Cleaning & Features', 'Clustering', 'Classification', 'Interpretation', 'Code'
])

# Overview tab
with tab_overview:
    st.header('Dataset overview')
    raw = load_raw_data()
    st.write('Rows, Columns:', raw.shape)

    st.subheader('First rows (head)')
    st.dataframe(raw.head().style.format(precision=2))

    st.subheader('Basic info (dtypes / nulls / unique)')
    dtypes = raw.dtypes.astype(str)
    nulls = raw.isnull().sum()
    uniques = raw.nunique()
    info_df = pd.DataFrame({'dtype': dtypes, 'nulls': nulls, 'unique': uniques})
    st.dataframe(info_df)

    st.subheader('Feature short descriptions')
    feature_desc = {
        'BasketDate': 'Date of the basket / visit (parsed day-first).',
        'BasketID': 'Identifier for each basket / transaction.',
        'CustomerID': 'Customer identifier.',
        'ProdID': 'Product identifier.',
        'Sale': 'Unit sale price (currency).',
        'Qta': 'Quantity in the line (integer).',
        'Amount': 'Line amount = Sale * Qta'
    }
    for col in raw.columns:
        desc = feature_desc.get(col, 'No short description available.')
        st.markdown(f"- **{col}**: {desc}")

# Cleaning & Features tab
with tab_features:
    st.header('Cleaning & Feature Engineering')
    clean = clean_data(raw)
    st.write('After cleaning - rows, columns:', clean.shape)

    st.subheader('Cleaning steps applied')
    st.markdown('- Converted `BasketDate` to datetime\n- Ensured `Sale` and `Qta` are numeric\n- Computed `Amount = Sale * Qta`\n- Removed negative `Qta` and zero `Sale` rows\n- Dropped rows with missing `CustomerID`')

    st.subheader('Customer-level features')
    df_customer = compute_customer_features(clean)
    st.write('Customer-level features shape:', df_customer.shape)
    st.dataframe(df_customer.head().style.format(precision=2))

    st.subheader('Feature descriptions')
    st.markdown(
        """
- **I**: total number of product rows for the customer (proxy for total items purchased across time)
- **Iu**: number of unique products purchased by the customer
- **Imax**: maximum items in a single basket for that customer
- **Entropy**: spending entropy across baskets for the customer (higher -> spending spread across many baskets). Formally: `Entropy = -\u03A3_b p_b log(p_b)` where `p_b` is the proportion of a customer's spend in basket `b` (natural log -> nats).
- **BasketNum**: number of baskets (visits)
- **SumExp**: total expenditure (sum across baskets)
- **AvgExp**: average basket expenditure
"""
    )

    st.subheader('Summary statistics (after outlier removal)')
    new_df = remove_outliers(df_customer)
    st.write(new_df.describe().T)

    st.subheader('Pairwise relationships (sampled)')
    sample_n = min(250, len(new_df))
    pp_sample = new_df.sample(sample_n, random_state=42)
    pp = sns.pairplot(pp_sample[['I','Iu','Imax','Entropy','BasketNum','SumExp','AvgExp']], corner=True, plot_kws={'alpha':0.4,'s':12}, height=1.6)
    st.pyplot(pp.fig)
    st.markdown('**Interpretation:** The pairwise plots show relationships between features; look for strong linear relationships (positive/negative) and potential clusters or outliers.')

    st.subheader('Feature correlation heatmap')
    fig_h, ax_h = plt.subplots(figsize=(5,2.2))
    corr = new_df[['I','Iu','Imax','Entropy','BasketNum','SumExp','AvgExp']].corr()
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), cmap=sns.diverging_palette(220,10,as_cmap=True), annot=True, fmt='.2f', ax=ax_h)
    ax_h.set_title('Feature correlation (new_df)')
    st.pyplot(fig_h)
    st.markdown('**Interpretation:** The heatmap highlights which features move together. For example, `BasketNum` and `SumExp` are often positively correlated, meaning more visits lead to higher total spend.')

# Clustering tab
with tab_clustering:
    st.header('Clustering')
    # Use the same KMeans as the notebook (fixed n_clusters = 4)
    n_clusters = 4
    st.write(f'KMeans (n={n_clusters})')
    kmeans, centers = run_kmeans(new_df, n_clusters=n_clusters)

    st.subheader(f'KMeans (n={n_clusters})')
    fig_k, ax_k = plt.subplots(figsize=(4.5,3))
    ix_I = new_df.columns.tolist().index('I')
    ix_B = new_df.columns.tolist().index('BasketNum')
    centers_xy = centers[:, [ix_I, ix_B]]
    ax_k.scatter(new_df['I'], new_df['BasketNum'], c=kmeans.labels_, s=18, cmap='tab10', alpha=0.6)
    ax_k.scatter(centers_xy[:,0], centers_xy[:,1], s=90, marker='*', c='k')
    ax_k.set_xlabel('I (total items)')
    ax_k.set_ylabel('BasketNum (# baskets)')
    ax_k.set_title('I vs BasketNum colored by KMeans cluster')
    st.pyplot(fig_k)

    # Interpretation for KMeans scatter
    counts = pd.Series(kmeans.labels_).value_counts().sort_index()
    centers_df = pd.DataFrame(centers_xy, columns=['I_center','BasketNum_center'])
    st.markdown('**Interpretation:** The scatter shows clusters differentiated by purchase frequency (`I`) and number of baskets. Cluster centers (stars) show typical customer profiles:')
    for i, row in centers_df.iterrows():
        st.write(f'- Cluster {i}: center I={row.I_center:.1f}, BasketNum={row.BasketNum_center:.1f}, size={int(counts.get(i,0))}')

    st.subheader('3D: I vs BasketNum vs SumExp (KMeans clusters)')
    from mpl_toolkits.mplot3d import Axes3D
    fig3d = plt.figure(figsize=(5,3))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.scatter(new_df['I'], new_df['BasketNum'], new_df['SumExp'], c=kmeans.labels_, marker='o', cmap='tab10', s=18, alpha=0.7)
    ax3d.set_xlabel('I (total items)')
    ax3d.set_ylabel('BasketNum (# baskets)')
    ax3d.set_zlabel('SumExp (total spend)')
    ax3d.set_title('3D: I vs BasketNum vs SumExp (KMeans clusters)')
    st.pyplot(fig3d)

    # Summarize cluster centers and point out high-value cluster
    centers_df_full = pd.DataFrame(centers, columns=new_df.columns)
    top_cluster = int(centers_df_full['SumExp'].idxmax())
    st.markdown(f'- Cluster {top_cluster} has the highest SumExp = {centers_df_full.loc[top_cluster, "SumExp"]:.1f}')

    st.subheader('Cluster sizes')
    fig_c, ax_c = plt.subplots(figsize=(3.5,2.2))
    counts.plot(kind='bar', color='C0', ax=ax_c)
    ax_c.set_xlabel('Cluster')
    ax_c.set_ylabel('Count')
    ax_c.set_title('KMeans cluster sizes')
    st.pyplot(fig_c)
    st.markdown('**Interpretation:** The bar chart shows the relative sizes of clusters; large clusters indicate common customer profiles, small clusters might indicate niche behaviors or outliers.')

    st.subheader('DBSCAN (k-distance plot)')
    dbscan, distances = run_dbscan(new_df)
    fig_d, ax_d = plt.subplots(figsize=(5,2))
    ax_d.plot(distances)
    ax_d.set_title('k-distance plot for DBSCAN')
    ax_d.set_xlabel('Points sorted')
    ax_d.set_ylabel('k-distance')
    st.pyplot(fig_d)
    st.markdown('**Interpretation:** The k-distance plot helps identify an `eps` threshold: look for the elbow where distances rise, which indicates the transition from dense to sparse neighborhoods.')

    # Make counts JSON-serializable: convert numpy types to native Python types and label -1 as 'noise'
    labels, counts = np.unique(dbscan.labels_, return_counts=True)
    counts_dict = {int(l): int(c) for l, c in zip(labels, counts)}
    if -1 in counts_dict:
        counts_dict['noise'] = counts_dict.pop(-1)
    st.write('DBSCAN cluster counts:', counts_dict)

    # Small DBSCAN parameter sweep for diagnostics
    st.subheader('DBSCAN parameter sweep (eps percentiles vs min_samples)')
    grid = run_dbscan_grid(new_df)
    st.dataframe(grid[['eps_percentile','min_samples','n_clusters','noise_count','noise_pct']].sort_values(['eps_percentile','min_samples']))

    # Recommend parameter combinations (prefer >=2 clusters with lowest noise)
    candidates = grid[grid['n_clusters']>=2].sort_values(['noise_pct','n_clusters'], ascending=[True, False]).head(3)
    if not candidates.empty:
        st.markdown('**Recommended DBSCAN parameter combinations** (more clusters, lower noise):')
        st.table(candidates[['eps_percentile','min_samples','n_clusters','noise_pct']])
        st.markdown('**Interpretation:** Prefer parameter sets with lower `noise_pct` and at least 2 clusters; higher `eps` tends to reduce noise but may merge distinct clusters.')
    else:
        st.info('No DBSCAN parameter combo produced at least 2 clusters for the tested grid.')

    # visualize noise percentage heatmap
    pivot = grid.pivot(index='eps_percentile', columns='min_samples', values='noise_pct')
    fig_heat, ax_heat = plt.subplots(figsize=(4,2.2))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='Reds', ax=ax_heat)
    ax_heat.set_title('DBSCAN noise percentage')
    st.pyplot(fig_heat)

    st.subheader('Hierarchical (truncated dendrogram)')
    link = hierarchical_clustering(new_df)
    from scipy.cluster.hierarchy import dendrogram
    fig_dn, ax_dn = plt.subplots(figsize=(6,2))
    dendrogram(link, truncate_mode='lastp', p=30, ax=ax_dn)
    ax_dn.set_title('Truncated Hierarchical Dendrogram')
    st.pyplot(fig_dn)
    st.markdown('**Interpretation:** The dendrogram visualizes hierarchical merges; truncated view highlights groupings of similar customers. Use it to decide plausible numbers of clusters or to detect sub-cluster structure.')

# Classification tab
with tab_classif:
    st.header('Classification')
    X_train, X_test, y_train, y_test = build_classification_dataset(new_df)
    st.write('Train size:', len(X_train), 'Test size:', len(X_test))

    models = fit_quick_models(X_train, y_train)

    st.subheader('Models performance (quick defaults)')
    for name, model in models.items():
        y_pred = model.predict(X_test)
        st.markdown(f'**{name}**')
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(3,2.2))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_title(f'Confusion matrix: {name}')
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        st.pyplot(fig_cm)

        # Automated brief interpretation per model
        labels_unique = sorted(list(set(y_test)))
        p, r, f1, sup = precision_recall_fscore_support(y_test, y_pred, labels=labels_unique)
        best_idx = int(np.argmax(f1))
        worst_idx = int(np.nanargmin(f1))
        st.markdown(f"**Interpretation:** Best F1 class: **{labels_unique[best_idx]}** (F1={f1[best_idx]:.2f}). Worst F1 class: **{labels_unique[worst_idx]}** (F1={f1[worst_idx]:.2f}). This indicates which spend segment the model predicts most/least reliably.")

    # Compute comparative performance summary (test set) and store it for the Interpretation tab (do not display here)
    comp_records = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = float((y_pred == y_test).mean())
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        comp_records.append({'model': name, 'accuracy': float(acc), 'precision_macro': float(p_macro), 'recall_macro': float(r_macro), 'f1_macro': float(f1_macro)})
    comp_df = pd.DataFrame(comp_records).set_index('model')

    # single source of truth: store summary for Interpretation tab
    st.session_state['model_comparison'] = comp_df.to_dict(orient='index')
    st.info('Model comparison stored; see the Interpretation tab for a concise comparison and selection rationale.')

    st.subheader('Random Forest tuning (limited search)')
    with st.spinner('Tuning Random Forest (fast search, this may take a moment)...'):
        tuned_rf, best_params, stats = tune_random_forest(X_train, y_train, n_iter=20, cv=3)
    st.write('Best params:', best_params)

    # cross-validated predictions for more robust metrics
    y_cv_pred = cross_val_predict(tuned_rf, X_train, y_train, cv=5)
    st.text('Cross-validated classification report (train set, cv=5):')
    st.text(classification_report(y_train, y_cv_pred))

    st.subheader('Random Forest feature importances (tuned)')
    if hasattr(tuned_rf, 'feature_importances_'):
        feat_imp = pd.Series(tuned_rf.feature_importances_, index=X_train.columns).sort_values(ascending=True)
        fig_imp, ax_imp = plt.subplots(figsize=(4.2,2.2))
        feat_imp.plot(kind='barh', color='C2', ax=ax_imp)
        ax_imp.set_title('Random Forest feature importances (tuned)')
        ax_imp.set_xlabel('Importance')
        st.pyplot(fig_imp)

    # Permutation importance (model-agnostic) computed on the test set to validate feature relevance
    with st.spinner('Computing permutation importances (test set, n_repeats=10)...'):
        perm = permutation_importance(tuned_rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=True)
    fig_perm, ax_perm = plt.subplots(figsize=(4.2,2.2))
    perm_imp.plot(kind='barh', color='C3', ax=ax_perm)
    ax_perm.set_title('Permutation importances (test set)')
    ax_perm.set_xlabel('Mean drop in score')
    st.pyplot(fig_perm)

    st.markdown('**Note:** Permutation importance measures the change in the model score when a feature is randomly shuffled; it complements internal importance scores and is less biased towards variables with many categories or ranges.')

    # Show more in-depth CV statistics
    st.subheader('Cross-validated scores (tuned RF)')
    cv_res = cross_validate(tuned_rf, X_train, y_train, cv=5, scoring=['accuracy','precision_macro','recall_macro','f1_macro'])
    cv_df = pd.DataFrame({k: [v.mean(), v.std()] for k,v in cv_res.items() if k.startswith('test_')}).T
    cv_df.columns = ['mean','std']
    st.write(cv_df)

    # store summary for the Interpretation tab
    st.session_state['rf_cv_summary'] = {
        'accuracy_mean': float(cv_res['test_accuracy'].mean()),
        'accuracy_std': float(cv_res['test_accuracy'].std()),
        'f1_mean': float(cv_res['test_f1_macro'].mean()),
        'f1_std': float(cv_res['test_f1_macro'].std())
    }

    st.markdown(f"**Interpretation:** Tuned Random Forest cross-validated accuracy = {st.session_state['rf_cv_summary']['accuracy_mean']:.3f} ± {st.session_state['rf_cv_summary']['accuracy_std']:.3f}. The mean F1 (macro) = {st.session_state['rf_cv_summary']['f1_mean']:.3f} ± {st.session_state['rf_cv_summary']['f1_std']:.3f}. This suggests the model has a moderate ability to distinguish spend tertiles; use feature importances to guide targeting.")

# Interpretation tab
with tab_interpret:
    st.header('Interpretation — concise summary')

    st.subheader('Dataset & features')
    st.markdown(
        """
- **Dataset:** transactions aggregated by baskets; cleaned of negative quantities, zero-priced rows and missing `CustomerID`.
- **Key features:** `I`, `Iu`, `Imax`, `BasketNum`, `SumExp`, `AvgExp`, `Entropy` (-Σ_b p_b log p_b; higher = spend more evenly spread across baskets).
"""
    )

    st.subheader('Clustering (brief)')
    st.markdown('- **KMeans:** segments customers by frequency and spend; centers summarize typical profiles.\n- **DBSCAN:** finds dense regions and labels sparse customers as noise (noise % reports dispersion).')

    st.subheader('Results and practical implications')
    if 'model_comparison' in st.session_state:
        comp_df = pd.DataFrame.from_dict(st.session_state['model_comparison'], orient='index')
        comp_df = comp_df[['accuracy','precision_macro','recall_macro','f1_macro']]
        st.dataframe(comp_df.style.format({"accuracy":"{:.3f}","precision_macro":"{:.3f}","recall_macro":"{:.3f}","f1_macro":"{:.3f}"}))
        st.bar_chart(comp_df['f1_macro'])
        best_model = comp_df['f1_macro'].idxmax()
        best_score = comp_df['f1_macro'].max()
        mean_acc = comp_df['accuracy'].mean()
        acc_min, acc_max = float(comp_df['accuracy'].min()), float(comp_df['accuracy'].max())

        st.markdown(f"**Summary:** The best model is **{best_model}** (test macro F1 = **{best_score:.3f}**). The models have a mean accuracy of **{mean_acc:.3f}** (range {acc_min:.3f}–{acc_max:.3f}), which means a typical tertile prediction is correct about **{mean_acc*100:.0f}%** of the time.")

        st.markdown("**What this means in practice**")
        st.markdown("- Use predictions to prioritize actions (for example, target predicted-high spenders for retention or upsell). Treat these predictions as *signals*, not certain labels — validate with experiments (A/B tests) before scaling.")
        st.markdown("- Expect misclassification, especially for customers near the tertile boundaries; consider using predicted probabilities or calibrated thresholds if costs differ across errors.")
        st.markdown("- If your business penalizes false positives more than false negatives (or vice versa), inspect per-class precision/recall and choose the model that minimizes the relevant cost.")

        st.markdown("**Assumptions and recommended next steps**")
        st.markdown("- The analysis assumes past behavior is representative of future behavior; monitor for concept drift and retrain periodically.")
        st.markdown("- Adding recency, category, and time-based features is likely to improve predictions.")
        st.markdown("- If a continuous estimate is needed, consider modeling `SumExp` directly (regression) instead of tertiles.")
    else:
        st.info('Run the Classification tab to generate the model comparison table and view these notes.')

    st.subheader('Final takeaways & actions')
    if 'rf_cv_summary' in st.session_state:
        s = st.session_state['rf_cv_summary']
        st.markdown(f"- Tuned RF (cv=5) mean accuracy: **{s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f}**; mean F1 (macro): **{s['f1_mean']:.3f} ± {s['f1_std']:.3f}**.")
    st.markdown(
        """
- **Actionable:** use predicted tertiles to prioritize customers (e.g., target high predicted spenders), but validate with experiments due to misclassification risk.\n- **Improve:** add recency/category/time features or perform regression on `SumExp` for continuous predictions.\n- **Caveats:** class imbalance and borderline customers (middle tertile) often cause lower precision/recall — consider probability thresholds or calibrated classifiers for production use.
"""
    )

# Code tab
with tab_code:
    st.header('Notebook code & explanations')
    codes = extract_notebook_code()
    for i, entry in enumerate(codes, 1):
        st.subheader(f'Code cell {i}')
        st.code(entry.get('code',''), language='python')
        st.markdown('**Explanation / interpretation**')
        st.write(entry.get('explanation',''))

# End of app
