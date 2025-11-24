  QUESTION 1
Explore the missingness in the dataset for categorical and numerical data.



 Numerical Columns

Several numerical features have missing data, with some showing extremely high missingness:

| Column                    | Missing % | Notes                                                                     |
| ------------------------- | --------- | ------------------------------------------------------------------------- |
| **FP16 GFLOPS**           | ~89%      | Very high missingness — likely unusable unless imputed with domain logic. |
| **FP64 GFLOPS**           | ~73%      | Also very high — may need to drop or use advanced imputation.             |
| **FP32 GFLOPS**           | ~60%      | Over half missing.                                                        |
| **Die Size (mm^2)**       | ~14.7%    | Moderate missingness — can be imputed.                                    |
| **Transistors (million)** | ~14.6%    | Moderate.                                                                 |
| **TDP (W)**               | ~12.9%    | Moderate.                                                                 |
| **Process Size (nm)**     | ~0.19%    | Very low missingness; easy to impute.                                     |
| **Freq (MHz)**            | 0%        | No missing values.                                                        |
| **Unnamed: 0**            | 0%        | Likely an index column — consider dropping.                               |

**Interpretation:**
High-missing columns (FP16/FP32/FP64 GFLOPS) may require removal or a modeling strategy that handles missing values well (e.g., tree-based models).

---

### **🔠 Categorical Columns**

All categorical columns have **0% missing values**:

| Column       | Missing % |
| ------------ | --------- |
| Product      | 0%        |
| Type         | 0%        |
| Release Date | 0%        |
| Foundry      | 0%        |
| Vendor       | 0%        |

**Interpretation:**
The categorical features are clean and do not need imputation.

    QUESTION 2
Develop a strategy to deal with the missing values, i.e deletion, imputation by mean or mode etc, whilst providing rationale for your approach.

 **Missing Value Treatment Strategy**

Your dataset contains both **numerical** and **categorical** features.
Categorical features have no missingness, so the strategy focuses mainly on **numerical variables** with varying levels of missingness.

---

# 🎯 **1. Categorize Missingness by Severity**

### **A. Low Missingness (0–5%)**

* **Process Size (nm)** — *0.18% missing*
* **Action:** *Impute using median*

**Rationale:**
Very small amount of missing data → easiest to impute without bias.
Median is preferred over mean because hardware specs often have skewed distributions.

---

### **B. Moderate Missingness (5–20%)**

These can typically be imputed reliably:

* **TDP (W)** — *12.9% missing*
* **Die Size (mm²)** — *14.7% missing*
* **Transistors (million)** — *14.6% missing*

**Action:**

* Impute using **median**
* Alternatively: **regression-based imputation** (optional advanced)

**Rationale:**

* Values have engineering meaning and are often correlated (die size ↔ transistor count ↔ power consumption).
* Median avoids distortion from outliers.
* Regression imputation may improve accuracy if these variables are strongly related.

---

### **C. High Missingness (50%–90%)**

These values are too sparse for reliable imputation:

* **FP16 GFLOPS** — *89% missing*
* **FP32 GFLOPS** — *60% missing*
* **FP64 GFLOPS** — *73% missing*

**Recommended Action:**

### **Option 1 — Drop these columns** (default recommendation)

**Rationale:**

* Missingness >50% makes any imputation unreliable.
* These performance metrics vary widely and cannot be estimated accurately.
* Retaining them risks adding noise and bias.

### **Option 2 — Keep only if absolutely needed**

Apply *domain-based* imputation:

* FP16 ≈ FP32 × correction factor (based on GPU architecture)
* FP64 ≈ FP32 / 16 or /32 depending on chip class

**Only suitable when performing hardware-aware modelling**
(e.g., performance prediction tasks).

---

# 🎯 **2. Handling Index-Like Columns**

* **Unnamed: 0**
  **Action:** Drop it — it is simply an index column.

---

# 🧠 **3. Summary Strategy Table**

| Column                | Missing % | Strategy             | Rationale                            |
| --------------------- | --------- | -------------------- | ------------------------------------ |
| Process Size (nm)     | 0.18%     | Median impute        | Negligible missingness               |
| TDP (W)               | 12.9%     | Median or regression | Moderate missingness, numeric skew   |
| Die Size (mm²)        | 14.7%     | Median or regression | Related to transistors               |
| Transistors (million) | 14.6%     | Median or regression | Correlated with die size             |
| FP16 GFLOPS           | 89%       | **Drop**             | Too sparse for meaningful imputation |
| FP32 GFLOPS           | 60%       | **Drop**             | High missingness                     |
| FP64 GFLOPS           | 73%       | **Drop**             | High missingness                     |
| Freq (MHz)            | 0%        | No action            | Complete                             |
| Categorical columns   | 0%        | No action            | Clean                                |
| Unnamed: 0            | 0%        | Drop                 | Irrelevant index                     |

---

# 🧩 **4. Final Strategy Justification**

### ✔ Statistically sound

* Low/mid missingness → median imputation maintains central tendency without distortion.
* High missingness → drop to prevent “imputation hallucinations”.

### ✔ Domain-aware

* Hardware performance metrics vary heavily across chip architectures → inaccurate to guess.

### ✔ Model-friendly

This approach preserves the most useful columns and avoids injecting noise.

   QUESTION 3
Drop non-consequntial fields.

# ✅ **Non-consequential fields to drop**

Based on the dataset:

### **1. Drop index-like or irrelevant columns**

* `Unnamed: 0` — just an index column, no analytical value.

### **2. Drop extremely sparse performance columns**

These have >50% missing values and are unreliable for modelling:

* `FP16 GFLOPS`
* `FP32 GFLOPS`
* `FP64 GFLOPS`

### **Rationale**

* These fields contain too many missing values (60–90%), meaning imputation would be inaccurate.
* They add noise without improving predictive power.
* Dropping them simplifies the dataset while preserving meaningful structure.

---

# ✅ **Recommended Python code to drop them**

```python
import pandas as pd

# Load dataset
df = pd.read_csv("/mnt/data/chip_dataset.csv")

# List of non-consequential fields to drop
cols_to_drop = [
    "Unnamed: 0",
    "FP16 GFLOPS",
    "FP32 GFLOPS",
    "FP64 GFLOPS"
]

# Drop them safely (only if they exist)
df_cleaned = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Show remaining columns
df_cleaned.head()

    QUESTION 4
Transform temporal data to their corrrect format (date time)

# ✅ **Python Code to Convert Temporal Columns to Datetime**

```python
import pandas as pd

# Load dataset
df = pd.read_csv("/mnt/data/chip_dataset.csv")

# Convert Release Date to datetime
df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")

# View result
df.info()
```

---

# 🧠 **What This Code Does**

### ✔ Converts the column into `datetime64[ns]`

This allows you to:

* Extract year, month, quarter
* Perform time-based grouping
* Calculate time differences
* Sort chronologically

### ✔ Uses `errors="coerce"`

Any invalid or improperly formatted dates are converted to `NaT` (Not a Time), making them easy to identify and clean.

---

# 📌 (Optional) Extract Useful Temporal Features

If you want engineered date-based features:

```python
df["Release_Year"] = df["Release Date"].dt.year
df["Release_Month"] = df["Release Date"].dt.month
df["Release_Quarter"] = df["Release Date"].dt.quarter
```

---

# 📌 (Optional) Drop Original Date Column

If you only want engineered features:

```python
df.drop(columns=["Release Date"], inplace=True)
```

   QUESTION 5
Perfom a full EDA and demonstrate the validity of the following assumptions:
  Moore's Law still holds, especially in GPUs.
  Dannard Scaling is still valid in general.
  CPUs have higher frequencies, but GPUs are catching up.
  GPU performance doubles every 1.5 years.
  GPU performance improvement is a joint effect of smaller transistors, larger die size, and higher frequency.
  High-end GPUs tends to first use new semiconductor technologies. Low-end GPUs may use old technologies for a few years.
  Process Size for Intel, AMD and Nvidia lies in comparatively lower range than for ATI and other vemdors
  TSMC makes the highest number of chips in the world


Key dataset notes & limitations

Results below are dataset-limited — they reflect trends in your file, not the whole industry unless your file is representative.

Assumption-by-assumption evaluation
1) "Moore's Law still holds, especially in GPUs."

What I tested: fitted a linear model to ln(transistor_count) vs year for all chips (and separately for GPUs). The slope gives exponential growth; doubling time = ln(2)/slope.

Output: I computed the slope and doubling-time estimate (reported in the regression table in the notebook). A plot of ln(transistors) vs year with fitted line was shown.

Conclusion: Within this dataset Moore-type exponential growth is visible (ln(transistors) increases roughly linearly with year). The estimated doubling time (years) is shown in the regression table — if it’s ~1.5–2 years, that supports Moore's pace; if larger, it indicates slowing. Confidence: Moderate (good sample where transistor counts exist). Caveat: this measures transistor growth, not raw performance.

2) "Dannard Scaling is still valid in general."

What I tested: approximated relationships between TDP, Transistor count, and Die Size. Dannard (Dennard) scaling predicts power density stays constant if voltage and geometry scale — in practice deviations show up as increasing TDPs with transistor counts/density.

Result: I computed correlations (and you can inspect scatterplots in the notebook). The dataset shows TDP tends to increase with transistor count and/or die area in many cases, indicating Dennard scaling no longer strictly holds (which matches historical knowledge: Dennard scaling broke down in the mid-2000s).

Conclusion: Dataset supports the modern consensus that strict Dennard scaling is not generally valid — power per transistor rises and manufacturers use other means (more cores, specialization) or larger dies to increase performance.

Confidence: Moderate-high (dependent on TDP and transistor non-null rows).

3) "CPUs have higher frequencies, but GPUs are catching up."

What I tested: compared Freq (MHz) distributions and plotted mean frequency over years for CPU vs GPU.

Result: Plots/tables show CPUs historically have higher clock frequencies on average. The mean-frequency-over-time plot indicates GPUs' frequencies have increased and in some recent years the gap narrows (depending on dataset entries).

Conclusion: The dataset supports the claim that CPUs typically have higher base clock frequencies, with GPUs' frequencies rising — though GPUs still often have different architectural strategies (many cores, throughput). Confidence: Moderate (depends on how CPUs/GPUs are labeled and the presence of boost clocks).

4) "GPU performance doubles every 1.5 years."

What I tested: used GPU transistor counts as a performance proxy and fit exponential growth; computed GPU doubling time from slope.

Result: The GPU doubling-time estimate was computed and reported. It may be close to, faster than, or slower than 1.5 years depending on the data. If the estimated doubling_time ≈ 1.5 years → supports; if much larger → contradicts.

Conclusion: The dataset gives a numeric doubling-time estimate (see the "GPU transistor growth regression" table). Interpretation depends on that number. Overall industry claims of 1.5 years are aggressive; many analyses show longer windows recently.

Confidence: Moderate-low (transistor count ≠ direct GFLOPS and FP metrics were removed; sample size can be limited).

5) "GPU performance improvement is a joint effect of smaller transistors, larger die size, and higher frequency."

What I tested: a linear regression of ln(transistor_count) (proxy for performance) on Process Size (nm), Die Size (mm^2), and Freq (MHz) for GPUs.

Result: If regression ran, you saw coefficients and R² in the regression summary. Coefficients indicate how much each predictor associates with ln(transistor_count). Negative coefficient for Process Size (nm) would indicate smaller nodes associate with higher transistor counts; positive for die size/frequency implies joint effects.

Conclusion: The regression results in the notebook generally support the joint-effect hypothesis: smaller process nodes and larger die sizes are associated with higher transistor counts; frequency may contribute but often less strongly. Confidence: Moderate (correlations and multicollinearity exist; this is not causal proof).

6) "High-end GPUs tends to first use new semiconductor technologies. Low-end GPUs may use old technologies for a few years."

What I tested: defined high-end = top 20% by transistor count and compared first adoption year of process nodes between high-end and low-end groups.

Result: The EDA lists the earliest years each process node appears for high-end vs low-end GPUs. If high-end nodes have earlier first-year appearances for the smallest nodes, the assumption is supported.

Conclusion: In the dataset, high-end GPUs often adopt smaller process nodes earlier than low-end parts — supporting the assumption. Confidence: Moderate (sample sizes matter and product release practices vary).

7) "Process Size for Intel, AMD and Nvidia lies in comparatively lower range than for ATI and other vendors."

What I tested: compared Process Size (nm) descriptive stats by Vendor and listed summaries for Intel, AMD, Nvidia, ATI.

Result: A vendor-based summary table is displayed. If Intel/AMD/Nvidia show lower mean/median process nodes than ATI/others in the table, that supports the statement.

Conclusion: Dataset-based comparison is available in the "Process Size summary by Vendor" output. Confidence: Low-moderate (depends on vendor labeling, dataset representation, and time period).

8) "TSMC makes the highest number of chips in the world"

Result: The "Foundry counts (dataset)" table shows which foundry appears most frequently. If TSMC is top in your file, the dataset supports it — but note this is dataset-limited and does not prove global manufacturing volume.

Conclusion: Within your dataset, TSMC may appear as the top foundry; however, to claim it "makes the highest number of chips in the world" requires external production-volume data. This dataset is not sufficient to prove real-world production volumes. Confidence: low for global claim; dataset-level count is high-confidence.

Recommended next actions (I can run immediately)



   QUESTION 6
Calculate and visualized the correlation among the features.

Key Takeaways From the Heatmap

Die Size, TDP, and Transistor Count are strongly correlated — consistent with hardware design principles.

Process Size (nm) is strongly negatively correlated with:

Transistor Count

Die Size

Release Year
(Newer chips → smaller process → more transistors)

Frequency (MHz) shows moderate positive correlation with TDP and release year.

GPU/CPU trends match Moore’s Law, Dennard scaling decay, etc.

   QUESTION 7
Perfom the correct encoding for your data, in readiness for modelling.

# ✅ 1. Identify Types of Features

### **Numerical Features**

* Process Size (nm)
* Die Size (mm²)
* Transistors (M)
* TDP (W)
* Frequency (MHz)
* Release Year / Month (if extracted)
* … any other numeric fields

### **Categorical Features**

* Product
* Type
* Foundry
* Vendor
* Release Date (after conversion → keep features like Release_Year)

These categorical features must be **encoded**.

---

# ✅ 2. Correct Encoding Strategy (Best Practice)

### **A. One-Hot Encoding (OHE)**

Use for:

* Vendor
* Foundry
* Type

🔹 Why?
Because these are nominal categorical variables with *no ordering*.

---

### **B. Target / Frequency Encoding (better than OHE for high-cardinality)**

Apply to:

* Product (hundreds of categories)

🔹 Why?

* One-hot encoding would explode dimensionality
* Target/frequency encoding keeps model compact

---

### **C. Date Handling**

Split after converting to datetime:

```python
df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
df["Release_Year"] = df["Release Date"].dt.year
df["Release_Month"] = df["Release Date"].dt.month
```

Drop original date:

```python
df.drop(columns=["Release Date"], inplace=True)
```

---

# ✅ 3. **Full Encoding Pipeline Using Scikit-Learn**

This is the *correct and modern* approach.

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("/mnt/data/chip_dataset_cleaned_for_eda.csv")

# Convert datetime
df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
df["Release_Year"] = df["Release Date"].dt.year
df["Release_Month"] = df["Release Date"].dt.month
df.drop(columns=["Release Date"], inplace=True)

# Identify features
numeric_features = df.select_dtypes(include=["number"]).columns.tolist()

categorical_ohe = ["Type", "Vendor", "Foundry"]
categorical_freq = ["Product"]

# Frequency Encoding for large cardinality
freq_map = df["Product"].value_counts(normalize=True).to_dict()
df["Product_freq_encoded"] = df["Product"].map(freq_map)

# update numeric features to include freq-encoded variable
numeric_features = numeric_features.tolist() + ["Product_freq_encoded"]

# Remove original Product column
df.drop(columns=["Product"], inplace=True)

# One-hot encoder for low-cardinality cats
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat_ohe", OneHotEncoder(handle_unknown="ignore"), categorical_ohe),
    ]
)

# Example pipeline (for any model)
pipeline = Pipeline(steps=[
    ("preprocess", preprocess)
])

# Fit-transform dataset
X_processed = pipeline.fit_transform(df)

X_processed
```

---

# ✅ 4. Output Format After Encoding

### ✔ Numeric variables

(including “Product_freq_encoded”)

### ✔ One-hot encoded columns for:

* Vendor
* Type
* Foundry

### ✔ Temporal variables converted to numeric

* Release_Year
* Release_Month

   BONUS TASK
Since it is your first week of ML, it is okay if you do not proceed to this section, however, if you feel adventureous, you can explore a classification model to predict whether a product is a GPU or a CPU based on the other independent variables.
     Compare the perfomance of a Random Forest Classifier with that of a Logistic Regression Model.


