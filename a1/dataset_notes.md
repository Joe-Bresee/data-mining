# SENG 474 A1 — Dataset Notes & Findings

## Dataset Overview
- **6821 rows**, 13 columns (11 float, 2 int)
- Likely wine quality dataset (red + white wine combined via `class` column)
- Target variables: `quality` (discrete 3–8), `class` (binary 0/1)

---

## Missing Data
- **alcohol**: 18.87% NaN values
- **Decision: median imputation** (median = 10.3)
  - Median chosen over mean because outliers hadn't been removed yet — median is more robust
  - 18.87% is under the ~20% acceptable threshold for imputation
  - Side effect: spike visible at 10.3 in alcohol histogram — note this in report

---

## Gaussian Noise Columns
Identified by IQR flagging >10% of rows as outliers at 3×IQR threshold:
- **volatile acidity**: 1699 flagged (~25% of data)
- **class**: 1677 flagged (~25% of data)

**volatile acidity** — confirmed noise beyond doubt:
  - Near-zero correlation with every other feature (max ~0.04)
  - Distribution goes **negative** in histogram — physically impossible for wine acidity
  - **Decision: drop this column entirely** (signal completely destroyed)

**class** — noisy but still informative:
  - Strong correlations despite noise (chlorides: -0.718, total sulfur dioxide: 0.697)
  - **Decision: keep, do not apply IQR to it**

---

## Outlier Removal
- Applied **3×IQR fencing** (not 1.5× — assignment says "synthetic extreme outliers", 3× targets only extreme points)
- Skipped `volatile acidity` and `class` columns in IQR loop (noise columns, not true outliers)
- **~8.6% of rows removed** after IQR cleaning
- Remaining rows: ~6234

---

## Correlations (Notable)
**Redundant feature pairs:**
- `free sulfur dioxide` / `total sulfur dioxide`: **0.719** — strong overlap
- `density` / `alcohol`: **-0.643**
- `density` / `residual sugar`: **0.551**
- `density` is essentially a derived measurement of alcohol + residual sugar
- **Decision: consider dropping `density` and/or `free sulfur dioxide`** — justify in report

**Target variable (quality) correlations:**
- `alcohol`: 0.395 — strongest predictor by far
- `density`: -0.316 — second strongest
- Most other features have very weak correlation with quality

---

## Distribution Analysis (Histograms)
**Right-skewed — apply log transform:**
- `residual sugar` — heavily skewed, long right tail
- `chlorides` — skewed with secondary bump
- `free sulfur dioxide` — right skewed
- `total sulfur dioxide` — right skewed
- `sulphates` — right skewed
- **Use `np.log1p()`** (handles near-zero values safely)

**Roughly normal — no transform needed:**
- `fixed acidity`, `pH`, `citric acid`, `alcohol`

---

## Class Imbalance
- `class` is binary and **imbalanced**: ~1200 zeros vs ~4700 ones
- Important to note for classification model performance — accuracy alone will be misleading
- Consider using F1 score or balanced accuracy for classification evaluation

## Quality is Discrete
- `quality` ranges from 3–8 — can be treated as:
  - **Regression target** (predict numeric score) 
  - **Classification target** (predict quality bucket)
- Decide early and be consistent — affects which models you apply it to

---

## Recommended Final Preprocessing Order
1. Median impute `alcohol`
2. Identify noise columns via IQR >10% threshold
3. Remove extreme outliers with 3×IQR, skipping noise columns
4. Drop `volatile acidity` (signal destroyed)
5. Consider dropping `density` (redundant with alcohol + residual sugar)
6. Log transform skewed columns (`residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `sulphates`)
7. Scale with **StandardScaler** on clean data (after outlier removal)
   - MinMaxScaler avoided — sensitive to remaining outliers
8. Split into train/val/test before fitting scaler — fit scaler on train only, transform val/test

---

## Report Justification Checklist
- [ ] Why median over mean for imputation
- [ ] Why 3×IQR over 1.5×IQR
- [ ] How Gaussian noise columns were identified (IQR >10% flag)
- [ ] Why volatile acidity was dropped (negative values, near-zero correlations)
- [ ] Why log transforms were applied (right skew)
- [ ] Why StandardScaler over MinMaxScaler
- [ ] Note alcohol imputation spike at 10.3
- [ ] Note class imbalance for classification evaluation
