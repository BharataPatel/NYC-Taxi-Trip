# NYC Taxi Trip Analysis & Fare Prediction

A machine learning project that explores NYC Yellow Taxi trips and serves total trip amount predictions through a Streamlit app. The project covers data cleaning, feature engineering, regression, high-tip classification, trip clustering, dimensionality reduction, and saving trained models for inference.

The interactive app accepts trip details and uses a saved Random Forest regressor to estimate **`total_amount` in US dollars**.

## Contents

- [Project features](#project-features)
- [Dataset](#dataset)
- [Project files](#project-files)
- [Technology stack](#technology-stack)
- [Installation and setup](#installation-and-setup)
- [Run the prediction app](#run-the-prediction-app)
- [App inputs](#app-inputs)
- [Analysis and modeling workflow](#analysis-and-modeling-workflow)
- [Recorded results](#recorded-results)
- [Saved models and Python inference](#saved-models-and-python-inference)
- [Reproducibility and limitations](#reproducibility-and-limitations)
- [Troubleshooting](#troubleshooting)
- [Possible improvements](#possible-improvements)

## Project features

- **Interactive prediction:** Enter distance, duration, passenger count, pickup time, taxi zones, and rate code in Streamlit.
- **Data exploration:** Inspect missing values, duplicates, distributions, outliers, and correlations.
- **Regression:** Compare Linear Regression, Decision Tree, Random Forest, and Histogram Gradient Boosting.
- **Preprocessing pipeline:** Explore one-hot encoding of pickup and drop-off zones with a Random Forest estimator.
- **Model evaluation:** Measure MAE, RMSE, and R²; inspect residuals and errors by fare group, pickup hour, and rate code.
- **Model selection:** Apply cross-validation, randomized hyperparameter search, and feature importance analysis.
- **High-tip classification:** Compare Logistic Regression and Random Forest, including threshold analysis and ROC/precision-recall curves.
- **Unsupervised learning:** Explore trip segments using K-Means and visualize standardized trip features with PCA.
- **Model persistence:** Export models with Joblib and compare predictions before and after loading.

## Dataset

The downloader uses **January 2024 NYC Yellow Taxi trip records** from the [NYC Taxi and Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

| Detail | Value |
| --- | --- |
| Source file | `yellow_tripdata_2024-01.parquet` |
| Local CSV | `NYC_Taxi_2024.csv` |
| Raw shape recorded in the notebook | 2,964,624 trips and 19 columns |
| Period | January 2024; the local filename does not represent a full year |
| Prediction target | `total_amount` |
| Local CSV size | Approximately 314 MB |

The records include pickup/drop-off timestamps and zones, trip distance, passenger count, rate code, payment type, fares, tips, taxes, tolls, and surcharges. TLC defines `total_amount` as the amount charged to the passenger, excluding cash tips. Distance is recorded in miles. See the [Yellow Taxi data dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf) for field definitions.

To regenerate the CSV:

```bash
python dataset_downloader.py
```

The script downloads the [January 2024 Parquet file](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet), reads it with pandas, and writes `NYC_Taxi_2024.csv` in the current directory. Internet access and a Parquet reader such as PyArrow are required. The download is needed for notebook analysis and retraining; the app uses the saved model directly.

## Project files

```text
.
|-- app.py                       # Streamlit prediction interface
|-- dataset_downloader.py        # Download January 2024 data and create CSV
|-- main.ipynb                   # Data preparation, experiments, evaluation, exports
|-- main2.ipynb                  # Load saved models and try pipeline inference
|-- nyc_taxi_random_forest.pkl    # Nine-feature model loaded by app.py
|-- nyc_taxi_pipeline.pkl         # Separate preprocessing + Random Forest pipeline
|-- NYC_Taxi_2024.csv             # Generated local dataset
|-- .gitignore
`-- README.md
```

The CSV can be regenerated with the downloader instead of being stored with the source code. To run the app from a fresh checkout, ensure `nyc_taxi_random_forest.pkl` is available beside `app.py`, or regenerate it from the training notebook.

## Technology stack

| Tool | Purpose |
| --- | --- |
| Python 3.11 | Project runtime |
| Streamlit | Interactive prediction app |
| pandas and NumPy | Data preparation and numerical operations |
| scikit-learn | Preprocessing, training, evaluation, clustering, and PCA |
| Matplotlib and Seaborn | Exploratory and diagnostic plots |
| Joblib | Saving and loading fitted models |
| Requests and PyArrow | Downloading and reading the source data |
| Jupyter / VS Code notebooks | Interactive analysis |

## Installation and setup

Download or clone the repository, then open a terminal in its root directory.

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install app dependencies

```bash
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy joblib scikit-learn==1.9.0
```

The saved models were checked successfully with Python 3.11, scikit-learn 1.9.0, pandas 3.0.4, NumPy 2.4.6, and Joblib 1.5.3; the local Streamlit version was 1.66.1. These are the observed environment versions, not a dependency lockfile. Use a compatible training environment when loading serialized scikit-learn models.

### 3. Install analysis dependencies if needed

```bash
python -m pip install matplotlib seaborn requests pyarrow jupyter ipykernel
```

## Run the prediction app

From the directory containing `app.py` and `nyc_taxi_random_forest.pkl`, run:

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

1. Enter the trip's distance, duration, and passenger count.
2. Set the pickup hour, day of week, and weekend flag.
3. Supply pickup/drop-off taxi zone IDs and the rate code.
4. Click **Predict Fare**.
5. Read the **Predicted Total Amount** displayed in dollars.

The app rejects nonpositive distances and durations. Its passenger input starts at 1, and pickup hour/day inputs are bounded to their valid ranges. The app currently provides regression inference; classification, clustering, and PCA are explored in the notebook.

## App inputs

The app constructs a one-row pandas DataFrame with the following columns, in this order:

| Feature | Meaning / expected input | Example |
| --- | --- | --- |
| `trip_distance` | Trip distance in miles; greater than zero | `5.2` |
| `trip_duration_minutes` | Trip duration in minutes; greater than zero | `22.5` |
| `passenger_count` | Number of passengers; integer of at least 1 | `1` |
| `pickup_hour` | Pickup hour from 0 to 23 | `18` |
| `pickup_dayofweek` | Monday = 0 through Sunday = 6 | `4` |
| `is_weekend` | Weekday = 0; Saturday/Sunday = 1 | `0` |
| `PULocationID` | Pickup TLC taxi zone ID | `132` |
| `DOLocationID` | Drop-off TLC taxi zone ID | `161` |
| `RatecodeID` | Trip rate code | `1` |

Keep `is_weekend` consistent with the day of week; the current app asks for it separately. Location IDs identify taxi zones rather than latitude/longitude coordinates. The [TLC taxi zone lookup table](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv) maps IDs to zone names.

Common rate codes include 1 for standard rate, 2 for JFK, 3 for Newark, 4 for Nassau/Westchester, 5 for negotiated fare, and 6 for group ride. See the [official data dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf) for the complete definitions.

## Analysis and modeling workflow

### 1. Inspect and clean the data

`main.ipynb` inspects schema, missingness, duplicates, descriptive statistics, and extreme observations. It parses pickup/drop-off timestamps, drops rows with missing values, and restricts both timestamps to January 2024.

Further filters retain positive distance, fare, and total amount; trip durations between 1 and 120 minutes; and average speeds greater than zero and no more than 80 mph.

**Current training scope:** An active early filter requires `tolls_amount > 0`. The following `tolls_amount >= 0` filter does not restore trips already removed. The saved notebook outputs show **198,602 cleaned trips**, so the current analysis represents a subset of trips with tolls. Broader taxi coverage requires changing the cleaning policy and retraining.

### 2. Engineer features

- `trip_duration_minutes`: Difference between drop-off and pickup timestamps, in minutes.
- `average_speed_mph`: Distance divided by duration in hours, used for cleaning.
- `pickup_hour`: Hour extracted from the pickup timestamp.
- `pickup_dayofweek`: Day index extracted from the pickup timestamp.
- `is_weekend`: Whether the pickup day is Saturday or Sunday.
- `pickup_date`: Calendar date used for exploration.

The initial regression experiments use eight features. Later experiments add `RatecodeID`, producing the nine-feature schema used by the app.

### 3. Train and compare regression models

The notebook uses an 80/20 train/test split with `random_state=42`. It compares Linear Regression, Decision Tree, Random Forest, and `HistGradientBoostingRegressor`, plus a Random Forest pipeline with one-hot encoded pickup/drop-off zones.

The app's exported Random Forest is configured with **100 trees**, **`max_depth=12`**, **`random_state=42`**, and **`n_jobs=-1`**.

Additional experiments include five-fold cross-validation, permutation importance, a reduced feature set, and `RandomizedSearchCV` with 15 candidates and three-fold validation. The search explores tree count, depth, minimum split size, and minimum leaf size. Its fitted estimator is separate from the Random Forest exported for the app.

Sampling steps use `min(requested_size, len(data))` to handle datasets smaller than their requested sample sizes while preserving feature/target index alignment.

### 4. Analyze prediction errors

The notebook compares actual and predicted amounts, examines residual distributions, inspects the largest absolute errors, and groups errors by fare range, pickup hour, and rate code. It also compares training/test performance and evaluates feature importance.

### 5. Predict high-tip trips

Classification focuses on credit-card trips (`payment_type == 1`). After an initial fare-based exploration, the later target definition uses:

```text
pre_tip_amount = total_amount - tip_amount
tip_percentage = 100 * tip_amount / pre_tip_amount
high_tip = 1 if tip_percentage >= 20 else 0
```

Trips with nonpositive pre-tip amounts are excluded. Logistic Regression uses a StandardScaler pipeline; Random Forest provides the second classifier. The split is stratified, and evaluation includes accuracy, precision, recall, F1, confusion matrices, threshold comparisons, ROC-AUC, average precision, and permutation importance.

### 6. Explore trip segments and PCA

K-Means uses standardized distance, duration, and `fare_amount`. The notebook explores the elbow method, silhouette scores, and two- and three-cluster solutions, then examines cluster means, medians, and centers in the original units.

PCA projects these three standardized features into two components and visualizes trips both with and without cluster coloring.

## Recorded results

The tables below use the **saved outputs of the notebook's evaluation cells**. They were not regenerated for this README. The manually entered tables and narrative in the notebook's final findings contain different numbers, so they should be refreshed after a consistent rerun.

### Regression

MAE and RMSE are in US dollars; lower is better. Higher R² indicates a better fit on the evaluated split.

| Experiment | MAE ($) | RMSE ($) | R² |
| --- | ---: | ---: | ---: |
| Linear Regression, eight features | 9.5493 | 15.2950 | 0.6571 |
| Decision Tree, eight features | 7.0463 | 12.0175 | 0.7883 |
| Random Forest, eight features | 6.7702 | 10.9855 | 0.8231 |
| Random Forest + one-hot zone encoding | 6.6875 | 10.7384 | 0.8369 |
| **Random Forest + RatecodeID, nine features** | **5.8577** | **8.7324** | **0.8882** |
| Histogram Gradient Boosting, 100 iterations | 5.8336 | 8.9230 | 0.8833 |
| Histogram Gradient Boosting, 300 iterations | 5.8216 | 8.8937 | 0.8841 |
| Random Forest, five-feature subset | 5.8706 | 8.8178 | 0.8860 |

The Linear Regression RMSE above is calculated from its saved MSE of `233.9364215599125`. These experiments use different feature sets and, in some cases, sampling, so they are exploratory comparisons. The nine-feature Random Forest has the highest recorded R² in this table; the 300-iteration gradient boosting experiment has the lowest MAE.

The initial Random Forest's five-fold mean R² is **0.8367**. The randomized search records a best cross-validation R² of **0.8407**, with 100 trees, depth 16, minimum split size 10, and minimum leaf size 1.

### High-tip classification

Saved metrics at the default 0.5 decision threshold:

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6359 | 0.5916 | 0.1644 | 0.2573 |
| Random Forest | 0.6913 | 0.6219 | 0.4983 | 0.5533 |

The Random Forest records **ROC-AUC = 0.7030** and **average precision = 0.5835**. Its threshold experiment illustrates the tradeoff between identifying more high-tip trips and increasing precision.

### Clustering and PCA

The saved silhouette experiment records scores of **0.5351 for k=2** and **0.5470 for k=3**, with lower scores for k=4 through k=6. The notebook also explores a two-cluster interpretation; this is separate from selecting the highest silhouette score in that experiment.

The saved PCA output explains **79.29%** of variance with PC1 and **16.66%** with PC2, for **95.95% combined**. These values describe the recorded subset and preprocessing.

## Saved models and Python inference

| Artifact | Contents | Usage |
| --- | --- | --- |
| `nyc_taxi_random_forest.pkl` | Fitted Random Forest with nine input features, including `RatecodeID` | Loaded by `app.py` |
| `nyc_taxi_pipeline.pkl` | ColumnTransformer with zone encoding followed by Random Forest; fitted with eight original features | Explored in `main2.ipynb` |

The artifacts have different input schemas and predictions. The pipeline's fitted schema does not include `RatecodeID`.

To predict directly in Python, run this from the project root:

```python
import joblib
import pandas as pd

model = joblib.load("nyc_taxi_random_forest.pkl")

new_trip = pd.DataFrame({
    "trip_distance": [5.2],
    "trip_duration_minutes": [22.5],
    "passenger_count": [1],
    "pickup_hour": [18],
    "pickup_dayofweek": [4],
    "is_weekend": [0],
    "PULocationID": [132],
    "DOLocationID": [161],
    "RatecodeID": [1],
})

prediction = model.predict(new_trip)[0]
print(f"Predicted total amount: ${prediction:.2f}")
```

The current saved Random Forest produces **`$44.86`** for this example; this prediction was checked directly while preparing the README. Retraining can change the result. The separate saved pipeline produces approximately `$46.58` for the corresponding notebook example.

To regenerate the app model, run the data preparation and nine-feature Random Forest training section in `main.ipynb`, then export that fitted estimator:

```python
joblib.dump(rf_model, "nyc_taxi_random_forest.pkl")
```

The notebook also exports `rf_pipeline` to `nyc_taxi_pipeline.pkl` and checks that predictions from the Random Forest match before and after serialization.

## Reproducibility and limitations

- **Notebook execution:** Open `main.ipynb` in VS Code or start Jupyter with `python -m jupyter notebook`. Run prerequisite data preparation and training cells before their dependents. The notebook contains exploratory reuse of variables; in the clustering section, early `cluster_df` inspection cells appear before its assignment and must be run after creating it. It is not currently a clean, automated training script.
- **Result provenance:** Saved outputs, manually entered result tables, and final narrative are not fully synchronized. Rerun the relevant evaluation cells before presenting a newly trained model's metrics.
- **Training coverage:** January 2024 data and the current positive-toll filter limit coverage. Results should not be assumed to represent all NYC taxi trips or current prices.
- **Duration availability:** Actual trip duration is known after a trip. For a prediction made before travel, the user must supply an estimated duration, which adds uncertainty beyond the notebook evaluation.
- **Target interpretation:** The app estimates the historical `total_amount` field, which includes recorded charges and electronic tips. It does not separately calculate fare components or predict cash tips.
- **Input consistency:** The current app accepts numeric zone/rate IDs and a separate weekend flag. It does not validate every ID against a lookup table or derive the weekend flag automatically.
- **Evaluation design:** A random train/test split evaluates the recorded subset; there is no separate chronological holdout in the current workflow.
- **Compute requirements:** The source CSV has nearly three million rows. Full preprocessing, cross-validation, randomized search, and permutation importance can use substantial memory and CPU time.

## Troubleshooting

| Issue | What to check |
| --- | --- |
| Model file not found | Run from the project root and ensure `nyc_taxi_random_forest.pkl` exists there. Regenerate the export if needed. |
| Missing Python module | Install the relevant dependency using `python -m pip` in the environment running Streamlit or the notebook kernel. |
| Streamlit command not recognized | Use `python -m streamlit run app.py`. |
| Missing CSV | Run `python dataset_downloader.py` before loading the dataset in the notebook. |
| Parquet reader error | Install `pyarrow` in the environment running the downloader. |
| Cannot take a larger sample than population | Use `n=min(requested_size, len(data))`; the notebook's sampling cells already use this pattern. |
| Feature names/count mismatch | Use the nine app features for the Random Forest, preserving names/order; inspect `model.feature_names_in_` when working with another artifact. |
| Notebook variable not defined | Run the cells that create that variable first; move the clustering inspections after the `cluster_df` assignment when rerunning. |
| Serialized model version mismatch | Match the environment used for training, or retrain and export in the environment used for inference. |

## Possible improvements

- Extract a repeatable preprocessing/training script and generate result tables from computed metrics.
- Revisit the toll filter, train on broader trip coverage, and evaluate on later months.
- Add a dependency lockfile and model metadata recording feature schema, training scope, and evaluation results.
- Derive the weekend flag from the selected day and replace numeric zone inputs with searchable zone names.
- Cache model loading and batch related Streamlit inputs in a form.
- Add prediction intervals, clearer input bounds, and separate evaluation for estimated trip duration.

Data attribution: [NYC Taxi and Limousine Commission trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
