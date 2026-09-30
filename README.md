# Thesis Projects

A collection of machine learning notebooks from collaborative thesis work, organized by contributor. Each top-level folder holds one project's notebook(s) and its datasets. All notebooks are Python 3 and were run locally (paths in the code refer to local dataset locations).

The projects cover five problem areas: autism spectrum disorder (ASD) screening classification for toddlers, maize yield prediction from agronomic and climate data, automatic HS-code (tariff classification) assignment from product descriptions, reinforcement-learning-based inventory reorder optimization, and firearm detection in images.

## Contents

| Folder | Project | Technique |
| ------ | ------- | --------- |
| `Edith/` | ASD trait prediction in toddlers | Random Forest, KNN (with hyperparameter tuning) |
| `Faith/` | HS-code classification from product descriptions | Doc2Vec + Logistic Regression |
| `Michelle/` | Inventory reorder optimization | Q-learning (reinforcement learning) vs EOQ |
| `Shumi/` | Maize yield prediction | Linear Regression, Random Forest, XGBoost, ANN |
| `Taps/` | Firearm detection in images | Custom CNN, AlexNet, VGG16, InceptionV3 |

---

## Edith - ASD Trait Prediction in Toddlers

Notebooks: `ASD_Model (1).ipynb` (initial analysis) and `ASD_Mo Optimize.ipynb` (refined, tuned version). Data: `ASD_Toddler_dataset.csv`.

**Objective.** Predict ASD traits in toddlers (`Class_ASD_Traits`) from Q-Chat screening responses and demographic information.

**Data.** 3,998 toddler records with 17 columns: ten Q-Chat screening item answers (`BF1`-`BF10`), the total `QchatBF1-10Score`, `Age_Mons`, `Gender`, `Jaundice`, `Family_mem_with_ASD`, `Province` (Zimbabwean provinces), and the target `Class_ASD_Traits`. No missing values; about 67.4% of toddlers are ASD positive.

**Methods.** Exploratory data analysis (class balance, gender/jaundice/age/province breakdowns, correlation analysis - the highest correlation with the screening score is about 0.35, feature-importance analysis identifies BF3 as the most important item and BF7 the least). Preprocessing includes encoding, scaling with `StandardScaler`, and an 80/20 train-test split. Models: Random Forest (500/1500 trees) and KNN (k = 13), followed by `GridSearchCV` hyperparameter tuning of both models in the optimize notebook.

**Results.** The initial notebook reports a Random Forest confusion matrix with 229 correct ASD predictions versus 27 misclassifications, against 214/42 for KNN, and its notes record ROC AUC of about 0.57 (Random Forest) and 0.52 (KNN) at that stage. The optimize notebook's comparison:

| Model | Accuracy |
| ----- | -------- |
| Original Random Forest | 0.9638 |
| Tuned Random Forest | 0.9663 |
| Original KNN | 0.7875 |
| Tuned KNN | 0.9525 |

Tuning lifts KNN from 78.8% to 95.3% accuracy; the tuned Random Forest is the best model at 96.6%.

## Faith - HS-Code Classification with Doc2Vec

Notebook: `faith Doc2Vec Model.ipynb`. Data: `train_data.csv`, `test_dataA.csv`.

**Objective.** Automatically assign Harmonized System (HS) tariff codes to product descriptions - a many-class text classification problem for customs/trade data.

**Data.** Product records with a text description column (`DESC`) and an `HSCODE` label covering dozens of 10-digit HS codes.

**Methods.** Tokenizes descriptions, trains a gensim Doc2Vec model (vector size 100, 100 epochs, min count 2) over `TaggedDocument`s, then trains a Logistic Regression classifier on the document vectors, with `GridSearchCV` tuning and 5-fold cross-validation. The trained classifier is exported with joblib to `classifier_model.pkl` and exercised on unseen descriptions.

**Results.** About 84% accuracy and 0.84 F1 score on the held-out data; per-code precision/recall varies from roughly 0.6 to 1.0 across the HS codes (reported via `classification_report` and a confusion matrix).

## Michelle - Inventory Reorder Optimization with Q-Learning

Notebooks: `Working Code Backup - Copy.ipynb` and `Working Code Backup - Copy.ipynb2.ipynb` (near-duplicates; the second adds the EOQ comparison and stockout evaluation). Data: `product_data.csv`.

**Objective.** Learn reorder quantities per product that minimize total inventory cost (holding plus production/ordering cost), and compare the learned policy against the classic Economic Order Quantity (EOQ) heuristic.

**Data.** 4,986 records across grocery products (e.g. sauces and soups, instant porridge, water) with `product_name`, `sales`, `price`, `reorder_quantity`, `stock_level`, `holding_cost`, `Production_cost`, `currency_fluctuation`, `stockout`, and `timestamp`. Split 80/20 into train/test.

**Methods.** Defines an `InventoryEnv` simulation where the state is (product, stock level), the action is a reorder amount, and the reward penalizes holding and production costs. Trains a tabular Q-learning agent (alpha 0.01, gamma 0.995, epsilon-greedy with decay, 1,000 episodes, Q-table of 59 possible actions), then evaluates the learned policy on the test set, visualizes reward convergence, and benchmarks against EOQ (`sqrt(2*D*O/H)`) and a do-nothing baseline. Reports total cost before/after, EOQ cost, MAE of the new stock level against sales, stockout precision/recall, and service level.

**Results.** Total test-set cost falls from 2,030,122.36 (baseline) to 753,392.22 under the learned policy, beating the EOQ policy at 781,594.79. Stockout prediction recall is 1.00 with precision 0.34; 53 of 998 test rows end at zero stock.

## Shumi - Maize Yield Prediction

Notebook: `analysis.ipynb`. Data: `analysistrue1.csv` (main), plus supporting files (`analysis.csv`, `analysistrue.csv`, `rainfall.csv`, `prec.csv`, `climate maybe.csv`, `maize dataset cimmyt.xlsx`, processed/unique-record exports).

**Objective.** Predict maize yield from agronomic and climate variables in CIMMYT maize trial data from Zimbabwe.

**Data.** Trial records with fields including `country` (Zimbabwe), `year`, `location` (e.g. Chiredzi), `region`, `yield` (target), `anthesisDate`, `Asi`, `plant_height`, `ear_position`, `stem_lodging`, `Epp`, `Ear_rot`, alongside rainfall and other climate variables.

**Methods.** EDA on stations, average yield per sector, distributions and boxplots; encoding of categorical features, `MinMaxScaler` scaling, and a 70/30 train-test split. Trains Linear Regression, Random Forest, XGBoost, and a Keras ANN (64-32 dense layers with dropout and early stopping), evaluates with MSE/MAE/R-squared, runs `GridSearchCV` for XGBoost and Random Forest, tunes the ANN architecture via a KerasRegressor grid search, and compares feature importances across models.

**Results.** Base models: Linear Regression R2 0.687, Random Forest 0.804, XGBoost 0.819, ANN 0.820. After tuning, XGBoost reaches R2 0.828 (RMSE 0.088) and Random Forest 0.807 (RMSE 0.093), making tuned XGBoost the best yield predictor.

## Taps - Firearm Detection in Images

Notebook: `mine.ipynb`. Data: `Firearms_Detection_Dataset/` (Train/Validation/Test image folders, included in this repository).

**Objective.** Binary image classification to detect whether a scene is "Weaponized" (firearm present) or "Peaceful".

**Data.** Two-class image dataset: 5,291 training and 605 validation images in one configuration (4,234/1,057 with a 40-image test set in another). Images resized to 224x224.

**Methods.** Keras `ImageDataGenerator` pipelines with normalization and augmentation; builds and compares four models - a custom CNN, a from-scratch AlexNet-style architecture, and transfer-learning classifiers on frozen VGG16 and InceptionV3 backbones with new dense heads. Evaluates each with confusion matrices, ROC curves, precision/recall/F1, and t-SNE visualization of learned feature maps.

**Results.** Training logs in the notebook reach about 100% validation accuracy with near-zero validation loss on this dataset (for example, the first trained configuration reports val_accuracy 1.0000, val_precision 1.0000, val_recall 1.0000), indicating the models separate the two classes well on the provided data.

---

## Repository Layout

```
.
├── Edith/
│   ├── ASD_Model (1).ipynb
│   ├── ASD_Mo Optimize.ipynb
│   └── ASD_Toddler_dataset.csv
├── Faith/
│   ├── faith Doc2Vec Model.ipynb
│   ├── train_data.csv
│   └── test_dataA.csv
├── Michelle/
│   ├── Working Code Backup - Copy.ipynb
│   ├── Working Code Backup - Copy.ipynb2.ipynb
│   └── product_data.csv
├── Shumi/
│   ├── analysis.ipynb
│   └── ... (CSV/XLSX climate and maize datasets)
├── Taps/
│   ├── mine.ipynb
│   └── Firearms_Detection_Dataset/  (Train/Validation/Test images)
└── README.md
```

## Reproducing

The notebooks were written for local execution (some cell paths point to local Windows folders, e.g. `Taps/mine.ipynb`). To run a notebook, place its sibling dataset file(s) in the same working directory, or adjust the path constants at the top of the notebook.

Main dependencies: Python 3, pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, TensorFlow/Keras (with scikeras), gensim, wordcloud, joblib.
