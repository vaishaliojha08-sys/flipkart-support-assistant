import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer 
from sklearn.dummy import DummyClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.inspection import permutation_importance 
from sklearn.impute import SimpleImputer 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import ( accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, ) 
from sklearn.model_selection import ( GridSearchCV, StratifiedKFold, train_test_split, ) 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------

df = pd.read_csv("orders_dataset.csv")

print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# 2. Separate features (X) and target (y)
# ---------------------------------------------------------

# 'returned' is the value we want our model to predict.
y = df["returned"]

# 'order_id' is only an identifier, so we don't use it as a
# machine-learning feature.
X = df.drop(columns=["returned", "order_id"])


# ---------------------------------------------------------
# 3. Define the categorical and numeric features
# ---------------------------------------------------------

categorical_features = [
    "product_category",
    "payment_method",
]

numeric_features = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]


# ---------------------------------------------------------
# 4. Create the numeric preprocessing pipeline
# ---------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        # Replace missing numeric values with the median
        ("imputer", SimpleImputer(strategy="median")),

        # Standardize numeric features
        ("scaler", StandardScaler()),
    ]
)


# ---------------------------------------------------------
# 5. Create the categorical preprocessing pipeline
# ---------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        # Replace missing categorical values with the most
        # frequently occurring category (mode)
        ("imputer", SimpleImputer(strategy="most_frequent")),

        # Convert categories into one-hot encoded columns
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# 6. Combine both pipelines using ColumnTransformer
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ---------------------------------------------------------
# 7. Create the required 80/20 stratified split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)


# ---------------------------------------------------------
# 8. Display the split sizes
# ---------------------------------------------------------

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTest data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ---------------------------------------------------------
# 10. Transform both training and test data
# ---------------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# ---------------------------------------------------------
# 11. Check the transformed data
# ---------------------------------------------------------

print("\nProcessed training data shape:")
print(X_train_processed.shape)

print("\nProcessed test data shape:")
print(X_test_processed.shape)

# ------------------------------------------------------------
# STEP 23 - DUMMY CLASSIFIER BASELINE 
# ------------------------------------------------------------
print("\n" + "=" * 70) 
print("STEP 23 - DUMMY CLASSIFIER BASELINE") 
print("=" * 70) 

dummy_pipeline = Pipeline( 
    steps=[ 
        ( "preprocessor", 
         preprocessor,
           ), 
           ( "classifier", 
            DummyClassifier( 
                strategy="most_frequent"
            ), 
        ), 
    ] 
) 

# Train 
dummy_pipeline.fit( X_train, y_train, ) 

# Predict 
y_pred_dummy = dummy_pipeline.predict( X_test ) 

# Metrics 
dummy_accuracy = accuracy_score( y_test, y_pred_dummy, ) 
dummy_f1 = f1_score( y_test, y_pred_dummy, pos_label=1, zero_division=0, ) 
print( "\nDummyClassifier Accuracy:", round(dummy_accuracy, 4), ) 
print( "DummyClassifier F1 (returned=1):", round(dummy_f1, 4), ) 
print("\nInterpretation:") 
print( "The DummyClassifier predicts the majority class for every order. " 
      "Because returned orders are the minority class, it can achieve " 
      "apparently reasonable accuracy while identifying zero returned orders. " 
      "This is the high-accuracy, zero-recall trap. Therefore, accuracy alone " 
      "is misleading and the model must be compared with a baseline and " 
      "evaluated using business-relevant metrics such as recall, precision " 
      "and F1-score for returned orders." )

# ------------------------------------------------------------
# STEP 24 - LOGISTIC REGRESSION 
# ------------------------------------------------------------
print("\n" + "=" * 70) 
print("STEP 24 - LOGISTIC REGRESSION") 
print("=" * 70) 

logistic_pipeline = Pipeline( 
    steps=[ 
        ( "preprocessor", 
         preprocessor, 
         ), 
         ( "classifier", 
          LogisticRegression( 
              class_weight="balanced", 
              max_iter=1000, random_state=42, 
              ), 
        ), 
    ] 
) 
# Train Logistic Regression 
logistic_pipeline.fit( X_train, y_train, ) 

# Default 
threshold = 0.5 
y_pred_lr = logistic_pipeline.predict( 
    X_test ) 

# Probability of returned = 1 
y_prob_lr = logistic_pipeline.predict_proba( 
    X_test 
    )[:, 1] 
lr_accuracy = accuracy_score( y_test, y_pred_lr, ) 
lr_f1 = f1_score( y_test, y_pred_lr, pos_label=1, zero_division=0, ) 
lr_recall = recall_score( y_test, y_pred_lr, pos_label=1, zero_division=0, ) 
lr_precision = precision_score( y_test, y_pred_lr, pos_label=1, zero_division=0, ) 
lr_roc_auc = roc_auc_score( y_test, y_prob_lr, ) 
print("\nLogistic Regression - threshold 0.5") 
print("--------------------------------------") 
print( "Accuracy :", round(lr_accuracy, 4), ) 
print( "F1 :", round(lr_f1, 4), ) 
print( "Recall :", round(lr_recall, 4), ) 
print( "Precision:", round(lr_precision, 4), ) 
print( "ROC-AUC :", round(lr_roc_auc, 4), )

# ------------------------------------------------------------
# STEP 25 - LOGISTIC REGRESSION THRESHOLD SWEEP 
# ------------------------------------------------------------
print("\n" + "=" * 70) 
print("STEP 25 - LOGISTIC REGRESSION THRESHOLD SWEEP") 
print("=" * 70) 
# Thresholds from 0.10 to 0.90. 
# Step size = 0.01, which satisfies the assignment. 
thresholds = np.arange( 0.10, 0.901, 0.01, ) 
threshold_results = [] 
for threshold in thresholds: 
    y_pred_threshold = ( y_prob_lr >= threshold ).astype(int) 
    threshold_f1 = f1_score( y_test, y_pred_threshold, pos_label=1, zero_division=0, ) 
    threshold_recall = recall_score( y_test, y_pred_threshold, pos_label=1, zero_division=0, ) 
    threshold_precision = precision_score( y_test, y_pred_threshold, pos_label=1, zero_division=0, ) 
    threshold_results.append( 
        { "threshold": threshold, 
         "f1": threshold_f1, 
         "recall": threshold_recall, 
         "precision": threshold_precision, 
         } 
         ) 
    

threshold_df = pd.DataFrame( threshold_results ) 

# Find threshold with maximum F1 
best_lr_row = threshold_df.loc[ threshold_df["f1"].idxmax() ] 
best_lr_threshold = float( best_lr_row["threshold"] ) 
best_lr_f1 = float( best_lr_row["f1"] ) 
best_lr_recall = float( best_lr_row["recall"] ) 
best_lr_precision = float( best_lr_row["precision"] ) 
print("\nBest Logistic Regression threshold:") 
print( "Threshold :", round(best_lr_threshold, 2), ) 
print( "F1 :", round(best_lr_f1, 4), ) 
print( "Recall :", round(best_lr_recall, 4), ) 
print( "Precision :", round(best_lr_precision, 4), )


# ------------------------------------------------------------
#  Compare default recall with best threshold recall 
# ------------------------------------------------------------ 
recall_improvement = ( best_lr_recall - lr_recall ) 
precision_change = ( best_lr_precision - lr_precision ) 
print( "\nRecall improvement over 0.5:", round(recall_improvement, 4), ) 
print( "Precision change from 0.5:", round(precision_change, 4), ) 
print("\nThreshold results:") 
print( 
    threshold_df.to_string( 
        index=False, 
        formatters={ 
            "threshold": "{:.2f}".format, 
            "f1": "{:.4f}".format, 
            "recall": "{:.4f}".format, 
            "precision": "{:.4f}".format, 
        }, 
    ) 
) 
print("\nBusiness trade-off:") 
print( "Lowering the decision threshold makes the model more willing to " 
      "flag an order as likely to be returned. This increases recall and " 
      "reduces false negatives, meaning fewer genuinely returned orders " 
      "are missed. However, it also decreases precision because more " 
      "orders that will not actually be returned are flagged. The business " 
      "therefore accepts more false positives in exchange for avoiding " 
      "more costly false negatives." )


# ============================================================ 
# STEP 26 - RANDOM FOREST + GRID SEARCH 
# ============================================================ 
print("\n" + "=" * 70) 
print("STEP 26 - RANDOM FOREST GRID SEARCH") 
print("=" * 70) 
rf_pipeline = Pipeline( 
     steps=[ 
          ( "preprocessor", preprocessor, 
           ), 
           ( "classifier", 
            RandomForestClassifier( 
                 class_weight="balanced", 
                 random_state=42, 
                 n_jobs=-1, 
            ), 
        ), 
    ] 
) 
#Required GridSearch parameter combinations 
param_grid = { "classifier__n_estimators": 
              [ 100, 
               200, ], 
               "classifier__max_depth": 
               [ 6, 10, None, 
            ], 
        } 
# 5-fold Stratified CV 
cv = StratifiedKFold( n_splits=5, shuffle=True, random_state=42, ) 
grid_search = GridSearchCV( 
     estimator=rf_pipeline, 
     param_grid=param_grid, 
     scoring="roc_auc", 
     cv=cv, 
     n_jobs=-1, 
     verbose=1, 
     ) 
print("\nStarting GridSearchCV...") 
print("6 parameter combinations x 5 folds") 
grid_search.fit( X_train, y_train, ) 
print("\nBest parameters:") 
print( grid_search.best_params_ ) 
print( "\nBest cross-validated ROC-AUC:", round(grid_search.best_score_, 4), ) 
# Winning pipeline 
best_rf_pipeline = ( grid_search.best_estimator_ )


# ============================================================ 
# STEP 27 - RANDOM FOREST TEST EVALUATION 
# ============================================================ 
# Predict probability for returned = 1 
y_prob_rf = best_rf_pipeline.predict_proba( 
    X_test 
)[:, 1] 
# Default threshold = 0.5 
y_pred_rf = ( 
    y_prob_rf >= 0.50 
    ).astype(int) 

rf_accuracy = accuracy_score( y_test, y_pred_rf, ) 
rf_f1 = f1_score( y_test, y_pred_rf, pos_label=1, zero_division=0, ) 
rf_recall = recall_score( y_test, y_pred_rf, pos_label=1, zero_division=0, ) 
rf_precision = precision_score( y_test, y_pred_rf, pos_label=1, zero_division=0, ) 
rf_roc_auc = roc_auc_score( y_test, y_prob_rf, ) 
cv_auc = grid_search.best_score_ 
auc_difference = abs( cv_auc - rf_roc_auc ) 
print("\nWinning Random Forest parameters:") 
print( grid_search.best_params_ ) 
print( "\nBest CV ROC-AUC:", round(cv_auc, 4), ) 
print( "Test ROC-AUC:", round(rf_roc_auc, 4), ) 
print( "CV/Test ROC-AUC difference:", round(auc_difference, 4), ) 
print( "Test Accuracy:", round(rf_accuracy, 4), ) 
print( "Test F1:", round(rf_f1, 4), ) 
print( "Test Recall:", round(rf_recall, 4), ) 
print( "Test Precision:", round(rf_precision, 4), )


# ============================================================ 
# STEP 28 - RANDOM FOREST FEATURE IMPORTANCE 
# ============================================================

# Get fitted preprocessing component 
fitted_preprocessor = ( 
    best_rf_pipeline.named_steps[ 
        "preprocessor" 
        ] 
    ) 
# Get fitted Random Forest 
fitted_rf = ( 
    best_rf_pipeline.named_steps[ 
        "classifier" 
        ] 
    ) 
# Feature names after preprocessing 
feature_names = ( fitted_preprocessor 
                 .get_feature_names_out() 
                 ) 

# Random Forest impurity-based feature importance 
importances = ( 
    fitted_rf.feature_importances_ 
    ) 

feature_importance_df = pd.DataFrame( 
    { 
        "feature": feature_names, 
        "importance": importances, 
        } 
    ).sort_values( 
        "importance", ascending=False, 
        ) 

print("\nTop 10 features:") 
print( feature_importance_df .head(10) .to_string(index=False) ) 
print("\nTop 5 features:") 

top5_features = ( 
    feature_importance_df 
    .head(5) 
    .copy() 
    ) 

print( top5_features.to_string( index=False ) )

# ============================================================ 
# STEP 29 - PERMUTATION IMPORTANCE 
# ============================================================

# Use the fitted preprocessor to transform test data. 
X_test_processed = ( 
    fitted_preprocessor.transform( 
        X_test 
        ) 
    ) 
permutation_result = permutation_importance( 
    fitted_rf, 
    X_test_processed, 
    y_test, 
    scoring="roc_auc", 
    n_repeats=10, 
    random_state=42, 
    n_jobs=-1, 
) 
permutation_df = pd.DataFrame( 
    { 
        "feature": feature_names, 
        "permutation_importance": permutation_result.importances_mean, 
        } 
    ).sort_values( "permutation_importance", ascending=False, 
            )
print( "\nTop 10 permutation-importance features:" ) 
print( permutation_df .head(10) .to_string(index=False) 
      )


# ------------------------------------------------------------ 
# Compare original top 5 with permutation importance 
# ------------------------------------------------------------ 
top5_names = ( 
    top5_features["feature"] 
    .tolist() 
) 
comparison_df = top5_features.merge( 
    permutation_df, 
    on="feature", 
    how="left", 
    ) 

comparison_df = ( 
    comparison_df 
    .sort_values( 
        "importance", 
        ascending=False, 
        ) 
    ) 

print( "\nTop-5 feature importance comparison:" ) 
print( comparison_df.to_string( index=False ) ) 
print( "\nInterpretation:" ) 
print( 
      "Impurity-based Random Forest importance can overrate a noisy " 
      "continuous feature because continuous variables offer many possible " 
      "split points, giving the feature more opportunities to appear useful " 
      "even when its actual predictive contribution on unseen data is small." 
      )

# ============================================================ 
# STEP 30 - SUBGROUP ANALYSIS 
# ============================================================

# Create test-set results table 
test_results = X_test.copy() 
test_results["actual_returned"] = ( y_test.values ) 
test_results["predicted_returned"] = ( y_pred_rf )


# ------------------------------------------------------------ 
# Product category 
# ------------------------------------------------------------ 
category_results = [] 

for category in sorted( 
    test_results[ 
        "product_category" 
        ].unique() 
    ): 
    subset = test_results[ 
        test_results[ 
            "product_category" 
            ] == category 
            ] 
    
    category_recall = recall_score( 
        subset["actual_returned"], 
        subset["predicted_returned"], 
        zero_division=0, 
        ) 
    
    category_precision = precision_score( 
        subset["actual_returned"], 
        subset["predicted_returned"], 
        zero_division=0, 
        ) 
    
    category_results.append( 
        { 
            "product_category": category, 
            "count": len(subset), 
            "recall": category_recall, 
            "precision": category_precision, 
            } 
        ) 
    
    category_df = pd.DataFrame( 
        category_results 
        ) 
    
    print( "\nPerformance by product category:" ) 
    print( 
        category_df.to_string( 
            index=False, 
            formatters={ 
                "recall": "{:.4f}".format, 
                "precision": "{:.4f}".format, 
                }, 
            ) 
        )
    

# ------------------------------------------------------------ 
# Payment method 
# ------------------------------------------------------------ 
payment_results = [] 

for payment_method in sorted( 
    test_results[ 
        "payment_method" 
        ].unique() 
    ): 
    subset = test_results[ 
        test_results[ 
            "payment_method" 
            ] == payment_method 
            ] 
    
    payment_recall = recall_score( 
        subset["actual_returned"], 
        subset["predicted_returned"], 
        zero_division=0, 
        ) 
    
    payment_precision = precision_score( 
        subset["actual_returned"], 
        subset["predicted_returned"], 
        zero_division=0, 
        ) 
    
    payment_results.append( 
        { 
            "payment_method": payment_method, 
            "count": len(subset), 
            "recall": payment_recall, 
            "precision": payment_precision, 
            } 
            ) 
    
    payment_df = pd.DataFrame( 
        payment_results 
        ) 
    print( "\nPerformance by payment method:" ) 
    print( 
        payment_df.to_string( 
            index=False, 
            formatters={ 
                "recall": "{:.4f}".format, 
                "precision": "{:.4f}".format, 
                }, 
            ) 
        )
    

# ------------------------------------------------------------ 
# Overall test-set metrics for comparison 
# ------------------------------------------------------------ 
print("\nOverall test-set performance:") 
print( "Recall :", 
      round(rf_recall, 4), 
      ) 
print( "Precision:", 
      round(rf_precision, 4), 
      ) 
# Find weakest recall subgroup 
weakest_category = category_df.loc[ 
    category_df["recall"].idxmin() 
    ] 

weakest_payment = payment_df.loc[ 
    payment_df["recall"].idxmin() 
    ] 

print( "\nWeakest product-category subgroup:" ) 
print( weakest_category.to_string() ) 
print( "\nWeakest payment-method subgroup:" ) 
print( weakest_payment.to_string() ) 
print( "\nPossible concrete improvement:" ) 
print( 
      "A category-specific decision threshold could be tested for the " 
      "weakest subgroup. For example, if one product category has materially " 
      "lower recall, lowering that category's risk threshold would increase " 
      "the number of potentially returned orders flagged for proactive " 
      "support, at the cost of additional false positives." 
      )


# ============================================================ 
# STEP 31 - RANDOM FOREST THRESHOLD SWEEP 
# ============================================================

rf_threshold_results = [] 

for threshold in thresholds: 
    y_pred_threshold_rf = ( 
        y_prob_rf >= threshold 
        ).astype(int) 
    threshold_f1 = f1_score( 
        y_test, 
        y_pred_threshold_rf, 
        pos_label=1, 
        zero_division=0, 
        ) 
    threshold_recall = recall_score( 
        y_test, 
        y_pred_threshold_rf, 
        pos_label=1, 
        zero_division=0, 
        ) 
    threshold_precision = precision_score( 
        y_test, 
        y_pred_threshold_rf, 
        pos_label=1, 
        zero_division=0, 
        ) 
    rf_threshold_results.append( 
        { 
            "threshold": threshold, 
            "f1": threshold_f1, 
            "recall": threshold_recall, 
            "precision": threshold_precision, 
            } 
        ) 
    
    
rf_threshold_df = pd.DataFrame( 
    rf_threshold_results 
    ) 

# Find F1-maximising Random Forest threshold 
best_rf_row = rf_threshold_df.loc[ 
    rf_threshold_df["f1"].idxmax() 
    ] 
t_rf = float( 
    best_rf_row["threshold"] 
    ) 
rf_best_f1 = float( 
    best_rf_row["f1"] 
    ) 
rf_best_recall = float( 
    best_rf_row["recall"] 
    ) 
rf_best_precision = float( 
    best_rf_row["precision"] 
    ) 
print( "\nRandom Forest F1-maximising threshold:" ) 
print( "t*_rf :", round(t_rf, 2), ) 
print( "F1 :", round(rf_best_f1, 4), ) 
print( "Recall :", round(rf_best_recall, 4), ) 
print( "Precision :", round(rf_best_precision, 4), ) 
print( "\nRandom Forest threshold results:" ) 
print( rf_threshold_df.to_string( 
    index=False, 
    formatters={ 
        "threshold": "{:.2f}".format, 
        "f1": "{:.4f}".format, 
        "recall": "{:.4f}".format, 
        "precision": "{:.4f}".format, 
        }, 
    ) 
)


# ============================================================ 
# STEP 32 - SAVE FINAL RANDOM FOREST PIPELINE 
# ============================================================

# Create models directory 
os.makedirs( 
    "models", 
    exist_ok=True, 
    )

model_path = ( 
    "models/return_risk_model.pkl" 
    ) 
joblib.dump( best_rf_pipeline, model_path, )

# Save t*_rf separately. 
# Part 3 will use this value for: 
# probability < t*_rf -> Low 
# t*_rf <= probability < t*_rf + 0.15 -> Medium 
# probability >= t*_rf + 0.15 -> High 
 
threshold_path = ( "models/return_risk_threshold.txt" )

with open( 
    threshold_path, 
    "w", 
    encoding="utf-8", 
    ) as f: 
    f.write( 
        str(t_rf) 
        )
    

print( "\nModel saved to:", model_path, ) 
print( "t*_rf saved to:", threshold_path, ) 
print( "\nt*_rf =", round(t_rf, 2), ) 
print( "\nPart 3 bucket cut points:" ) 
print( "Low : probability <", round(t_rf, 2), ) 
print( 
    "Medium :", 
    round(t_rf, 2), 
    "<= probability <", 
    round(t_rf + 0.15, 2), 
    ) 
print( 
    "High : probability >=", round(t_rf + 0.15, 2), 
      )


# ============================================================ 
# STEP 33 - VERIFY SAVED MODEL 
# ============================================================

# ------------------------------------------------------------ 
# Load the saved pipeline 
# ------------------------------------------------------------ 
 
loaded_model = joblib.load( "models/return_risk_model.pkl" )

# ------------------------------------------------------------ 
# Load saved Random Forest threshold 
# ------------------------------------------------------------ 


with open( 
    "models/return_risk_threshold.txt", 
    "r", 
    encoding="utf-8", 
    ) as f: 
    loaded_threshold = float( 
        f.read().strip() 
        )
    
# ------------------------------------------------------------ 
# Predict using the SAVED model 
# ------------------------------------------------------------ 
loaded_probabilities = ( 
    loaded_model.predict_proba( 
        X_test 
        )[:, 1] 
    )

# ------------------------------------------------------------ 
# Display first 5 predictions 
# ------------------------------------------------------------ 

print( "\nFirst 5 probabilities from saved model:" ) 
for probability in ( 
    loaded_probabilities[:5] 
    ): 
    print( round( probability, 
                 6, 
    ) 
)
    
# ------------------------------------------------------------ 
# Verify probability range 
# ------------------------------------------------------------

assert np.all( 
    ( 
        loaded_probabilities >= 0 
        ) 
        & 
        ( 
            loaded_probabilities <= 1 
        ) 
    )

# ------------------------------------------------------------ 
# Verify threshold 
# ------------------------------------------------------------

assert ( 
    0.10 
    <= loaded_threshold 
    <= 0.90 
    )

# ------------------------------------------------------------ 
# Verify model file exists 
# ------------------------------------------------------------

assert os.path.exists( 
    "models/return_risk_model.pkl" 
    )


# ------------------------------------------------------------ 
# Verify the loaded model produces the same 
# probabilities as the original model 
# ------------------------------------------------------------

original_probabilities = ( 
    best_rf_pipeline.predict_proba( 
        X_test 
        )[:, 1] 
        )

probabilities_match = np.allclose( 
    original_probabilities, 
    loaded_probabilities, 
    )

print( 
    "\nLoaded t*_rf:", 
    round( 
        loaded_threshold, 
        2, 
    ), 
)

print( 
    "Saved model probabilities match " "original model:", 
    probabilities_match, 
    ) 

assert probabilities_match

# ------------------------------------------------------------ 
# Final verification 
# ------------------------------------------------------------

print("\n" + "=" * 70) 
print("PART 1 MODEL VERIFICATION SUCCESSFUL") 
print("=" * 70) 
print( "\nFinal model:", "Tuned Random Forest Pipeline", ) 
print( "Model artifact:", "models/return_risk_model.pkl", ) 
print( "t*_rf:", round( loaded_threshold, 2, ), ) 
print( "\nThe saved model is ready to be consumed by Part 3." ) 
print("\n" + "=" * 70) 
print("END OF PART 1 TRAINING SCRIPT") 
print("=" * 70)