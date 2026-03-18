# ==============================================================================
# 08_CANN.py — Combined Actuarial Neural Network
# ==============================================================================
# Fits neural networks that boost GLM predictions by learning residual patterns.
# Compares CANN built on GLM1 (benchmark) vs CANN built on GLM9 (optimised).
#
# RUN: python 08_CANN.py   (from your project directory in RStudio terminal)
#
# REQUIRES: learn_for_python.csv and test_for_python.csv
#           (generated from R — see instructions)
#
# OUTPUTS:  cann_results.csv        (comparison table)
#           cann_training_history.csv (training curves)
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

print("TensorFlow version:", tf.__version__)
print("Loading data...")

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

learn = pd.read_csv("learn_for_python.csv")
test = pd.read_csv("test_for_python.csv")

print(f"  Learn: {learn.shape[0]} rows")
print(f"  Test:  {test.shape[0]} rows")

# ==============================================================================
# 2. PREPARE FEATURES
# ==============================================================================
# Use the same raw features as the paper's neural network.
# Categorical variables get label-encoded then embedded (or one-hot encoded).
# Continuous variables get standardised.

continuous_vars = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
categorical_vars = ['Area', 'VehBrand', 'VehGas', 'Region']

# Log-transform Density (same as GLM)
learn['LogDensity'] = np.log(learn['Density'].clip(lower=1))
test['LogDensity'] = np.log(test['Density'].clip(lower=1))

continuous_features = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'LogDensity']

# Standardise continuous variables
scaler = StandardScaler()
X_learn_cont = scaler.fit_transform(learn[continuous_features].values)
X_test_cont = scaler.transform(test[continuous_features].values)

# Label-encode categorical variables and one-hot encode
label_encoders = {}
X_learn_cat_list = []
X_test_cat_list = []

for var in categorical_vars:
    le = LabelEncoder()
    le.fit(pd.concat([learn[var], test[var]]).astype(str))
    label_encoders[var] = le
    
    learn_encoded = le.transform(learn[var].astype(str))
    test_encoded = le.transform(test[var].astype(str))
    
    # One-hot encode
    n_classes = len(le.classes_)
    learn_onehot = np.eye(n_classes)[learn_encoded]
    test_onehot = np.eye(n_classes)[test_encoded]
    
    X_learn_cat_list.append(learn_onehot)
    X_test_cat_list.append(test_onehot)

X_learn_cat = np.hstack(X_learn_cat_list)
X_test_cat = np.hstack(X_test_cat_list)

# Combine continuous and categorical
X_learn_all = np.hstack([X_learn_cont, X_learn_cat])
X_test_all = np.hstack([X_test_cont, X_test_cat])

print(f"  Feature matrix: {X_learn_all.shape[1]} columns")

# Response and exposure
y_learn = learn['ClaimNb'].values
y_test = test['ClaimNb'].values
exposure_learn = learn['Exposure'].values
exposure_test = test['Exposure'].values

# GLM predictions (log scale) for use as CANN offset
log_glm1_learn = np.log(learn['glm1_pred'].values.clip(min=1e-10))
log_glm1_test = np.log(test['glm1_pred'].values.clip(min=1e-10))
log_glm9_learn = np.log(learn['glm9_pred'].values.clip(min=1e-10))
log_glm9_test = np.log(test['glm9_pred'].values.clip(min=1e-10))

# ==============================================================================
# 3. POISSON DEVIANCE LOSS (same metric as R scripts)
# ==============================================================================

def poisson_deviance(y_true, y_pred, exposure):
    """Average Poisson deviance in units of 10^-2 (matching R function)."""
    mu = y_pred  # predicted rate
    dev = np.where(
        y_true == 0,
        2 * mu * exposure,
        2 * y_true * np.log(y_true / (mu * exposure + 1e-10)) + 
        2 * (mu * exposure - y_true)
    )
    return np.mean(dev) * 100

# ==============================================================================
# 4. BUILD CANN MODEL
# ==============================================================================

def build_cann(n_features, glm_offset_learn, glm_offset_test, 
               y_learn, exposure_learn, y_test, exposure_test,
               X_learn, X_test, model_name,
               hidden_units=[64, 32], dropout_rate=0.2,
               epochs=50, batch_size=5000, learning_rate=0.001):
    """
    Build and train a CANN model.
    
    The network learns a correction term on top of the GLM's log-prediction:
        log(mu) = log(GLM_pred) + log(Exposure) + NN(features)
    
    This is equivalent to: mu = GLM_pred * Exposure * exp(NN(features))
    """
    
    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"{'='*60}")
    
    # Input layers
    feature_input = keras.Input(shape=(n_features,), name='features')
    offset_input = keras.Input(shape=(1,), name='offset')
    
    # Neural network correction term
    x = feature_input
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation='relu', name=f'hidden_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
    
    # Single output: the log-correction
    nn_output = layers.Dense(1, activation='linear', name='nn_correction')(x)
    
    # Combine: log(mu) = offset + nn_correction
    # offset = log(GLM_pred) + log(Exposure)
    combined = layers.Add(name='combined')([offset_input, nn_output])
    
    # Exponentiate to get mu (predicted count)
    output = layers.Activation('exponential', name='predicted_count')(combined)
    
    model = keras.Model(inputs=[feature_input, offset_input], outputs=output)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='poisson'
    )
    
    model.summary()
    
    # Prepare offsets: log(GLM_pred) + log(Exposure)
    offset_train = (glm_offset_learn + np.log(exposure_learn)).reshape(-1, 1)
    offset_test_arr = (glm_offset_test + np.log(exposure_test)).reshape(-1, 1)
    
    # Early stopping to prevent overfitting
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    # Train
    history = model.fit(
        [X_learn, offset_train], y_learn,
        validation_data=([X_test, offset_test_arr], y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Predict rates
    pred_learn = model.predict([X_learn, offset_train], verbose=0).flatten() / exposure_learn
    pred_test = model.predict([X_test, offset_test_arr], verbose=0).flatten() / exposure_test
    
    # Evaluate
    in_dev = poisson_deviance(y_learn, pred_learn, exposure_learn)
    out_dev = poisson_deviance(y_test, pred_test, exposure_test)
    gap = out_dev - in_dev
    
    print(f"\n  Results for {model_name}:")
    print(f"    In-sample deviance:  {in_dev:.5f}")
    print(f"    Out-of-sample deviance: {out_dev:.5f}")
    print(f"    Gap: {gap:.5f}")
    
    return {
        'model_name': model_name,
        'in_sample': round(in_dev, 5),
        'out_sample': round(out_dev, 5),
        'gap': round(gap, 5),
        'epochs_trained': len(history.history['loss']),
        'history': history.history
    }

# ==============================================================================
# 5. FIT BOTH CANN MODELS
# ==============================================================================

n_features = X_learn_all.shape[1]

# Set seed for reproducibility
tf.random.set_seed(100)
np.random.seed(100)

# CANN on GLM1 (benchmark — what the paper does)
result_glm1 = build_cann(
    n_features, log_glm1_learn, log_glm1_test,
    y_learn, exposure_learn, y_test, exposure_test,
    X_learn_all, X_test_all,
    model_name="CANN (GLM1 base)",
    hidden_units=[64, 32],
    dropout_rate=0.2,
    epochs=50,
    batch_size=5000
)

# Reset seed
tf.random.set_seed(100)
np.random.seed(100)

# CANN on GLM9 (your optimised GLM)
result_glm9 = build_cann(
    n_features, log_glm9_learn, log_glm9_test,
    y_learn, exposure_learn, y_test, exposure_test,
    X_learn_all, X_test_all,
    model_name="CANN (GLM9 base)",
    hidden_units=[64, 32],
    dropout_rate=0.2,
    epochs=50,
    batch_size=5000
)

# ==============================================================================
# 6. ALSO FIT A STANDALONE NEURAL NETWORK (no GLM offset)
# ==============================================================================
# This replicates the paper's plain NN for comparison.

tf.random.set_seed(100)
np.random.seed(100)

# For the standalone NN, the offset is just log(Exposure)
log_exposure_learn = np.zeros(len(exposure_learn))  # no GLM offset
log_exposure_test = np.zeros(len(exposure_test))

result_nn = build_cann(
    n_features, log_exposure_learn, log_exposure_test,
    y_learn, exposure_learn, y_test, exposure_test,
    X_learn_all, X_test_all,
    model_name="Standalone NN",
    hidden_units=[64, 32],
    dropout_rate=0.2,
    epochs=50,
    batch_size=5000
)

# ==============================================================================
# 7. SAVE RESULTS
# ==============================================================================

results_df = pd.DataFrame([
    {
        'Model': r['model_name'],
        'InSample': r['in_sample'],
        'OutOfSample': r['out_sample'],
        'Gap': r['gap'],
        'Epochs': r['epochs_trained']
    }
    for r in [result_glm1, result_glm9, result_nn]
])

# Add reference rows for GLM1 and GLM9 (no NN)
glm_refs = pd.DataFrame([
    {'Model': 'GLM1 (no NN)', 
     'InSample': poisson_deviance(y_learn, learn['glm1_pred'].values, exposure_learn),
     'OutOfSample': poisson_deviance(y_test, test['glm1_pred'].values, exposure_test),
     'Gap': None, 'Epochs': None},
    {'Model': 'GLM9 (no NN)', 
     'InSample': poisson_deviance(y_learn, learn['glm9_pred'].values, exposure_learn),
     'OutOfSample': poisson_deviance(y_test, test['glm9_pred'].values, exposure_test),
     'Gap': None, 'Epochs': None}
])

results_df = pd.concat([results_df, glm_refs], ignore_index=True)
results_df = results_df.sort_values('OutOfSample')

print("\n" + "="*60)
print("  FINAL COMPARISON")
print("="*60)
print(results_df.to_string(index=False))

results_df.to_csv("cann_results.csv", index=False)

# Save training histories
for r in [result_glm1, result_glm9, result_nn]:
    name = r['model_name'].replace(' ', '_').replace('(', '').replace(')', '')
    hist_df = pd.DataFrame({
        'epoch': range(1, len(r['history']['loss']) + 1),
        'train_loss': r['history']['loss'],
        'val_loss': r['history']['val_loss']
    })
    hist_df.to_csv(f"cann_history_{name}.csv", index=False)

print("\nResults saved to: cann_results.csv")
print("Training histories saved to: cann_history_*.csv")
print("Done.")
