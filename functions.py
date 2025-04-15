import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

# Metrics

def click_rank_utility(df, y_pred):
    """
    Implémente la fonction Click-Rank Utility U(f).
    """
    df_test = df.loc[y_pred.index].copy()
    df_test = df_test.assign(pred=y_pred)

    # Evite le biais de position
    df_test = df_test[df_test['displayrandom'] == 1].copy()

    # Trier les produits affichés dans chaque impression par score de prédiction croissant
    df_test = df_test.sort_values(by=['impression_id', 'pred'], ascending=[True, True])  

    # Calcul du rank
    df_test['rank_order'] = df_test.groupby('impression_id')['pred'].rank(method='min', ascending=True)

    # I(Y_D = 1) * rank_J f(X_J, D)
    df_test['utility'] = df_test['click'] * df_test['rank_order']

    # Moyenne de U(f) sur chaque impression’
    click_rank_per_impression = df_test.groupby('impression_id')['utility'].mean()

    # Moyenne finale sur toutes les bannières
    return click_rank_per_impression.mean()

def nllh(y_true, y_pred):
    y_pred_np = np.array(y_pred) + 1e-15
    y_true_np = np.array(y_true)
    
    epsilon = 1e-15

    y_pred_clip = np.clip(y_pred_np, epsilon, 1 - epsilon)

    log_loss_np = y_true_np * np.log(y_pred_clip) + (1 - y_true_np) * np.log(1 - y_pred_clip)
    return -np.mean(log_loss_np)

def DP(df, y_pred : pd.Series):

    df_test = df.loc[y_pred.index].copy()
    df_test['y_pred'] = y_pred

    df_senior = df_test [df_test['senior'] == 1]
    df_senior_male = df_senior[df_senior['protected_attribute'] == 1]
    df_senior_female = df_senior[df_senior['protected_attribute'] == 0]
    return np.mean(df_senior_male['y_pred']) - np.mean(df_senior_female['y_pred'])

def AUC(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    return auc(fpr, tpr)

def report_results(df, y_pred_prob, y_pred):

    if not isinstance(y_pred, pd.Series):
        raise ValueError("y_pred must be a pandas Series")
    
    df_test = df.loc[y_pred.index].copy()
    y_true = df_test['click']

    click_rank_value = click_rank_utility(df, y_pred)
    nllh_value = nllh(y_true, y_pred_prob)
    auc_value = AUC(y_true, y_pred_prob)
    dp_value = DP(df, y_pred)

    return {
    'Click Rank Utility': round(click_rank_value, 5),
    'Negative Log-Likelihood': round(nllh_value, 5),
    'AUC': round(auc_value, 5),
    'Demographic Parity': round(dp_value, 5)
}
