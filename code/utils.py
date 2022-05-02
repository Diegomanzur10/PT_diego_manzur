import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr

# A continuación se definen las métricas de ajuste
def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def media_e2(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    mean_actual = np.mean(actual)
    mean_pred = np.mean(pred)
    sd_actual = np.std(actual)
    sd_pred = np.std(pred)
    corr, _ = pearsonr(actual, pred)
    return (mean_actual - mean_pred)**2 + (sd_actual - sd_pred)**2 + 2*(1-corr)*sd_actual*sd_pred

def PSM(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    mean_actual = np.mean(actual)
    mean_pred = np.mean(pred)
    return ((mean_actual - mean_pred)**2) / media_e2(actual, pred)

def PSV(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    sd_actual = np.std(actual)
    sd_pred = np.std(pred)
    return ((sd_actual - sd_pred)**2) / media_e2(actual, pred)

def PC_(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    sd_actual = np.std(actual)
    sd_pred = np.std(pred)
    corr, _ = pearsonr(actual, pred)
    return (2*(1-corr)*sd_actual*sd_pred) / media_e2(actual, pred)