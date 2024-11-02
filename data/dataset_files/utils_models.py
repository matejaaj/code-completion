import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import warnings

warnings.filterwarnings("ignore")

def plot_correlation_for_col(df, col_name):
    plt.figure(figsize=(12,6))
    correlation_matrix = df.corr()
    sorted_col_corr = correlation_matrix[col_name].sort_values(ascending=True)
    sorted_col_corr = sorted_col_corr.drop(col_name)
    sb.barplot(x=sorted_col_corr.index, y=sorted_col_corr.values, palette='RdBu')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_explained_variance(pca_model):
    plt.figure(figsize=(9,3))
    explained_variance = pca_model.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.8, align='center')
    plt.xlabel('Glavna komponenta')
    plt.ylabel('Objasnjena varijansa')
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, '--o')
    plt.xlabel('Broj glavnih komponenti')
    plt.ylabel('Kumulativna varijansa')
    plt.tight_layout()
    plt.show()

def plot_pc_loading(pca_model, pc_idx, columns, largest_n_pc=None):
    plt.figure(figsize=(12,6))
    pc_loadings_df = pd.DataFrame(data=pca_model.components_, columns=columns)
    loading = pc_loadings_df.iloc[pc_idx]
    sorted_loading_abs = loading.abs().sort_values(ascending=True)
    largest_n_pc = 0 if largest_n_pc is None else largest_n_pc
    sorted_loading = loading[sorted_loading_abs.index][-largest_n_pc:]
    sb.barplot(x=sorted_loading.index, y=sorted_loading.values, palette='Reds')
    plt.xticks(rotation=90)
    plt.title(f'Correlation with {pc_idx}. component')
    plt.tight_layout()
    plt.show()

def visualize_principal_components(principal_components: np.ndarray, n_principal_components: int, 
                                   target_col: pd.Series = None, n_samples: int = None):
    if n_samples is not None and n_samples < principal_components.shape[0]:
        indices = np.random.choice(principal_components.shape[0], n_samples, replace=False)
        principal_components = principal_components[indices, :]
        if target_col is not None:
            target_col = target_col.iloc[indices]
    if n_principal_components == 2:
        fig = px.scatter(x=principal_components[:, 0], y=principal_components[:, 1],
                         opacity=0.6, color=target_col, color_continuous_scale='RdBu', width=700, height=600)
        fig.update_traces(marker={'size': 10})
        fig.update_layout(title='Principal components visualisations', xaxis_title="PC1", yaxis_title="PC2")
        fig.show()
    elif n_principal_components == 3:
        fig = px.scatter_3d(x=principal_components[:, 0], y=principal_components[:, 1], z=principal_components[:, 2],
                            opacity=0.6, color=target_col, color_continuous_scale='RdBu', width=1000)
        fig.update_traces(marker={'size': 6})
        fig.update_layout(title='Principal components visualisations', 
                          scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
                                     xaxis_autorange='reversed', yaxis_autorange='reversed'))
        fig.show()
    else:
        raise Exception('number of principal components must be 2 or 3')

def getPcaModel(df, n_components=2, random_state=42):
    x = df.drop(columns=['league_rank', 'league_id'])
    x_scaled = scale(x)
    pca_model = PCA(n_components=n_components, random_state=random_state)
    principal_components = pca_model.fit_transform(x_scaled)
    return pca_model, principal_components

def getRandomForestClassifier(X_train, y_train):
    model = RandomForestClassifier(n_estimators=300, random_state=42,
                                   max_depth=10, bootstrap=True, min_samples_leaf=1,
                                   min_samples_split=10)
    model.fit(X_train, y_train)
    return model

def getMLPClassifier(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(50,50), activation='relu', solver='adam',
                          max_iter=1000, learning_rate='constant', alpha=0.05,
                          early_stopping=True, random_state=42)
    model.fit(scale(X_train), y_train)
    return model

def getMLPRegressorModel(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, 
                         learning_rate_init=0.001, early_stopping=True, random_state=42)
    model.fit(scale(X_train), y_train)
    return model

def getRandomForestModel(X_train, y_train):
    model = RandomForestRegressor(n_estimators=300, random_state=42,
                                  max_depth=10, bootstrap=True, min_samples_leaf=4,
                                  min_samples_split=2)
    model.fit(X_train, y_train)
    return model

def getLinearRegressionModel(X_train, y_train):
    x_with_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, x_with_const).fit()
    return model

def getFeatureImportancesPca(model):
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("PCA Component Importances")
    plt.bar(range(len(feature_importances)), feature_importances[sorted_idx], align="center")
    plt.xticks(range(len(feature_importances)), [f"PC{i}" for i in sorted_idx], rotation=90)
    plt.xlim([-1, len(feature_importances)])
    plt.show()

def getFeatureImportances(model, columns):
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(columns)), feature_importances[sorted_idx], align="center")
    plt.xticks(range(len(columns)), [columns[i] for i in sorted_idx], rotation=90)
    plt.xlim([-1, len(columns)])
    plt.show()

def league_test_split(df, test_size=0.2, random_state=42):
    leagues = df['league_id'].unique()
    train_leagues, test_leagues = train_test_split(leagues, test_size=test_size, random_state=random_state)
    X_train = df[df['league_id'].isin(train_leagues)].drop(['league_rank', 'league_id'], axis=1)
    y_train = df[df['league_id'].isin(train_leagues)]['league_rank']
    X_test = df[df['league_id'].isin(test_leagues)].drop(['league_rank', 'league_id'], axis=1)
    y_test = df[df['league_id'].isin(test_leagues)]['league_rank']
    return X_train, X_test, y_train, y_test

def scale(x):
    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(x)

def evaluate_model_performance(model_name, y_true, predictions, num_attributes, print_accuracy=False):
    print(f"Model: {model_name}")
    if not print_accuracy:
        mae = mean_absolute_error(y_true, predictions)
        print(f"Mean Absolute Error (MAE): {mae}")
        r2 = r2_score(y_true, predictions)
        ar2 = get_rsquared_adj(r2, len(predictions), num_attributes)
        print(f"Adjusted R-squared: {r2}")
    else:
        accuracy = accuracy_score(y_true, predictions)
        print(f'Accuracy: {accuracy:.4f}')

def plot_predictions_distribution(model_name, y_true, predictions):
    plt.hist(predictions, bins=len(set(y_true)), edgecolor='black')
    plt.xlabel('League rank')
    plt.ylabel('Number of predictions')
    plt.title(f'Distribution {model_name}')
    plt.show()

def get_rsquared_adj(r_squared, n, p):
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    return adjusted_r_squared

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df, columns_to_normalize, columns_to_drop, groupby_column='league_id'):
    for col in columns_to_normalize:
        normalized_col_name = f'normalized_{col}'
        if col in df.columns:
            normalized_values = df.groupby(groupby_column)[col].transform(lambda x: (x - x.mean()) / x.std())
            df[normalized_col_name] = normalized_values.fillna(0)
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    return df
