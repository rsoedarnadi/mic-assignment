1# %% [markdown]
# # ICT665 Assignment - Genetic Algorithm Feature Selection and Polynomial Regression 

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Assume df is your DataFrame with numerical, scaled predictors
def calculate_vif(data):
    vif_df = pd.DataFrame()
    vif_df['Feature'] = data.columns
    vif_df['VIF'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_df.sort_values(by='VIF', ascending=False)

# 1. Setup Models
degree = 2
models = {
    'LinearRegression': (LinearRegression(), {}),
    'Lasso': (Lasso(), {"alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}),
    'Ridge': (Ridge(), {"alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}),
    'ElasticNet': (ElasticNet(), {"alpha": [0.1, 1.0, 5.0], "l1_ratio": [0.5]}),
    'RandomForest': (RandomForestRegressor(), {"n_estimators": [100, 200], "max_depth": [None, 10]})
}


def get_performance(X_train, y_train, X_test, y_test, model, config):
    grid = GridSearchCV(model, config, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    predictions = best_model.predict(X_test)
    
    correlation = np.corrcoef(y_test, predictions)[0, 1]
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    return pd.Series({
        "Best Config": best_params,
        "Correlation": correlation,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# %%
data = pd.read_csv("/Users/rsoedarnadi/Documents/Github/mic-peptide-assignment/Data.csv")
data['TARGET'] = data['TARGET'].str.replace(r'\s+', '', regex=True).astype(float)
X = data.drop("TARGET", axis=1)
y = data["TARGET"]
print("Peptide dataset:\n",X.shape[0],"Records\n",X.shape[1],"Features")
data.tail()

# %%
print(data.isna().sum())

# %% [markdown]
# ## Data Manipulation and Handling Multicollinearity

# %%
import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = X.corr()

plt.figure(figsize=(16, 16)) 
sns.heatmap(
    correlation_matrix, 
    cmap='coolwarm',
    fmt='.2f',          
    linewidths=0.1,
    center=0  
)
# 4. Add a title and display the plot
plt.title("Correlation Matrix of Peptide Features")
plt.show()

vif_data = calculate_vif(X)
print("Variance Inflation Factor (VIF) for each feature:")
print(vif_data.sort_values(by='VIF', ascending=False))

# %%
# 5. Drop features with high correlation (r > 0.8)
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(np.absolute(upper[column]) > 0.7)]
print(f"Dropping features: {to_drop}")
X = X.drop(columns=to_drop)
correlation_matrix = X.corr()

plt.figure(figsize=(16, 16)) 
sns.heatmap(
    correlation_matrix, 
    cmap='coolwarm',
    fmt='.2f',          
    linewidths=0.1,
    center=0  
)
# 4. Add a title and display the plot
plt.title("Correlation Matrix of Peptide Features")
plt.show()

vif_data = calculate_vif(X)
print("Variance Inflation Factor (VIF) for each feature:")
print(vif_data.sort_values(by='VIF', ascending=False))

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# %%
results_list = []

for model_name, (model_obj, config) in models.items():
    print(f"Now training and tuning: {model_name}...")
    
    performance_metrics = get_performance(X_train, y_train, X_test, y_test, model_obj, config)
    performance_metrics.name = model_name
    results_list.append(performance_metrics)

final_report = pd.DataFrame(results_list)
print("\nFinal Performance Report:")
print(final_report)

# %%
# Polynomial Regression (Degree 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

final_report = pd.DataFrame(results_list)
print("Training Polynomial Degree 2...")
poly_results = get_performance(X_train_poly, y_train, X_test_poly, y_test, LinearRegression(), {})
poly_results.name = "Polynomial_Deg2"
final_report = pd.concat([final_report, poly_results.to_frame().T])
print("\nUpdated Performance Report with Polynomial Regression:")
print(final_report)

# %%
final_report = final_report.sort_values(by='R2', ascending=False)
final_report.to_csv("model_performance_report.csv", index=True)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
from sklearn.metrics import r2_score
from itertools import product
import pandas as pd

def initialization_of_population(size, n_feat):
    population = []
    for i in range(size):
        # Create a boolean mask for features
        chromosome = np.ones(n_feat, dtype=bool)     
        chromosome[:int(0.3 * n_feat)] = False             
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def selection(pop_after_fit, n_parents):
    return pop_after_fit[:n_parents]

import random

def crossover(pop_after_sel, crossover_prob):
    pop_nextgen = []
    for i in range(0, len(pop_after_sel) - 1, 2):
        child_1, child_2 = pop_after_sel[i], pop_after_sel[i+1]
        if random.random() < crossover_prob:
            cut = len(child_1) // 2
            new_child1 = np.concatenate((child_1[:cut], child_2[cut:]))
            new_child2 = np.concatenate((child_2[:cut], child_1[cut:]))
            pop_nextgen.extend([new_child1, new_child2])
        else:
            pop_nextgen.extend([child_1, child_2]) 
    return pop_nextgen

def mutation(pop_after_cross, mutation_rate, n_feat):   
    mutation_range = int(mutation_rate * n_feat)
    pop_next_gen = []
    for chromo in pop_after_cross:
        new_chromo = chromo.copy()
        for _ in range(max(1, mutation_range)):
            pos = randint(0, n_feat - 1)
            new_chromo[pos] = not new_chromo[pos]
        pop_next_gen.append(new_chromo)
    return pop_next_gen

def fitness_score(population, model, X_train, Y_train, X_test, Y_test):
    fitness_results = [] # To store all metrics per chromosome
    
    for chromosome in population:
        if not any(chromosome):
            # Penalize empty feature sets
            fitness_results.append({
                'r2': -np.inf, 'mae': np.inf, 'rmse': np.inf, 'corr': -1
            })
            continue
        
        # Handle both DataFrame and Numpy array indexing
        X_tr_sub = X_train.iloc[:, chromosome] if hasattr(X_train, 'iloc') else X_train[:, chromosome]
        X_te_sub = X_test.iloc[:, chromosome] if hasattr(X_test, 'iloc') else X_test[:, chromosome]
        
        # Train and Predict
        model.fit(X_tr_sub, Y_train)         
        predictions = model.predict(X_te_sub)
        
        # Calculate Metrics
        # Handle edge case where correlation might fail if predictions are constant
        try:
            correlation = np.corrcoef(Y_test, predictions)[0, 1]
            if np.isnan(correlation): correlation = 0
        except:
            correlation = 0
            
        mae = mean_absolute_error(Y_test, predictions)
        rmse = np.sqrt(mean_squared_error(Y_test, predictions))
        r2 = r2_score(Y_test, predictions)
        
        # Store as a dictionary
        fitness_results.append({
            'r2': r2, 
            'mae': mae, 
            'rmse': rmse, 
            'corr': correlation
        })
        
    scores_for_sorting = np.array([res['r2'] for res in fitness_results])
    population = np.array(population) 
    
    inds = np.argsort(scores_for_sorting)[::-1]
    
    sorted_metrics = [fitness_results[i] for i in inds]
    sorted_population = list(population[inds, :])
    
    return sorted_metrics, sorted_population

def run_genetic_evolution(model, size, n_parents, mutation_rate, n_gen, crossover_prob, X_train, X_test, Y_train, Y_test):
    n_feat = X_train.shape[1]
    
    population_nextgen = initialization_of_population(size, n_feat)
    
    for i in range(n_gen):
        # Using the dictionary-based fitness_score from the previous step
        metrics_list, pop_after_fit = fitness_score(population_nextgen, model, X_train, Y_train, X_test, Y_test)
        
        # Evolution steps
        pop_after_sel = selection(pop_after_fit, n_parents)
        pop_after_cross = crossover(pop_after_sel, crossover_prob)
        population_nextgen = mutation(pop_after_cross, mutation_rate, n_feat)
        
    # Return the absolute best from the final generation
    # metrics_list[0] contains {'r2': ..., 'mae': ..., 'rmse': ..., 'corr': ...}
    return pop_after_fit[0], metrics_list[0]


def ga_grid_search(model, X_train, X_test, Y_train, Y_test, param_grid):
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    results_list = []
    global_best_r2 = -float('inf')
    global_best_chromo = None
    
    print(f"Starting Grid Search for {len(combinations)} combinations...")
    
    for i, params in enumerate(combinations):
        # run_genetic_evolution now returns: (best_chromo, best_metrics_dict)
        best_chromo, metrics = run_genetic_evolution(
            model=model,
            size= 80,
            n_parents= 80 // 2,
            mutation_rate=params['mutation_rate'],
            crossover_prob=params['crossover_prob'],
            n_gen=params['n_gen'],
            X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test
        )
        
        # Track the best chromosome across ALL grid combinations
        if metrics['r2'] > global_best_r2:
            global_best_r2 = metrics['r2']
            global_best_chromo = best_chromo

        
        # Build the result row for the DataFrame
        entry = {**params, **metrics} # Merges params and metrics dictionaries
        results_list.append(entry)
        
        print(f"Config {i+1}/{len(combinations)}: R2 = {metrics['r2']:.4f}")
        
    # Return 1. The full table, 2. The best mask, 3. The best metrics
    return pd.DataFrame(results_list).sort_values(by='r2', ascending=False), global_best_chromo

ga_config = {
    'n_gen': [3, 5, 10],
    'mutation_rate': [0.1, 0.2],
    'crossover_prob': [0.5, 0.8]
}

# %%
best_linear = LinearRegression()
ga_metrics_linear, best_mask_linear = ga_grid_search(best_linear, X_train, X_test, y_train, y_test, ga_config)
print("GA Metrics for Linear Regression:\n", ga_metrics_linear)
print("Number of features selected:", np.sum(best_mask_linear))
print("Selected features:\n", X_train.columns[best_mask_linear])

# %%
best_rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
ga_metrics_rf, best_mask_rf = ga_grid_search(best_rf, X_train, X_test, y_train, y_test, ga_config)
print("GA Metrics for Random Forest:\n", ga_metrics_rf)
print("Number of features selected:", np.sum(best_mask_rf))
print("Selected features:\n", X_train.columns[best_mask_rf])

# %%
best_lasso = Lasso(alpha=0.01)
ga_metrics_lasso, best_mask_lasso = ga_grid_search(best_lasso, X_train, X_test, y_train, y_test, ga_config)
print("GA Metrics for Lasso:\n", ga_metrics_lasso)
print("Number of features selected:", np.sum(best_mask_lasso))
print("Selected features:\n", X_train.columns[best_mask_lasso])

# %%
best_ridge = Ridge(alpha=0.1)
ga_metrics_ridge, best_mask_ridge = ga_grid_search(best_ridge, X_train, X_test, y_train, y_test, ga_config)
print("GA Metrics for Ridge:\n", ga_metrics_ridge)
print("Number of features selected:", np.sum(best_mask_ridge))
print("Selected features:\n", X_train.columns[best_mask_ridge])

# %%
best_elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5)
ga_metrics_elasticnet, best_mask_elasticnet = ga_grid_search(best_elasticnet, X_train, X_test, y_train, y_test, ga_config)
print("GA Metrics for ElasticNet:\n", ga_metrics_elasticnet)
print("Number of features selected:", np.sum(best_mask_elasticnet))
print("Selected features:\n", X_train.columns[best_mask_elasticnet])

# %%
best_poly = LinearRegression()  # Using LinearRegression for polynomial features
ga_metrics_poly, best_mask_poly = ga_grid_search(best_poly, X_train_poly, X_test_poly, y_train, y_test, ga_config)
print("GA Metrics for Polynomial Regression:\n", ga_metrics_poly)
print("Number of features selected:", np.sum(best_mask_poly))
print("Selected features:\n", poly.get_feature_names_out(X_train.columns)[best_mask_poly])

# %%
X_train_linear = X_train.iloc[:, best_mask_linear]
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_model_ga = LinearRegression()
linear_model_ga.fit(X_train_linear, y_train)
linear_model_coef = pd.Series(linear_model.coef_, index=X_train.columns)
linear_model_ga_coef = pd.Series(linear_model_ga.coef_, index=X_train_linear.columns)
linear_coef_summary = pd.DataFrame([linear_model_coef.describe(), linear_model_ga_coef.describe()], index=['Linear', 'Linear_GA']).T
print("\nCoefficient Summary Comparison:")
print(linear_coef_summary)
linear_coef_summary.to_csv("linear_coefficients_comparison.csv", index=True)


# %%
print("Number of features before and after selection:")
print(f"Original features: {X_train.shape[1]}")
print(f"Linear selected features: {len(np.sum(best_mask_linear))}")
print(f"Original features (Polynomial Degree 2): {X_train_poly.shape[1]}")
print(f"Polynomial selected features: {len(best_mask_poly[best_mask_poly==True])}")



# %%
best_performance_ga = pd.DataFrame([ga_metrics_linear.iloc[0], ga_metrics_rf.iloc[0], ga_metrics_lasso.iloc[0], ga_metrics_ridge.iloc[0], ga_metrics_elasticnet.iloc[0], ga_metrics_poly.iloc[0]], index=['Linear', 'RandomForest', 'Lasso', 'Ridge', 'ElasticNet', 'Polynomial']).sort_values(by='r2', ascending=False)
best_performance_ga['num_features'] = [np.sum(best_mask_rf), np.sum(best_mask_poly), np.sum(best_mask_linear), np.sum(best_mask_lasso), np.sum(best_mask_ridge), np.sum(best_mask_elasticnet)]
print("\nBest GA Performance Summary:")
print(best_performance_ga)
best_performance_ga.to_csv("best_ga_performance_summary.csv", index=True)

# %%



