# Complete Weighted Feature Regression Analysis
import pandas as pd
import numpy as np
import os
import joblib
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Genetic Algorithm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from deap import base, creator, tools, algorithms
import random

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class GeneticAlgorithmRegressor(BaseEstimator, RegressorMixin):
    """Simple Genetic Algorithm for feature selection + Linear Regression"""
    
    def __init__(self, population_size=50, generations=20, mutation_prob=0.1, 
                 crossover_prob=0.7, tournament_size=3):
        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournament_size = tournament_size
        self.best_features_ = None
        self.regressor_ = LinearRegression()
        
    def _evaluate_individual(self, individual, X, y):
        """Evaluate fitness of an individual (feature subset)"""
        selected_features = [i for i, bit in enumerate(individual) if bit == 1]
        
        if len(selected_features) == 0:
            return (float('inf'),)  # Invalid solution
        
        X_selected = X[:, selected_features]
        
        # Use cross-validation to evaluate
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(self.regressor_, X_selected, y, 
                               cv=kf, scoring='neg_mean_squared_error')
        
        return (-np.mean(scores),)  # Return positive MSE
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_features = X.shape[1]
        
        # Initialize DEAP
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                        toolbox.attr_bool, n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_individual, X=X, y=y)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutation_prob)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
        # Create initial population
        pop = toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Evolution
        for gen in range(self.generations):
            # Selection
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            
            # Crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            pop[:] = offspring
        
        # Get best individual
        best_ind = tools.selBest(pop, 1)[0]
        self.best_features_ = [i for i, bit in enumerate(best_ind) if bit == 1]
        
        # Train final model
        if len(self.best_features_) > 0:
            X_selected = X[:, self.best_features_]
            self.regressor_.fit(X_selected, y)
        
        return self
    
    def predict(self, X):
        check_array(X)
        if self.best_features_ is None or len(self.best_features_) == 0:
            return np.zeros(X.shape[0])
        
        X_selected = X[:, self.best_features_]
        return self.regressor_.predict(X_selected)

def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("Loading data...")
    
    # Load dataset
    dataset_path = '3_merged_data3.txt'
    data = pd.read_csv(dataset_path, sep='\t')
    
    # Load p-values file
    pval_path = '3_transposed_headers_with_scores.txt'
    pvals = pd.read_csv(pval_path, sep='\t')
    
    # Extract features and target
    target_col = 'avg7_calingiri'
    feature_cols = [col for col in data.columns if col != target_col and col != 'ID']
    
    X = data[feature_cols]
    y = data[target_col]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y, pvals

def calculate_feature_weights(X, pvals):
    """Calculate feature weights based on p-values"""
    print("Calculating feature weights...")
    
    # Get list of features from dataset columns
    dataset_features = set(X.columns)
    pval_features = set(pvals['isoform'])
    
    # Find missing and extra features
    missing_in_pval = dataset_features - pval_features
    extra_in_pval = pval_features - dataset_features
    
    print(f"Features in dataset not in p-value file: {len(missing_in_pval)}")
    print(f"Extra features in p-value file not in dataset: {len(extra_in_pval)}")
    
    # Map p-values by feature
    pval_map = pvals.set_index('isoform')['p-value_lowest'].to_dict()
    
    # Calculate weights with improved logic
    weights = {}
    for feat in X.columns:
        p = pval_map.get(feat, None)
        if p is None:
            # For missing p-values, use neutral weight
            weights[feat] = 1.0
        else:
            # Convert p-value to weight: smaller p-value = higher weight
            # Use -log10(p) transformation to avoid extreme values
            if p <= 0:
                weights[feat] = 10.0  # Maximum weight for p=0
            else:
                weights[feat] = -np.log10(p) + 1  # Add 1 to ensure positive weights
    
    # Normalize weights to reasonable range [0.1, 2.0]
    weights_array = np.array(list(weights.values()))
    weights_min, weights_max = weights_array.min(), weights_array.max()
    
    if weights_max > weights_min:
        weights_normalized = {
            feat: 0.1 + (weight - weights_min) / (weights_max - weights_min) * 1.9
            for feat, weight in weights.items()
        }
    else:
        weights_normalized = {feat: 1.0 for feat in weights.keys()}
    
    print(f"Weight range: {min(weights_normalized.values()):.3f} - {max(weights_normalized.values()):.3f}")
    
    return weights_normalized

def preprocess_data(X, y, weights=None):
    """Preprocess data with optional weighting"""
    print("Preprocessing data...")
    
    # Handle missing values
    X_clean = X.fillna(X.median())
    
    # Apply weights if provided
    if weights is not None:
        weights_series = pd.Series(weights)
        X_weighted = X_clean * weights_series
        
        # Remove features with very low variance after weighting
        selector = VarianceThreshold(threshold=1e-6)
        X_filtered = pd.DataFrame(
            selector.fit_transform(X_weighted),
            columns=X_weighted.columns[selector.get_support()],
            index=X_weighted.index
        )
        
        print(f"Features after variance filtering: {X_filtered.shape[1]}")
        return X_filtered, y
    else:
        # Remove features with very low variance
        selector = VarianceThreshold(threshold=1e-6)
        X_filtered = pd.DataFrame(
            selector.fit_transform(X_clean),
            columns=X_clean.columns[selector.get_support()],
            index=X_clean.index
        )
        
        print(f"Features after variance filtering: {X_filtered.shape[1]}")
        return X_filtered, y

def get_algorithms():
    """Define regression algorithms to test"""
    algorithms = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1, max_iter=2000),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1),
        'CatBoost': CatBoostRegressor(n_estimators=100, random_state=42, verbose=False),
        'Genetic Algorithm': GeneticAlgorithmRegressor(population_size=30, generations=10)
    }
    
    return algorithms

def evaluate_algorithm(name, algorithm, X, y, cv_folds=5):
    """Evaluate algorithm using cross-validation"""
    try:
        # K-fold cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Metrics to track
        mse_scores = []
        r2_scores = []
        mae_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit and predict
            algorithm.fit(X_train_fold, y_train_fold)
            y_pred = algorithm.predict(X_val_fold)
            
            # Calculate metrics
            mse_scores.append(mean_squared_error(y_val_fold, y_pred))
            r2_scores.append(r2_score(y_val_fold, y_pred))
            mae_scores.append(mean_absolute_error(y_val_fold, y_pred))
        
        return {
            'Algorithm': name,
            'MSE_mean': np.mean(mse_scores),
            'MSE_std': np.std(mse_scores),
            'R2_mean': np.mean(r2_scores),
            'R2_std': np.std(r2_scores),
            'MAE_mean': np.mean(mae_scores),
            'MAE_std': np.std(mae_scores),
            'RMSE_mean': np.sqrt(np.mean(mse_scores))
        }
        
    except Exception as e:
        print(f"Error evaluating {name}: {str(e)}")
        return {
            'Algorithm': name,
            'MSE_mean': np.nan,
            'MSE_std': np.nan,
            'R2_mean': np.nan,
            'R2_std': np.nan,
            'MAE_mean': np.nan,
            'MAE_std': np.nan,
            'RMSE_mean': np.nan
        }

def run_experiments(X, y, weights, cv_folds=5):
    """Run experiments on original and weighted data"""
    print("Starting experiments...")
    
    # Prepare datasets
    X_original, y_clean = preprocess_data(X, y, weights=None)
    X_weighted, _ = preprocess_data(X, y, weights=weights)
    
    # Get algorithms
    algorithms = get_algorithms()
    
    results = []
    
    # Test on original data
    print("\nEvaluating on original data...")
    for name, algorithm in tqdm(algorithms.items(), desc="Original Data"):
        result = evaluate_algorithm(name, algorithm, X_original, y_clean, cv_folds)
        result['Data_Type'] = 'Original'
        results.append(result)
    
    # Test on weighted data
    print("\nEvaluating on weighted data...")
    for name, algorithm in tqdm(algorithms.items(), desc="Weighted Data"):
        result = evaluate_algorithm(name, algorithm, X_weighted, y_clean, cv_folds)
        result['Data_Type'] = 'Weighted'
        results.append(result)
    
    return pd.DataFrame(results)

def save_and_analyze_results(results_df):
    """Save results and perform analysis"""
    # Save to CSV
    output_file = 'regression_results_comparison.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Display summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Best performing algorithms
    print("\nTop 5 Algorithms by R² Score (Original Data):")
    original_results = results_df[results_df['Data_Type'] == 'Original'].copy()
    original_results = original_results.dropna(subset=['R2_mean'])
    original_results = original_results.sort_values('R2_mean', ascending=False)
    print(original_results[['Algorithm', 'R2_mean', 'RMSE_mean']].head())
    
    print("\nTop 5 Algorithms by R² Score (Weighted Data):")
    weighted_results = results_df[results_df['Data_Type'] == 'Weighted'].copy()
    weighted_results = weighted_results.dropna(subset=['R2_mean'])
    weighted_results = weighted_results.sort_values('R2_mean', ascending=False)
    print(weighted_results[['Algorithm', 'R2_mean', 'RMSE_mean']].head())
    
    # Comparison analysis
    print("\n" + "="*50)
    print("WEIGHTED vs ORIGINAL COMPARISON")
    print("="*50)
    
    comparison_results = []
    for algorithm in results_df['Algorithm'].unique():
        orig_r2 = original_results[original_results['Algorithm'] == algorithm]['R2_mean'].values
        weight_r2 = weighted_results[weighted_results['Algorithm'] == algorithm]['R2_mean'].values
        
        if len(orig_r2) > 0 and len(weight_r2) > 0 and not np.isnan(orig_r2[0]) and not np.isnan(weight_r2[0]):
            improvement = weight_r2[0] - orig_r2[0]
            comparison_results.append({
                'Algorithm': algorithm,
                'Original_R2': orig_r2[0],
                'Weighted_R2': weight_r2[0],
                'Improvement': improvement,
                'Improvement_Pct': (improvement / abs(orig_r2[0])) * 100 if orig_r2[0] != 0 else 0
            })
    
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.sort_values('Improvement', ascending=False)
    
    print("\nPerformance Improvement with P-value Weighting:")
    print(comparison_df[['Algorithm', 'Original_R2', 'Weighted_R2', 'Improvement_Pct']].round(4))
    
    # Statistical significance
    improved_algorithms = comparison_df[comparison_df['Improvement'] > 0]
    degraded_algorithms = comparison_df[comparison_df['Improvement'] < 0]
    
    print(f"\nAlgorithms improved by weighting: {len(improved_algorithms)}")
    print(f"Algorithms degraded by weighting: {len(degraded_algorithms)}")
    print(f"Average improvement: {comparison_df['Improvement_Pct'].mean():.2f}%")
    
    return results_df, comparison_df

def main():
    """Main execution function"""
    print("="*80)
    print("WEIGHTED FEATURE REGRESSION ANALYSIS")
    print("="*80)
    
    try:
        # Load data
        X, y, pvals = load_and_prepare_data()
        
        # Calculate weights
        weights = calculate_feature_weights(X, pvals)
        
        # Run experiments
        results_df = run_experiments(X, y, weights, cv_folds=5)
        
        # Analyze and save results
        results_df, comparison_df = save_and_analyze_results(results_df)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"Total experiments run: {len(results_df)}")
        print("Check 'regression_results_comparison.csv' for detailed results")
        
        return results_df, comparison_df
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results_df, comparison_df = main()