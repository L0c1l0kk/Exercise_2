"""
Test suite for RegressionTree implementation.
Run this file after implementing Custom RegressionTree class.

Usage:
    python test_regression_tree.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

# Import Custom implementation
# from Custom_module import RegressionTree


class TestData:
    """Generate various test datasets"""
    
    @staticmethod
    def linear_data(n_samples=100, noise=1.0, random_state=42):
        """Generate simple linear relationship data"""
        np.random.seed(random_state)
        X = np.linspace(0, 10, n_samples).reshape(-1, 1)
        y = 2 * X.flatten() + 1 + np.random.normal(0, noise, n_samples)
        return X, y
    
    @staticmethod
    def sine_wave(n_samples=200, noise=0.1, random_state=42):
        """Generate sine wave data"""
        np.random.seed(random_state)
        X = np.linspace(0, 2*np.pi, n_samples).reshape(-1, 1)
        y = np.sin(X.flatten()) + np.random.normal(0, noise, n_samples)
        return X, y
    
    @staticmethod
    def quadratic_data(n_samples=150, noise=0.5, random_state=42):
        """Generate quadratic relationship"""
        np.random.seed(random_state)
        X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
        y = X.flatten()**2 + np.random.normal(0, noise, n_samples)
        return X, y
    
    @staticmethod
    def multidimensional_data(n_samples=500, n_features=5, random_state=42):
        """Generate multi-dimensional data"""
        np.random.seed(random_state)
        X = np.random.randn(n_samples, n_features)
        # Complex relationship
        y = (3*X[:, 0] - 2*X[:, 1] + X[:, 2]**2 + 
             0.5*X[:, 3]*X[:, 4] + np.random.normal(0, 0.5, n_samples))
        return X, y
    
    @staticmethod
    def step_function(n_samples=200, random_state=42):
        """Generate step function data (perfect for trees)"""
        np.random.seed(random_state)
        X = np.linspace(0, 10, n_samples).reshape(-1, 1)
        y = np.zeros(n_samples)
        y[X.flatten() < 2.5] = 1
        y[(X.flatten() >= 2.5) & (X.flatten() < 5)] = 3
        y[(X.flatten() >= 5) & (X.flatten() < 7.5)] = 2
        y[X.flatten() >= 7.5] = 4
        y += np.random.normal(0, 0.1, n_samples)
        return X, y
    
    @staticmethod
    def pandas_dataframe(n_samples=200, random_state=42):
        """Generate pandas DataFrame with target column"""
        np.random.seed(random_state)
        df = pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.randn(n_samples) * 20000 + 50000,
            'credit_score': np.random.randint(300, 850, n_samples),
            'years_employed': np.random.randint(0, 40, n_samples),
        })
        # Target based on features
        df['loan_amount'] = (0.3 * df['income'] + 
                            50 * df['credit_score'] + 
                            1000 * df['years_employed'] + 
                            np.random.normal(0, 5000, n_samples))
        return df
    
    @staticmethod
    def constant_target(n_samples=100, random_state=42):
        """Data with constant target (edge case)"""
        np.random.seed(random_state)
        X = np.random.randn(n_samples, 3)
        y = np.ones(n_samples) * 42.0
        return X, y
    
    @staticmethod
    def single_feature(n_samples=100, random_state=42):
        """Single feature data"""
        np.random.seed(random_state)
        X = np.random.randn(n_samples, 1)
        y = 3 * X.flatten() + np.random.normal(0, 0.5, n_samples)
        return X, y


class RegressionTreeTester:
    """Test harness for RegressionTree"""
    
    def __init__(self, tree_class):
        self.tree_class = tree_class
        self.results = []
    
    def run_test(self, name, X, y, max_depth=5, min_size=10, test_size=0.2, 
                 plot=True, compare_sklearn=True):
        """Run a single test"""
        print("=" * 60)
        print(f"TEST: {name}")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train Custom tree
        tree = self.tree_class()
        tree.fit(X_train, y_train, max_depth=max_depth, min_size=min_size)
        y_pred_test = tree.predict(X_test)
        y_pred_train = tree.predict(X_train)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        
        print(f"Custom Tree (Test Set):")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²:  {r2:.4f}")
        
        print(f"\nCustom Tree (Train Set):")
        print(f"  MSE: {train_mse:.4f}")
        print(f"  R²:  {train_r2:.4f}")
        print(f"  Overfit Check: Train R² - Test R² = {train_r2 - r2:.4f} "
              f"({'⚠️ OVERFITTING' if train_r2 - r2 > 0.15 else '✓ OK'})")
        
        result = {
            'name': name,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'train_mse': train_mse,
            'train_r2': train_r2,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X.shape[1] if len(X.shape) > 1 else 1
        }
        
        # Compare with sklearn
        if compare_sklearn:
            sklearn_tree = DecisionTreeRegressor(
                max_depth=max_depth, 
                min_samples_split=min_size*2
            )
            sklearn_tree.fit(X_train, y_train)
            y_pred_sklearn_test = sklearn_tree.predict(X_test)
            y_pred_sklearn_train = sklearn_tree.predict(X_train)
            
            sklearn_mse = mean_squared_error(y_test, y_pred_sklearn_test)
            sklearn_r2 = r2_score(y_test, y_pred_sklearn_test)
            sklearn_train_r2 = r2_score(y_train, y_pred_sklearn_train)
            
            print(f"\nSklearn DecisionTreeRegressor (Test Set):")
            print(f"  MSE: {sklearn_mse:.4f}")
            print(f"  R²:  {sklearn_r2:.4f}")
            
            print(f"\nComparison:")
            print(f"  MSE Ratio (Customs/sklearn): {mse/sklearn_mse:.2f}x")
            print(f"  R² Difference: {r2 - sklearn_r2:.4f}")
            
            result['sklearn_mse'] = sklearn_mse
            result['sklearn_r2'] = sklearn_r2
            result['sklearn_train_r2'] = sklearn_train_r2
        
        # Plot if requested
        if plot and X.shape[1] == 1:
            self._plot_1d_results(X_train, y_train, X_test, y_test, 
                                 y_pred_train, y_pred_test,
                                 y_pred_sklearn_train if compare_sklearn else None,
                                 y_pred_sklearn_test if compare_sklearn else None,
                                 name)
        elif plot and X.shape[1] > 1:
            self._plot_nd_results(y_test, y_pred_test,
                                 y_pred_sklearn_test if compare_sklearn else None,
                                 name)
        
        self.results.append(result)
        print()
        return result
    
    def _plot_1d_results(self, X_train, y_train, X_test, y_test, 
                        y_pred_train, y_pred_test, 
                        y_pred_sklearn_train, y_pred_sklearn_test, title):
        """Plot results for 1D data - both train and test sets"""
        fig = plt.figure(figsize=(14, 8))
        
        # Sort for better visualization
        sort_idx_test = X_test.flatten().argsort()
        sort_idx_train = X_train.flatten().argsort()
        
        # Top row: Test set
        # Custom tree
        plt.subplot(231)
        plt.scatter(X_test[sort_idx_test], y_test[sort_idx_test], 
                   alpha=0.5, label='Actual', s=20, color='lightblue')
        plt.plot(X_test[sort_idx_test], y_pred_test[sort_idx_test], 
                'r-', linewidth=2, label='Custom Tree')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'{title}\nCustom Tree (Test Set)')
        plt.grid(True, alpha=0.3)
        
        # Sklearn tree
        if y_pred_sklearn_test is not None:
            plt.subplot(232)
            plt.scatter(X_test[sort_idx_test], y_test[sort_idx_test], 
                       alpha=0.5, label='Actual', s=20, color='lightblue')
            plt.plot(X_test[sort_idx_test], y_pred_sklearn_test[sort_idx_test], 
                    'g-', linewidth=2, label='Sklearn')
            plt.xlabel('X')
            plt.ylabel('y')
            plt.legend()
            plt.title('Sklearn Tree (Test Set)')
            plt.grid(True, alpha=0.3)
        
        # Actual vs Predicted (test)
        plt.subplot(233)
        plt.scatter(y_test, y_pred_test, alpha=0.5, label='Custom Tree', color='red')
        if y_pred_sklearn_test is not None:
            plt.scatter(y_test, y_pred_sklearn_test, alpha=0.5, 
                       label='Sklearn', color='green')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'k--', linewidth=2, label='Perfect')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.title('Actual vs Predicted (Test Set)')
        plt.grid(True, alpha=0.3)
        
        # Bottom row: Train set
        # Custom tree
        plt.subplot(234)
        plt.scatter(X_train[sort_idx_train], y_train[sort_idx_train], 
                   alpha=0.3, label='Actual', s=10, color='lightcoral')
        plt.plot(X_train[sort_idx_train], y_pred_train[sort_idx_train], 
                'r-', linewidth=2, label='Custom Tree')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.title('Custom Tree (Train Set)')
        plt.grid(True, alpha=0.3)
        
        # Sklearn tree
        if y_pred_sklearn_train is not None:
            plt.subplot(235)
            plt.scatter(X_train[sort_idx_train], y_train[sort_idx_train], 
                       alpha=0.3, label='Actual', s=10, color='lightcoral')
            plt.plot(X_train[sort_idx_train], y_pred_sklearn_train[sort_idx_train], 
                    'g-', linewidth=2, label='Sklearn')
            plt.xlabel('X')
            plt.ylabel('y')
            plt.legend()
            plt.title('Sklearn Tree (Train Set)')
            plt.grid(True, alpha=0.3)
        
        # Actual vs Predicted (train)
        plt.subplot(236)
        plt.scatter(y_train, y_pred_train, alpha=0.3, 
                   label='Custom Tree', color='red', s=10)
        if y_pred_sklearn_train is not None:
            plt.scatter(y_train, y_pred_sklearn_train, alpha=0.3, 
                       label='Sklearn', color='green', s=10)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                'k--', linewidth=2, label='Perfect')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.title('Actual vs Predicted (Train Set)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_nd_results(self, y_test, y_pred, y_pred_sklearn, title):
        """Plot results for multi-dimensional data"""
        plt.figure(figsize=(12, 4))
        
        # Custom tree
        plt.subplot(131)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{title}\nCustom Tree')
        plt.grid(True, alpha=0.3)
        
        # Sklearn tree
        if y_pred_sklearn is not None:
            plt.subplot(132)
            plt.scatter(y_test, y_pred_sklearn, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', linewidth=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Sklearn Tree')
            plt.grid(True, alpha=0.3)
        
        # Residuals
        plt.subplot(133)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """Print summary of all tests"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        for result in self.results:
            print(f"\n{result['name']}:")
            print(f"  R²: {result['r2']:.4f}")
            print(f"  MSE: {result['mse']:.4f}")
            if 'sklearn_r2' in result:
                print(f"  vs Sklearn R²: {result['sklearn_r2']:.4f} "
                      f"(diff: {result['r2'] - result['sklearn_r2']:.4f})")


def run_all_tests(tree_class):
    """Run complete test suite"""
    tester = RegressionTreeTester(tree_class)
    
    print("\n" + "=" * 60)
    print("REGRESSION TREE TESTING SUITE")
    print("=" * 60 + "\n")
    
    # Test 1: Linear data
    try:
        X, y = TestData.linear_data()
        tester.run_test("Linear Relationship", X, y, max_depth=5, min_size=5)
    except Exception as e:
        print(f"❌ Linear test failed: {e}\n")
    
    # Test 2: Sine wave
    try:
        X, y = TestData.sine_wave()
        tester.run_test("Sine Wave (Non-linear)", X, y, max_depth=8, min_size=5)
    except Exception as e:
        print(f"❌ Sine wave test failed: {e}\n")
    
    # Test 3: Quadratic
    try:
        X, y = TestData.quadratic_data()
        tester.run_test("Quadratic Function", X, y, max_depth=6, min_size=5)
    except Exception as e:
        print(f"❌ Quadratic test failed: {e}\n")
    
    # Test 4: Step function
    try:
        X, y = TestData.step_function()
        tester.run_test("Step Function (Perfect for Trees)", X, y, 
                       max_depth=4, min_size=5)
    except Exception as e:
        print(f"❌ Step function test failed: {e}\n")
    
    # Test 5: Multi-dimensional
    try:
        X, y = TestData.multidimensional_data()
        tester.run_test("Multi-dimensional (5 features)", X, y, 
                       max_depth=6, min_size=10, plot=True)
    except Exception as e:
        print(f"❌ Multi-dimensional test failed: {e}\n")
    
    # Test 6: Pandas DataFrame
    try:
        df = TestData.pandas_dataframe()
        X = df.drop(columns=['loan_amount']).values
        y = df['loan_amount'].values
        tester.run_test("Pandas DataFrame Input", X, y, 
                       max_depth=5, min_size=10, plot=False)
    except Exception as e:
        print(f"❌ Pandas test failed: {e}\n")
    
    # Test 7: Edge case - constant target
    try:
        X, y = TestData.constant_target()
        result = tester.run_test("Edge Case: Constant Target", X, y, 
                                max_depth=3, min_size=5, plot=False)
        if result['mse'] < 1e-10:
            print("✓ Constant target handled correctly (MSE ≈ 0)")
        else:
            print("⚠ Warning: MSE should be near 0 for constant target")
    except Exception as e:
        print(f"❌ Constant target test failed: {e}\n")
    
    # Test 8: Edge case - single feature
    try:
        X, y = TestData.single_feature()
        tester.run_test("Edge Case: Single Feature", X, y, 
                       max_depth=5, min_size=5, plot=False)
    except Exception as e:
        print(f"❌ Single feature test failed: {e}\n")
    
    # Print summary
    tester.print_summary()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


# Example usage
if __name__ == "__main__":
    
    from regtree import regtree as RegressionTree
    run_all_tests(RegressionTree)
    
    print("Import Custom RegressionTree class and run:")
    print("  run_all_tests(RegressionTree)")
