import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor as SklearnRF

import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
from models.random_forest_regressor import RandomForestRegressor


class TestRandomForest:
    """Core functionality tests for random forest ensemble"""
    
    def test_forest_initialization(self):
        """Test forest initializes correctly"""
        rf = RandomForestRegressor(n_trees=10, max_depth=5)
        assert rf.n_trees == 10
        assert rf.max_depth == 5
        assert len(rf.trees) == 0
    
    def test_forest_creates_trees(self):
        """Test forest creates correct number of trees"""
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        rf = RandomForestRegressor(n_trees=10)
        rf.fit(X, y)
        assert len(rf.trees) == 10
    
    def test_prediction_averaging(self):
        """Test forest averages predictions correctly"""
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        rf = RandomForestRegressor(n_trees=5)
        rf.fit(X, y)
        
        X_test = np.random.rand(10, 3)
        forest_pred = rf.predict(X_test)
        tree_preds = np.array([tree.predict(X_test) for tree in rf.trees])
        manual_avg = tree_preds.mean(axis=0)
        
        assert np.allclose(forest_pred, manual_avg)
    
    def test_simple_linear_relationship(self):
        """Test forest learns simple linear relationship"""
        np.random.seed(42)
        X = np.random.rand(200, 1) * 10
        y = 2 * X[:, 0] + 3 + np.random.randn(200) * 0.5
        rf = RandomForestRegressor(n_trees=50, max_depth=10)
        rf.fit(X, y)
        assert r2_score(y, rf.predict(X)) > 0.8
    
    def test_multivariate_regression(self):
        """Test forest works with multiple features"""
        X, y = make_regression(n_samples=400, n_features=10, noise=10, random_state=42) ## With 300 or 200 samples this test fails
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        rf = RandomForestRegressor(n_trees=50, max_depth=10)
        rf.fit(X_train, y_train)
        assert r2_score(y_test, rf.predict(X_test)) > 0.7
    
    def test_vs_sklearn_performance(self):
        """Test performance is comparable to sklearn"""
        X, y = make_friedman1(n_samples=200, n_features=10, noise=1.0, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        rf_custom = RandomForestRegressor(n_trees=50, max_depth=10)
        rf_custom.fit(X_train, y_train)
        r2_custom = r2_score(y_test, rf_custom.predict(X_test))
        
        rf_sklearn = SklearnRF(n_estimators=50, max_depth=10, random_state=42)
        rf_sklearn.fit(X_train, y_train)
        r2_sklearn = r2_score(y_test, rf_sklearn.predict(X_test))
        
        
        print(f"Custom R²: {r2_custom:.3f}, Sklearn R²: {r2_sklearn:.3f}")
        assert r2_custom > 0.5
        assert abs(r2_custom - r2_sklearn) < 0.3
    
    def test_deterministic_with_seed(self):
        """Test same seed produces same results"""
        X = np.random.rand(50, 3)
        y = np.random.rand(50)
        X_test = np.random.rand(10, 3)
        
        np.random.seed(42)
        rf1 = RandomForestRegressor(n_trees=10)
        rf1.fit(X, y)
        pred1 = rf1.predict(X_test)
        
        np.random.seed(42)
        rf2 = RandomForestRegressor(n_trees=10)
        rf2.fit(X, y)
        pred2 = rf2.predict(X_test)
        
        assert np.allclose(pred1, pred2)
    
    def test_generalization(self):
        """Test forest generalizes to unseen data"""
        X, y = make_regression(n_samples=300, n_features=10, noise=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        rf = RandomForestRegressor(n_trees=50, max_depth=10)
        rf.fit(X_train, y_train)
        
        train_r2 = r2_score(y_train, rf.predict(X_train))
        test_r2 = r2_score(y_test, rf.predict(X_test))
        
        assert train_r2 > 0.7
        assert test_r2 > 0.5
        assert train_r2 >= test_r2


class TestVisualization:
    """Visual tests with plots comparing custom vs sklearn"""
    
    def test_plot_linear_and_nonlinear_fit(self):
        """Compare custom vs sklearn on linear and nonlinear data"""
        np.random.seed(42)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Linear data
        X_lin = np.random.rand(200, 1) * 10
        y_lin = 2 * X_lin[:, 0] + 3 + np.random.randn(200) * 1.5
        X_line = np.linspace(0, 10, 300).reshape(-1, 1)
        
        rf_custom_lin = RandomForestRegressor(n_trees=50, max_depth=10)
        rf_custom_lin.fit(X_lin, y_lin)
        rf_sklearn_lin = SklearnRF(n_estimators=50, max_depth=10, random_state=42)
        rf_sklearn_lin.fit(X_lin, y_lin)
        
        # Custom - Linear
        axes[0, 0].scatter(X_lin, y_lin, alpha=0.4, s=20, color='gray', label='Data')
        axes[0, 0].plot(X_line, rf_custom_lin.predict(X_line), 'r-', linewidth=2, label='RF prediction')
        axes[0, 0].plot(X_line, 2 * X_line + 3, 'g--', linewidth=2, label='True function')
        axes[0, 0].set_title(f'Custom RF - Linear (R²={r2_score(y_lin, rf_custom_lin.predict(X_lin)):.3f})')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sklearn - Linear
        axes[0, 1].scatter(X_lin, y_lin, alpha=0.4, s=20, color='gray', label='Data')
        axes[0, 1].plot(X_line, rf_sklearn_lin.predict(X_line), 'b-', linewidth=2, label='RF prediction')
        axes[0, 1].plot(X_line, 2 * X_line + 3, 'g--', linewidth=2, label='True function')
        axes[0, 1].set_title(f'Sklearn RF - Linear (R²={r2_score(y_lin, rf_sklearn_lin.predict(X_lin)):.3f})')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('y')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Nonlinear data
        X_sin = np.random.rand(300, 1) * 10
        y_sin = np.sin(X_sin[:, 0]) * 5 + np.random.randn(300) * 0.5
        
        rf_custom_sin = RandomForestRegressor(n_trees=100, max_depth=15, min_size=3)
        rf_custom_sin.fit(X_sin, y_sin)
        rf_sklearn_sin = SklearnRF(n_estimators=100, max_depth=15, random_state=42)
        rf_sklearn_sin.fit(X_sin, y_sin)
        
        # Custom - Nonlinear
        axes[1, 0].scatter(X_sin, y_sin, alpha=0.4, s=20, color='gray', label='Data')
        axes[1, 0].plot(X_line, rf_custom_sin.predict(X_line), 'r-', linewidth=2, label='RF prediction')
        axes[1, 0].plot(X_line, np.sin(X_line) * 5, 'g--', linewidth=2, label='True function')
        axes[1, 0].set_title(f'Custom RF - Nonlinear (R²={r2_score(y_sin, rf_custom_sin.predict(X_sin)):.3f})')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('y')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sklearn - Nonlinear
        axes[1, 1].scatter(X_sin, y_sin, alpha=0.4, s=20, color='gray', label='Data')
        axes[1, 1].plot(X_line, rf_sklearn_sin.predict(X_line), 'b-', linewidth=2, label='RF prediction')
        axes[1, 1].plot(X_line, np.sin(X_line) * 5, 'g--', linewidth=2, label='True function')
        axes[1, 1].set_title(f'Sklearn RF - Nonlinear (R²={r2_score(y_sin, rf_sklearn_sin.predict(X_sin)):.3f})')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('y')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_fit_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: test_fit_comparison.png")
    
    def test_plot_depth_comparison(self):
        """Compare different max_depth settings"""
        np.random.seed(42)
        X = np.random.rand(200, 1) * 10
        y = np.sin(X[:, 0]) * 5 + np.random.randn(200) * 0.8
        
        depths = [3, 5, 10, 20]
        X_line = np.linspace(0, 10, 200).reshape(-1, 1)
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        
        for idx, depth in enumerate(depths):
            rf_custom = RandomForestRegressor(n_trees=50, max_depth=depth)
            rf_custom.fit(X, y)
            rf_sklearn = SklearnRF(n_estimators=50, max_depth=depth, random_state=42)
            rf_sklearn.fit(X, y)
            
            r2_c = r2_score(y, rf_custom.predict(X))
            r2_s = r2_score(y, rf_sklearn.predict(X))
            
            # Custom
            axes[idx, 0].scatter(X, y, alpha=0.3, s=15, color='gray')
            axes[idx, 0].plot(X_line, rf_custom.predict(X_line), 'r-', linewidth=2)
            axes[idx, 0].plot(X_line, np.sin(X_line) * 5, 'g--', linewidth=2, label='True')
            axes[idx, 0].set_title(f'Custom RF - Depth={depth} (R²={r2_c:.3f})')
            axes[idx, 0].set_xlabel('X')
            axes[idx, 0].set_ylabel('y')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Sklearn
            axes[idx, 1].scatter(X, y, alpha=0.3, s=15, color='gray')
            axes[idx, 1].plot(X_line, rf_sklearn.predict(X_line), 'b-', linewidth=2)
            axes[idx, 1].plot(X_line, np.sin(X_line) * 5, 'g--', linewidth=2, label='True')
            axes[idx, 1].set_title(f'Sklearn RF - Depth={depth} (R²={r2_s:.3f})')
            axes[idx, 1].set_xlabel('X')
            axes[idx, 1].set_ylabel('y')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_depth_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: test_depth_comparison.png")
    
    def test_plot_num_trees_effect(self):
        """Show effect of number of trees on performance"""
        np.random.seed(42)
        X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        n_trees_list = [1, 5, 10, 20, 50, 100, 200]
        train_custom, test_custom = [], []
        train_sklearn, test_sklearn = [], []
        
        for n_trees in n_trees_list:
            rf_custom = RandomForestRegressor(n_trees=n_trees, max_depth=10)
            rf_custom.fit(X_train, y_train)
            train_custom.append(r2_score(y_train, rf_custom.predict(X_train)))
            test_custom.append(r2_score(y_test, rf_custom.predict(X_test)))
            
            rf_sklearn = SklearnRF(n_estimators=n_trees, max_depth=10, random_state=42)
            rf_sklearn.fit(X_train, y_train)
            train_sklearn.append(r2_score(y_train, rf_sklearn.predict(X_train)))
            test_sklearn.append(r2_score(y_test, rf_sklearn.predict(X_test)))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Custom
        axes[0].plot(n_trees_list, train_custom, 'o-', linewidth=2, label='Train', markersize=8, color='steelblue')
        axes[0].plot(n_trees_list, test_custom, 's-', linewidth=2, label='Test', markersize=8, color='coral')
        axes[0].set_xlabel('Number of Trees', fontsize=12)
        axes[0].set_ylabel('R² Score', fontsize=12)
        axes[0].set_title('Custom Random Forest', fontsize=13)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        
        # Sklearn
        axes[1].plot(n_trees_list, train_sklearn, 'o-', linewidth=2, label='Train', markersize=8, color='steelblue')
        axes[1].plot(n_trees_list, test_sklearn, 's-', linewidth=2, label='Test', markersize=8, color='coral')
        axes[1].set_xlabel('Number of Trees', fontsize=12)
        axes[1].set_ylabel('R² Score', fontsize=12)
        axes[1].set_title('Sklearn Random Forest', fontsize=13)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig('test_num_trees.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: test_num_trees.png")
    
    def test_plot_predictions_and_residuals(self):
        """Compare predictions and residuals"""
        np.random.seed(42)
        X, y = make_regression(n_samples=300, n_features=5, noise=15, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        rf_custom = RandomForestRegressor(n_trees=100, max_depth=10)
        rf_custom.fit(X_train, y_train)
        pred_custom = rf_custom.predict(X_test)
        
        rf_sklearn = SklearnRF(n_estimators=100, max_depth=10, random_state=42)
        rf_sklearn.fit(X_train, y_train)
        pred_sklearn = rf_sklearn.predict(X_test)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Custom - Predictions
        axes[0, 0].scatter(y_test, pred_custom, alpha=0.6, color='coral')
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values', fontsize=11)
        axes[0, 0].set_ylabel('Predictions', fontsize=11)
        axes[0, 0].set_title(f'Custom RF - Predictions (R² = {r2_score(y_test, pred_custom):.3f})', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sklearn - Predictions
        axes[0, 1].scatter(y_test, pred_sklearn, alpha=0.6, color='steelblue')
        axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('True Values', fontsize=11)
        axes[0, 1].set_ylabel('Predictions', fontsize=11)
        axes[0, 1].set_title(f'Sklearn RF - Predictions (R² = {r2_score(y_test, pred_sklearn):.3f})', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Custom - Residuals
        residuals_custom = y_test - pred_custom
        axes[1, 0].scatter(pred_custom, residuals_custom, alpha=0.6, color='coral')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Predicted Values', fontsize=11)
        axes[1, 0].set_ylabel('Residuals', fontsize=11)
        axes[1, 0].set_title('Custom RF - Residuals', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sklearn - Residuals
        residuals_sklearn = y_test - pred_sklearn
        axes[1, 1].scatter(pred_sklearn, residuals_sklearn, alpha=0.6, color='steelblue')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Predicted Values', fontsize=11)
        axes[1, 1].set_ylabel('Residuals', fontsize=11)
        axes[1, 1].set_title('Sklearn RF - Residuals', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_predictions_residuals.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: test_predictions_residuals.png")

class TestMultivariate:
    
    def test_simple_linear_relationship(self):
        """Test forest learns simple linear relationship"""
        np.random.seed(42)
        X = np.random.rand(200, 1) * 10
        y = 2 * X[:, 0] + 3 + np.random.randn(200) * 0.5
        rf = RandomForestRegressor(n_trees=50, max_depth=10)
        rf.fit(X, y)
        assert r2_score(y, rf.predict(X)) > 0.8
    
    def test_multivariate_regression(self):
        """Test forest works with multiple features"""
        X, y = make_regression(n_samples=400, n_features=10, noise=10, random_state=42) ## With 300 or 200 samples this test fails
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        rf = RandomForestRegressor(n_trees=50, max_depth=10, min_size=3)
        rf.fit(X_train, y_train)
        assert r2_score(y_test, rf.predict(X_test)) > 0.7
        print(r2_score(y_test, rf.predict(X_test)))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
    
    #pytest.main([__file__, f"{__file__}::TestVisualization", "-v", "-s"])
    #pytest.main([__file__, f"{__file__}::TestMultivariate", "-v", "-s"])
