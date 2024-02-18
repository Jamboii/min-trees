# Simple Saplings

This repository contains some from-scratch builds of decision-tree-based models, starting from the standard Decision Tree, working up to XGBoost (and potentially beyond). For the sake of educational purposes, the implementations of each model will be as close to Minimum Viable Product (MVP) as possible.

As of now, these are the different tree models implemented:
- [Decision Tree Classifier](/decision_tree.py)
- [Decision Tree Regressor](/decision_tree.py)
- [Random Forest Classifier](/random_forest.py)
- [AdaBoost Binary Classifier](/adaboost.py)
- [Gradient Boosted Decision Tree Classifier](/gb_decision_tree.py)
- [XGBoost Classifier](/xgboost.py)

## TODO Items
- AdaBoost
    - Allow for multiclass classification
    - Fix division-by-zero runtime warning
- Optimize Decision Tree feature selection to use a sorted list like XGBoost
- Create regression models for:
    - Random Forest
    - Adaboost
    - Gradient Boost
    - XGBoost
- Potentially implement some of the presented optimizations for XGBoost (e.g. Weighted Quantile Sketch)
- Create notebooks walking through the implementations of each model
- More tree-based model implementations (if they're not *too* complicated)
    - LightGBM
    - CatBoost