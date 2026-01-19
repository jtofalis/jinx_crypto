"""
Pure Python LightGBM inference - no dependencies on lightgbm/onnx/sklearn
Extracted from the trained model for mobile deployment
"""
import numpy as np
import json

class LightGBMInference:
    """Lightweight LightGBM inference using exported model parameters"""
    
    def __init__(self, model_file, selector_file=None):
        with open(model_file, 'r') as f:
            self.model_data = json.load(f)
        self.n_classes = 2
        self.threshold = 0.56
        
        # Load feature selector if provided
        self.feature_mask = None
        if selector_file:
            with open(selector_file, 'r') as f:
                selector_data = json.load(f)
            self.feature_mask = np.array(selector_data['feature_mask'])
    
    def select_features(self, X):
        """Apply feature selection if selector loaded"""
        if self.feature_mask is not None:
            if len(X.shape) == 1:
                return X[self.feature_mask]
            else:
                return X[:, self.feature_mask]
        return X
    
    def predict_proba(self, X):
        """
        Predict probabilities using tree ensemble
        X: numpy array of shape (n_samples, n_features)
        Returns: array of shape (n_samples, 2) with [prob_class_0, prob_class_1]
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Apply feature selection
        X = self.select_features(X)
        
        n_samples = X.shape[0]
        raw_scores = np.zeros((n_samples, self.n_classes))
        
        # Sum predictions from all trees
        for tree in self.model_data['trees']:
            for i in range(n_samples):
                class_idx = tree['class']
                raw_scores[i, class_idx] += self._predict_tree(tree, X[i])
        
        # Convert to probabilities using sigmoid
        probas = np.zeros((n_samples, self.n_classes))
        for i in range(n_samples):
            probas[i] = self._sigmoid(raw_scores[i])
        
        return probas
    
    def _predict_tree(self, tree, x):
        """Navigate decision tree and return leaf value"""
        node = tree['root']
        while 'split_feature' in node:
            if x[node['split_feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['value']
    
    def _sigmoid(self, scores):
        """Convert raw scores to probabilities"""
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()
