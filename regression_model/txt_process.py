import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

class ExperimentalEnvironmentProcessor:
    def __init__(self, max_features=None):

        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.fitted = False
        self.feature_dim = 0
        
    def fit(self, text_data):

        self.vectorizer.fit(text_data)
        self.fitted = True
        self.feature_dim = len(self.vectorizer.get_feature_names_out())
        print(f"Fitted vectorizer with {self.feature_dim} features")
        return self.feature_dim
        
    def transform(self, text_data):

        if not self.fitted:
            raise ValueError("Vectorizer must be fitted before transform")
            

        features = self.vectorizer.transform(text_data).toarray()
        return torch.tensor(features, dtype=torch.float32)
    
    def fit_transform(self, text_data):

        self.fit(text_data)
        return self.transform(text_data)
    
    def get_feature_dim(self):

        if not self.fitted:
            raise ValueError("Vectorizer must be fitted before getting feature dimension")
        return self.feature_dim
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'feature_dim': self.feature_dim,
                'max_features': self.max_features
            }, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                self.vectorizer = data['vectorizer']
                self.feature_dim = data['feature_dim']
                self.max_features = data['max_features']
            else:
                self.vectorizer = data
                self.feature_dim = len(self.vectorizer.get_feature_names_out())
            
        self.fitted = True

def load_experimental_data(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]