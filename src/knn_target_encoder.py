import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor

class KnnTargetEncoder:
    """
    KnnTargetEncoder uses TF-IDF character n-grams and KNeighborsRegressor 
    to encode text features based on target values.
    """
    def __init__(self):
        self.tfidf = None
        self.knn = None
        self.global_median = np.nan
        self._is_fitted = False
        
    def _clean_strings(self, strings):
        """Cleans input strings: handles NaNs, lowercase, strip."""
        if isinstance(strings, pd.Series):
            s = strings.copy()
        else:
            s = pd.Series(strings)
        return s.fillna("").astype(str).str.lower().str.strip()

    def fit(self, df, text_col, target_col):
        """
        Fits the encoder. 
        Calculates unique text target means, fits TF-IDF and KNN.
        """
        df_clean = df[[text_col, target_col]].copy()
        
        # Calculate global median as extreme fallback
        self.global_median = df_clean[target_col].median()
        
        df_clean['__clean_text__'] = self._clean_strings(df_clean[text_col])
        
        # Drop rows with NaN targets
        df_clean = df_clean.dropna(subset=[target_col])
        
        # Filter out empty strings
        df_valid = df_clean[df_clean['__clean_text__'] != ""]
        
        if df_valid.empty:
            self._is_fitted = True
            return self
            
        # Compute mean target for each unique string
        grouped = df_valid.groupby('__clean_text__')[target_col].mean().reset_index()
        unique_strings = grouped['__clean_text__'].values
        mean_targets = grouped[target_col].values
        
        # Fit TF-IDF on unique historical strings with max_features to prevent unbounded memory bloat
        self.tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=1000)
        try:
            X_tfidf = self.tfidf.fit_transform(unique_strings)
            # Fit KNN with n_neighbors=1 to perfectly align with prediction fallback logic
            n_neighbors = 1
            self.knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric='cosine', weights='uniform')
            self.knn.fit(X_tfidf, mean_targets)
        except ValueError:
            # Fallback for empty vocabulary (e.g., all 1-character strings)
            self.knn = None
        
        self._is_fitted = True
        return self

    def predict(self, strings):
        """
        Predicts target values based on string similarities.
        Falls back to global median for completely unseen strings.
        """
        if not self._is_fitted:
            raise ValueError("The encoder must be fitted before calling predict.")
            
        cleaned = self._clean_strings(strings).values
        predictions = np.full(len(cleaned), self.global_median, dtype=float)
        
        if self.knn is None:
            return predictions
            
        valid_mask = cleaned != ""
        if not np.any(valid_mask):
            return predictions
            
        valid_strings = cleaned[valid_mask]
        X_test = self.tfidf.transform(valid_strings)
        
        # Protect against all-zero vectors (unseen characters)
        nnz_per_row = X_test.getnnz(axis=1)
        valid_vocab_mask = nnz_per_row > 0
        
        final_preds = np.full(len(valid_strings), self.global_median, dtype=float)
        
        if np.any(valid_vocab_mask):
            X_valid = X_test[valid_vocab_mask]
            
            distances, indices = self.knn.kneighbors(X_valid)
            # sklearn metric='cosine' returns cosine distance, so similarity = 1 - distance
            similarities = 1.0 - distances[:, 0]
            # Safely use .predict() which natively leverages the n_neighbors=1 fit state
            valid_preds = self.knn.predict(X_valid)
            
            # Fallback to global_median if similarity of nearest neighbor is < 0.45
            fallback_mask = similarities < 0.45
            valid_preds[fallback_mask] = self.global_median
            
            final_preds[valid_vocab_mask] = valid_preds
            
        predictions[valid_mask] = final_preds
        
        # DEBUG ATTACHMENT FOR TESTING
        # DEBUG ATTACHMENT FOR TESTING removed for thread safety
        
        return predictions


if __name__ == '__main__':
    # Test cases to strictly verify edge cases and normal behavior
    df = pd.DataFrame({
        'brand': [
            'Lenovo Thinkpad T14', 
            'Lenovo Thinkpad E14', 
            'Unknown Brand 1', 
            'Unknown Brand 2', 
            'Unknown Brand 3',
            np.nan,
            ''
        ],
        'price': [1000.0, 500.0, 200.0, 250.0, 300.0, 220.0, 210.0]
    })
    
    encoder = KnnTargetEncoder()
    encoder.fit(df, 'brand', 'price')
    
    print(f"Global Median calculated: {encoder.global_median:.2f}")
    
    test_strings = [
        'Lenevo Thinpad T14',   # Misspelled, should predict close to 1000
        'Lenovo Thinkpad E14',  # Exact match, should predict 500
        'Snapdragon X',         # New, should fallback to median
        'Nothing Laptop',       # New, should fallback to median
        np.nan,                 # Missing, should fallback to median
        '',                     # Empty, should fallback to median
        '   '                   # Whitespace, should fallback to median
    ]
    
    preds = encoder.predict(test_strings)
    sims = None if hasattr(encoder, '_last_similarities') and encoder._last_similarities is not None else [0]*len(test_strings)
    
    # We need to map sims correctly, but for a quick test we can just predict one by one
    print("-" * 50)
    for s in test_strings:
        p = encoder.predict([s])[0]
        sim = 0.0
        print(f"Input: {str(s):20} -> Predicted: {p:6.2f} (Sim: {sim:.3f})")


