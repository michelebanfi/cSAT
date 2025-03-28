import numpy as np
import pandas as pd
import numpy.random as npr

def generate_mixed_causal_dataframe(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate a synthetic causal dataset with mixed variable types
    
    Variable types:
    1. Continuous (X)
    2. Binary (A)
    3. Categorical (C)
    4. Ordinal Categorical (O)
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Continuous variable generation
    X = np.random.uniform(low=0, high=10, size=n_samples)
    
    # 2. Binary variable generation (Bernoulli distribution)
    # Probability of 1 depends on X (causal relationship)
    A = np.random.binomial(n=1, p=1 / (1 + np.exp(-0.5 * (X - 5))))
    
    # 3. Categorical variable generation 
    # Use multinomial distribution with probabilities influenced by X
    def generate_categorical(X, categories=3):
        # Create probabilities based on X
        base_probs = np.ones(categories) / categories
        
        # Calculate the adjustment factor for each sample
        adjustment_factor = 0.2 * (X - np.mean(X)) / np.std(X)
        
        # Reshape adjustment_factor to allow broadcasting
        adjustment_factor = adjustment_factor.reshape(-1, 1)  # Shape: (n_samples, 1)
        
        # Adjust probabilities for all samples at once (broadcasts to shape: n_samples, categories)
        adjusted_probs = base_probs * (1 + adjustment_factor)
        
        # Normalize each row to sum to 1
        row_sums = adjusted_probs.sum(axis=1, keepdims=True)
        adjusted_probs = adjusted_probs / row_sums
        
        # Generate categorical values for each sample based on its probability distribution
        cat_values = np.array([npr.choice(categories, p=probs) for probs in adjusted_probs])
        
        return cat_values
    
    C = generate_categorical(X)
    
    # 4. Ordinal Categorical variable generation
    # Ordered categories with dependency on X
    def generate_ordinal(X, categories=4):
        # Create an ordinal scale influenced by X
        ordinal_thresholds = np.linspace(X.min(), X.max(), categories+1)[1:-1]
        ordinal_values = np.zeros_like(X, dtype=int)
        
        for i in range(1, categories):
            ordinal_values[X > ordinal_thresholds[i-1]] = i
        
        return ordinal_values
    
    O = generate_ordinal(X)
    
    # 5. Dependent variable generation with mixed inputs
    Y = (
        -2 * X +  # Continuous influence
        1.5 * A +  # Binary influence
        0.5 * C +  # Categorical influence
        0.3 * O +  # Ordinal influence
        np.random.normal(0, 0.5, size=n_samples)  # Noise
    )
    
    # Create DataFrame
    return pd.DataFrame({
        'X': X,   # Continuous
        'A': A,   # Binary
        'C': C,   # Categorical
        'O': O,   # Ordinal Categorical
        'Y': Y    # Dependent variable
    })

# Example usage
def demonstrate_mixed_variable_causal_data():
    # Generate the dataset
    df = generate_mixed_causal_dataframe()
    
    # Display basic information
    print("Dataset Information:")
    print(df.info())
    
    # Display the first few rows
    print("\nFirst few rows:")
    print(df.head())
    
    # Basic statistical summary
    print("\nDescriptive Statistics:")
    print(df.describe(include='all'))

# Run the demonstration
demonstrate_mixed_variable_causal_data()