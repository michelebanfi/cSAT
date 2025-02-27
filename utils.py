import pandas as pd
import numpy as np
from causallearn.graph.Edge import Edge

# Initialize a counter for variable IDs
next_var_id = 1

# create causal dataframe
def basic_causal_dataframe() -> pd.DataFrame:
    X = np.random.uniform(size=1000)
    eps = np.random.normal(size=1000)
    delta = np.random.uniform(size=1000)
    Y = -7 * X + 0.5 * delta
    Z = 2 * X + Y + eps

    # Create DataFrame with named variables
    return pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})