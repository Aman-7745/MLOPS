import pandas as pd
import numpy as np

df = pd.read_csv('./data/features/train_bow.csv')

print("Shape of data:", df.shape)
print("Columns:", df.columns.tolist())
print("Last 5 rows:")
print(df.tail())

# Check class distribution
y_train = df.iloc[:, -1].values
print("\nUnique classes in y_train:", np.unique(y_train, return_counts=True))
