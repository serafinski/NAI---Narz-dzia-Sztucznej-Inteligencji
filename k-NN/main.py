import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the file using Pandas
df = pd.read_csv('iris.test.data', header=None)

# Use iloc to select the first four columns
df = df.iloc[:, :4]

# Convert the data to a 2D array
data = df.values.tolist()

# Print the resulting 2D array
print(data)