# MDAV-kA
A ready-to-use function for k-anonymity clustering use Maximum Distance to Average Vector (MDAV) microaggregation techniques

Currently, only function to use with Jupyter Notebook is available. Command line tool will be developed shortly after.

Examples to use with jupyter notebook

```
import pandas as pd
from MDAV_kA import anonymised_dataset

# Read original file
dataset = pd.read_csv('dataset_file.csv')

# Define number of k in k_anonymity
K = 5

# Define numerical QASI-identifier
NUM_QUASI = ['Age']

# Run function, will return anonymised dataset

anonymised_df = anonymised_dataset(dataset, k=K)

print(anonymised_df)
```
