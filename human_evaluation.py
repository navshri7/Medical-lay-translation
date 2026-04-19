import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
import krippendorff   




df = pd.read_csv("ratings.tsv", sep="\t")

print("=== AVERAGES ===")
print(df.mean())
print("\n")








def df_to_fleiss_format(series):
    """
    Convert a list of ratings (e.g., all evaluators' ratings for model1_fluency)
    into a Fleiss-kappa-compatible matrix: one row per item, column j = count of rating j.
    Ratings assumed to be integers.
    """
    categories = sorted(series.unique())
    items = pd.DataFrame({cat: (series == cat).astype(int) for cat in categories})
    return items.values

print("=== FLEISS' KAPPA ===")
for col in df.columns:
    matrix = df_to_fleiss_format(df[col])
    kappa = fleiss_kappa(matrix)
    print(f"{col}: {kappa}")
print("\n")





print("=== KRIPPENDORFF ALPHA (ORDINAL) ===")



data_matrix = df.T.values

alpha = krippendorff.alpha(reliability_data=data_matrix, level_of_measurement='ordinal')
print("Krippendorff Alpha (all columns together):", alpha)


for col in df.columns:
    col_data = np.array([df[col].values])  
    col_data = df[col].values.reshape(1, -1)
    a = krippendorff.alpha(reliability_data=col_data, level_of_measurement='ordinal')
    print(f"{col}: {a}")
print("\n")





print("=== PERCENT AGREEMENT ===")

def percent_agreement(series):
    """
    Computes percentage of annotators giving the *most common response*.
    """
    counts = series.value_counts()
    return counts.max() / counts.sum()

for col in df.columns:
    print(f"{col}: {percent_agreement(df[col]):.3f}")
