import pandas as pd
import os
from pathlib import Path

preprocessed_path = Path(__file__).parent.parent / "data" / "preprocessed_data.xlsx"
biomarkers_path = Path(__file__).parent.parent / "data" / "bibliography_biomarkers.xlsx"
output_path = Path(__file__).parent.parent / "data" / "pruned_preprocessed_data.xlsx"

if not os.path.exists(preprocessed_path):
    print(f"Error: Preprocessed data file not found at {preprocessed_path}")
    exit()

if not os.path.exists(biomarkers_path):
    print(f"Error: Biomarkers file not found at {biomarkers_path}")
    exit()

preprocessed_data = pd.read_excel(preprocessed_path, header=None)
biomarkers_data = pd.read_excel(biomarkers_path, header=None)

class_labels = preprocessed_data.iloc[0, :]

biomarker_miRNAs = biomarkers_data.iloc[:, 0].dropna().astype(str).tolist()

filtered_data = preprocessed_data[preprocessed_data.iloc[:, 0].astype(str).isin(biomarker_miRNAs)]
filtered_data = pd.concat([class_labels.to_frame().T, filtered_data], ignore_index=True)

filtered_data.to_excel(output_path, index=False, header=False)
print(f"Pruned data saved to '{output_path}'.")

