import csv
import pandas as pd
import os
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
output_path = DATA / "preprocessed_data.xlsx"

# Platform annotation GPL21572, mapping probe IDs to miRNA names. The GEO SOFT
# .txt is read directly; a converted .xlsx is also accepted.
mapper_path = next(
    (DATA / n for n in ("GPL21572.txt", "GPL21572-124634.xlsx")
     if (DATA / n).exists()),
    DATA / "GPL21572.txt",
)

# The series matrix is taken as downloaded from GEO (.txt, tab-separated) if it is
# there, otherwise as a converted .xlsx. Reading the .txt directly is preferred:
# it removes a manual conversion step for a ~200 MB file. Both give the same table,
# verified to yield 299 samples (158 sepsis / 141 non-sepsis).
original_data_path = next(
    (DATA / n for n in ("GSE134358_series_matrix.txt", "GSE134358_series_matrix.xlsx")
     if (DATA / n).exists()),
    DATA / "GSE134358_series_matrix.txt",
)

if not os.path.exists(mapper_path):
    print(f"Error: Mapper file not found at {mapper_path}")
    exit()

if not os.path.exists(original_data_path):
    print(f"Error: Series matrix not found. Expected one of "
          f"GSE134358_series_matrix.txt / .xlsx in {DATA}")
    exit()

if str(mapper_path).lower().endswith(".txt"):
    # GEO SOFT layout: 19 lines of metadata, then the platform table, then a
    # !platform_table_end marker. Columns are taken by name rather than position.
    df_mapper = pd.read_csv(mapper_path, sep="\t", skiprows=19,
                            usecols=["ID", "miRNA_ID"], dtype=str)
    df_mapper = df_mapper[~df_mapper["ID"].astype(str).str.startswith("!")]
else:
    df_mapper = pd.read_excel(mapper_path, usecols=[0, 15])

df_mapper.columns = ["ID", "miRNA_id"]
df_mapper["ID"] = df_mapper["ID"].astype(str).str.strip()

mapping_dict = df_mapper.set_index("ID")["miRNA_id"].to_dict()

def load_series_matrix(file_path):
    """The "disease state" row followed by the expression table, as one frame.

    Row 37 carries the disease-state characteristics and row 63 onwards the
    expression table. These are absolute positions in the series matrix as
    distributed by GEO; check them if the file is ever re-issued.
    """
    if not str(file_path).lower().endswith(".txt"):
        df = pd.read_excel(file_path, header=None)
        return pd.concat([df.iloc[37:38], df.iloc[63:]], ignore_index=True)

    # The .txt has ragged lines (the table markers carry a single field), so the
    # column count is fixed up front and short lines are padded rather than raising.
    with open(file_path, encoding="utf8", errors="ignore") as fh:
        head = [r for r, _ in zip(csv.reader(fh, delimiter="	", quotechar='"'),
                                  range(63))]
    ncols = max(len(r) for r in head)
    disease = pd.DataFrame([head[37] + [""] * (ncols - len(head[37]))])
    disease.columns = range(ncols)

    table = pd.read_csv(file_path, sep="	", skiprows=63, quotechar='"',
                        dtype=str, names=range(ncols))
    table = table[~table[0].astype(str).str.startswith("!")]   # drop the end marker

    return pd.concat([disease, table], ignore_index=True)


def preprocess_excel(file_path):
    try:
        df_filtered = load_series_matrix(file_path)
    except Exception as e:
        print(f"Error reading original data file: {e}")
        return

    df_filtered.iloc[0, 0] = ""

    disease_row = df_filtered.iloc[0]
    selected_columns = [0]

    label_map = {}
    for col_index in range(1, len(disease_row)):
        label = str(disease_row[col_index]).strip().lower()
        if "sepsis" in label:
            label_map[col_index] = 1
            selected_columns.append(col_index)
        elif "healthy" in label or "noninfectious" in label:
            label_map[col_index] = 0
            selected_columns.append(col_index)

    df_filtered = df_filtered.iloc[:, selected_columns]

    for col_index, binary_value in label_map.items():
        df_filtered.iat[0, selected_columns.index(col_index)] = binary_value

    processed_rows = []
    record_count = 1

    for row in df_filtered.iloc[1:].itertuples(index=False, name=None):
        print(f"Processing row {record_count}")
        record_count += 1
        query_id = str(row[0]).strip()

        mapped_value = mapping_dict.get(query_id, "")
        mapped_value = str(mapped_value).strip()

        if mapped_value and mapped_value.startswith("hsa"):
            row = list(row)
            row[0] = mapped_value
            processed_rows.append(row)

    if processed_rows:
        df_processed = pd.DataFrame([df_filtered.iloc[0].tolist()] + processed_rows)
    else:
        df_processed = df_filtered.iloc[:1]

    df_processed.to_excel(output_path, index=False, header=False)
    print(f"Preprocessed data saved as 'preprocessed_data.xlsx' with {len(df_processed)} rows.")

preprocess_excel(original_data_path)
