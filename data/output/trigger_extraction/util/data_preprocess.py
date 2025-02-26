import pandas as pd

def zero_label_deletion(input_csv, output_csv, column_name, value_to_remove):
    """
    Delete rows where the specified column has the value to remove.
    """
    df = pd.read_csv(input_csv)
    df_filtered = df[df[column_name] != value_to_remove]
    df_filtered.to_csv(output_csv, index=False)
    print(f"Filtered data saved to {output_csv}")

def make_value_column(file):
    """
    Extract 'value' from 'scenario' column and clean the 'scenario' text.
    """
    df = pd.read_csv(file)
    df['value'] = df['scenario'].str.extract(r'\[(.*?)\]')[0].str.lower()
    df['scenario'] = df['scenario'].str.replace(r'\[.*?\] ', '', regex=True)
    df = df[['value', 'scenario', 'label']]
    df.to_csv(file, index=False)
    print(f"Value column added and saved to {file}")