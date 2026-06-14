import pandas as pd
import re

def load_data(path):
    return pd.read_csv(path, encoding='latin-1')

def clean_data(df):
    # Dedupolicate
    df = df.drop_duplicates()

    