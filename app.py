import pandas as pd
from config.config import DATA_PATH

df = pd.read_csv(DATA_PATH)
print(df.head())
