import pandas as pd
import numpy as np



df = pd.read_csv('../watermark_text/7B_sample.csv', encoding='ISO-8859-1')
df['Word_Count'] = df['W-text'].str.split().str.len()
df['Bit_Count'] = df['Num-Bits']
Word_count = df['Word_Count']
Bit_count = df['Bit_Count']
payload = np.average(Count) / np.average(Bit_count)
print(payload)
