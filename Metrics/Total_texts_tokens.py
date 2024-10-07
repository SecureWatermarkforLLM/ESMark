import pandas as pd
import nltk
import os



folder_path = '../watermark_text'

total_tokens = 0
total_rows = 0



for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

        if 'W-text' in df.columns:
            non_empty_rows = df['W-text'].dropna()
            total_rows += len(non_empty_rows)

            text = ' '.join(non_empty_rows.astype(str))
            tokens = nltk.word_tokenize(text)
            total_tokens += len(tokens)


print(f'Total number of rows in W-text column across all CSV files: {total_rows}')
print(f'Total number of tokens in W-text column across all CSV files: {total_tokens}')
