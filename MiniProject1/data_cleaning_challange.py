import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

df = pd.read_csv(r"C:\Users\ADMIN\Desktop\Inheritance 2k25\MiniProject1\news.csv")
print(df.head())

# Using NumPy to find missing values
missing_counts = np.sum(df.isnull().values, axis=0)
print("Missing values per column:", missing_counts)

# Remove missing text entries as before
df = df.dropna(subset=['text'])
df = df.drop_duplicates()


def word_count(text):
    return len(str(text).split())

def sentence_count(text):
    # Simple sentence split by periods.
    return len(re.split(r'[.!?]', str(text))) - 1

# Compute counts using Pandas apply
df['word_count'] = df['text'].apply(word_count)
df['sentence_count'] = df['text'].apply(sentence_count)

# Use NumPy for mean calculations
word_lengths = df['word_count'].values
sentence_lengths = df['sentence_count'].values

print("Average word count:", np.mean(word_lengths))
print("Average sentence count:", np.mean(sentence_lengths))

plt.hist(word_lengths, bins=20)
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Distribution of Word Counts')
plt.show()

plt.hist(sentence_lengths, bins=20)
plt.xlabel('Sentence Count')
plt.ylabel('Frequency')
plt.title('Distribution of Sentence Counts')
plt.show()

df.to_csv('cleaned_file.csv', index=False)