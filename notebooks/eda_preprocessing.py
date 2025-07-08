import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv(r"C:\Users\Dawit Woldesenbet\Desktop\complaints.csv")

target_products = ["Credit card", "Personal loan", "Buy Now, Pay Later (BNPL)", "Savings account", "Money transfers"]
df = df[df['Product'].isin(target_products)]

df = df.dropna(subset=['Consumer complaint narrative'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_text)

df.to_csv("data/filtered_complaints.csv", index=False)

plt.figure(figsize=(10, 5))
df['Product'].value_counts().plot(kind='bar')
plt.title("Complaints by Product")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/product_distribution.png")

df['narrative_length'] = df['cleaned_narrative'].apply(lambda x: len(x.split()))
sns.histplot(df['narrative_length'], bins=50)
plt.title("Narrative Word Count Distribution")
plt.xlabel("Word Count")
plt.savefig("plots/narrative_lengths.png")
