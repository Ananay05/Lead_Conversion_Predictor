"""
Script to generate synthetic lead conversion dataset.
Run this once to create the training data.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

data = {
    "lead_id": range(1, n + 1),
    "age": np.random.randint(18, 65, n),
    "income": np.random.randint(20000, 150000, n),
    "lead_source": np.random.choice(["Organic", "Paid Ad", "Referral", "Social Media", "Email"], n),
    "website_visits": np.random.randint(1, 20, n),
    "time_spent_on_site": np.round(np.random.exponential(5, n), 2),
    "pages_viewed": np.random.randint(1, 15, n),
    "email_opened": np.random.randint(0, 2, n),
    "previous_interaction": np.random.randint(0, 2, n),
    "lead_score": np.random.randint(0, 100, n),
    "industry": np.random.choice(["Tech", "Finance", "Healthcare", "Education", "Retail"], n),
    "follow_up_calls": np.random.randint(0, 5, n),
}

df = pd.DataFrame(data)

# Create a realistic target variable
score = (
    (df["lead_score"] > 60).astype(int) * 2 +
    df["email_opened"] +
    df["previous_interaction"] +
    (df["website_visits"] > 5).astype(int) +
    (df["time_spent_on_site"] > 5).astype(int) +
    (df["follow_up_calls"] > 1).astype(int)
)
prob = score / score.max()
df["converted"] = (np.random.rand(n) < prob).astype(int)

df.to_csv("leads.csv", index=False)
print(f"Dataset generated: {len(df)} rows")
print(f"Conversion rate: {df['converted'].mean():.2%}")
