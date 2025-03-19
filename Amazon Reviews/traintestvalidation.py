import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel("Amazon_reviews_2_new.xlsx", engine="openpyxl")

# Chia dữ liệu thành train (80%) và temp (20%) với stratify theo 'rating'
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['rating'], random_state=42)

# Chia tập temp thành test và validation, mỗi tập chiếm 50% của temp
test_df, validation_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['rating'], random_state=42)

# Kiểm tra phân bố rating trong các tập
print("Phân bố rating tập train:")
print(train_df['rating'].value_counts(normalize=True))

print("Phân bố rating tập test:")
print(test_df['rating'].value_counts(normalize=True))

print("Phân bố rating tập validation:")
print(validation_df['rating'].value_counts(normalize=True))

# 4. Lưu các tập dữ liệu ra file CSV
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
validation_df.to_csv("validation.csv", index=False)
