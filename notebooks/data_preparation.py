# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import os
import matplotlib.pyplot as plt 
import pandas as pd

# ### Check the dataset

# Download dataset from [Kaggle](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1)
# to DATASET_PATH

DATASET_PATH = 'data'

print('DATASET_PATH contents:', os.listdir(DATASET_PATH))

# Reading the rows and dropping the ones with errors
df = pd.read_csv(os.path.join(DATASET_PATH, 'styles.csv'), error_bad_lines=False)
df.head(5)

print(f'Dataset shape: {df.shape}')

print(f'Unique articleTypes: {len(df.articleType.unique())}')

plt.figure(figsize=(7,7))
df.articleType.value_counts().head(20).plot(kind='barh')



# ### Prepare data

# Drop invalid entries
image_files = os.listdir(os.path.join(DATASET_PATH, 'images'))
df['image'] = df.apply(lambda row: str(row['id']) + '.jpg', axis=1)
df = df[df.image.isin(image_files)]
df = df.reset_index(drop=True)
print(f'Clean data shape: {df.shape}')

print(f'Unique articleTypes in clean dataset: {len(df.articleType.unique())}')

# create integer labels. The order is determined by frequency
type_counts = df.articleType.value_counts()
type_labels = dict((y, x) for (x, y) in enumerate(type_counts.index))
df['label'] = df.apply(lambda row: type_labels[row['articleType']], axis=1)
assert df.label.max() == len(df.articleType.unique()) - 1

print(f'Dataset with labels shape: {df.shape}')

# Top 20 classes
top20_classes = df.articleType.value_counts().head(20)
top20_classes.plot(kind='barh')

print(f'Total items in top-20 classes: {top20_classes.sum()}')
print(f'Total other classes: {len(df) - top20_classes.sum()}')



# Test set
test = df[df.year % 2 == 1]
print(f'Test shape: {test.shape}')

test.articleType.value_counts().head(20).plot(kind='barh')



# Train sets
train_top20 = df[(df.year % 2 == 0) & df.articleType.isin(top20_classes.index)]
train_other = df[(df.year % 2 == 0) & ~df.articleType.isin(top20_classes.index)]
print(f'Train top-20 shape: {train_top20.shape}')
print(f'Train other shape: {train_other.shape}')

# +
# Check splitting
# -

# Check which classes are in train set
check_train_top20 = top20_classes.reset_index()
check_train_top20.columns = ['articleType', 'counts']
check_train_top20['top20_is_in'] = check_train_top20.articleType.isin(train_top20.articleType)

print(f'Are all top 20 classes in train: {check_train_top20.top20_is_in.all()}')


# +
# Create final dataset indices and save them
# -

def create_index_from_df(df, name, use_columns=['articleType']):
    df_final = pd.DataFrame(columns=['image', 'label'])
    df_final['image'] = os.path.join(DATASET_PATH, 'images') + '/' + df.image
    df_final['label'] = df.label
    for column in use_columns:
        df_final[column] = df[column].to_numpy()
    df_final = df_final.reset_index(drop=True)
    df_final.to_pickle(name)
    return df_final


create_index_from_df(test, os.path.join(DATASET_PATH, 'test.p'))
create_index_from_df(train_top20, os.path.join(DATASET_PATH, 'train_top20.p'))
create_index_from_df(train_other, os.path.join(DATASET_PATH, 'train_other.p'))








