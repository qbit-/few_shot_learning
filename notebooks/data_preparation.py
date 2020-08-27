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
import numpy as np
import pandas as pd
from PIL import Image

# ### Check the dataset

# Download dataset from [Kaggle](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1)
# to DATASET_PATH

DATASET_PATH = 'data'
RANDOM_SEED=42

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
image_names = df.apply(lambda row: str(row['id']) + '.jpg', axis=1)
df = df[image_names.isin(image_files)]
df['image'] = df.apply(lambda row: os.path.join(DATASET_PATH, 'images', str(row['id']) + '.jpg'), axis=1)
df = df.reset_index(drop=True)
print(f'Clean data shape: {df.shape}')

print(f'Unique articleTypes in clean dataset: {len(df.articleType.unique())}')

# create integer labels. The order is determined by frequency
type_counts = df.articleType.value_counts()
type_labels = dict((y, x) for (x, y) in enumerate(type_counts.index))
df['label'] = df.apply(lambda row: type_labels[row['articleType']], axis=1)
assert df.label.max() == len(df.articleType.unique()) - 1

print(f'Dataset with labels shape: {df.shape}')

# create series with labels to type map
label_types = pd.Series(type_counts.index)
label_types.to_pickle(os.path.join(DATASET_PATH, 'label_types.p'))

# Test the size of the images
image_params = pd.DataFrame()
image_params['image'] = df.image
image_params['size'] = image_params.apply(lambda row: Image.open(row['image']).size, axis=1)
image_params['nelem'] = image_params.apply(lambda row: np.prod(row['size']), axis=1)

image_params.sort_values('nelem', inplace=True)
print(f'Min image size: {image_params.head(1)["size"].to_list()[0]},',
      f'Max image size: {image_params.tail(1)["size"].to_list()[0]}')



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
print(f'Train other shape:  {train_other.shape}')

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

def create_index_from_df(df, name):
    df_final = pd.DataFrame(columns=['image', 'label', 'label_type'])
    df_final['image'] = df.image
    df_final['label'] = df.label
    df_final['label_type'] = df.articleType
    df_final = df_final.reset_index(drop=True)
    df_final.to_pickle(name)
    return df_final


create_index_from_df(test, os.path.join(DATASET_PATH, 'test.p'))
create_index_from_df(train_top20, os.path.join(DATASET_PATH, 'train_top20.p'))
create_index_from_df(train_other, os.path.join(DATASET_PATH, 'train_other.p'))








