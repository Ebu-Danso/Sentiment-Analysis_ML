import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/labels.csv")

print(df.head())
print(df['label'].value_counts())

img = Image.open("data/raw/images/" + df.iloc[0]['image_name'])
plt.imshow(img)
plt.title(df.iloc[0]['label'])
plt.axis('off')
plt.show()
