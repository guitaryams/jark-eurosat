# ASEN 6337 - EuroSAT competition - Python loader
# Nothing beyond numpy is required. No imaging library, no GDAL, no rasterio.

import numpy as np

# ---------------------------------------------------------------- training set
d = np.load('../data/train.npz')

X = d['X']              # (18900, 64, 64, 13)  uint16   the patches
y = d['y']              # (18900,)             int64    label index, 0..9
classes = d['classes']  # (10,)                 str      class names

print(X.shape, X.dtype)
print('label of patch 0:', classes[y[0]])

# --------------------------------------------------------------------- test set
t = np.load('../data/test.npz')

Xt = t['X']             # (8100, 64, 64, 13)   uint16   the patches
ids = t['id']           # (8100,)              int64    submission ids

# ------------------------------------------------------------- band information
# Bands are in CANONICAL Sentinel-2 order. B8A is index 8.
bands = d['bands']              # ['B01' 'B02' ... 'B08' 'B8A' 'B09' 'B10' 'B11' 'B12']
wavelength_nm = d['wavelength_nm']
resolution_m = d['resolution_m']

B = {name: i for i, name in enumerate(bands)}   # look a band up by name

red = X[..., B['B04']]
nir = X[..., B['B08']]
ndvi = (nir.astype(np.float32) - red) / (nir + red + 1e-6)

# ------------------------------------------------------------ writing an answer
# One row per test id. Labels are the class-name strings, exactly as in `classes`.
import csv

preds = [classes[0]] * len(ids)          # replace with your model's output

with open('../results/submission.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'label'])
    for i, p in zip(ids, preds):
        w.writerow([int(i), p])
