"""
Turn the numbers in train.npz back into pictures you can look at.

    python view_patches.py                 one example of each class, true colour
    python view_patches.py --false-colour  the same patches in false colour
    python view_patches.py --index 17      one specific patch, every band

Nothing here is needed to compete. It is here so you can look at the data before
you model it, which is usually time well spent.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Band order in the file. Index 0 is B01, index 3 is B04, and so on.
BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
         'B8A', 'B09', 'B10', 'B11', 'B12']
RED, GREEN, BLUE = 3, 2, 1               # B04, B03, B02
NIR = 7                                  # B08, used for false colour


SCALE = 2500.0      # see note below


def stretch(img):
    """Turn raw reflectance numbers into something that looks like a photograph.

    The stored numbers are reflectance multiplied by 10,000. Over land most of
    the visible bands sit below about 2,500, so dividing by that and clipping
    gives a sensible picture. Brighter than 25 percent reflectance comes out
    white, which is what you want for roofs and bare ground.

    Use the SAME divisor for red, green and blue. If you stretch each band
    separately, uniform patches such as forest or open water have almost no
    variation to stretch, so the noise gets amplified and the patch comes out
    looking like confetti.
    """
    return np.clip(img.astype(np.float32) / SCALE, 0, 1)


def to_rgb(patch, false_colour=False):
    """patch is (64, 64, 13) uint16. Returns something plt.imshow can draw."""
    idx = [NIR, RED, GREEN] if false_colour else [RED, GREEN, BLUE]
    return stretch(patch[:, :, idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='data/train.npz')
    ap.add_argument('--false-colour', action='store_true')
    ap.add_argument('--index', type=int, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    d = np.load(a.npz)
    X = d['X']
    classes = [str(c) for c in d['classes']]

    if a.index is not None:
        patch = X[a.index]
        fig, axes = plt.subplots(2, 7, figsize=(16, 5))
        for i, ax in enumerate(axes.ravel()):
            if i < 13:
                band = patch[:, :, i].astype(np.float32)
                ax.imshow(np.clip(band / SCALE, 0, 1), cmap='gray', vmin=0, vmax=1)
                ax.set_title(BANDS[i], fontsize=9)
            else:
                ax.imshow(to_rgb(patch))
                ax.set_title('true colour', fontsize=9)
            ax.axis('off')
        label = classes[int(d['y'][a.index])] if 'y' in d.files else ''
        fig.suptitle('patch %d   %s' % (a.index, label))
    else:
        y = d['y']
        fig, axes = plt.subplots(2, 5, figsize=(13, 6.2))
        for k, ax in enumerate(axes.ravel()):
            i = int(np.where(y == k)[0][0])
            ax.imshow(to_rgb(X[i], a.false_colour))
            ax.set_title(classes[k], fontsize=10)
            ax.axis('off')
        fig.suptitle('false colour (infrared as red)' if a.false_colour else 'true colour')

    fig.tight_layout(h_pad=1.8)
    out = a.out or ('patches_false.png' if a.false_colour else 'patches_true.png')
    fig.savefig(out, dpi=110)
    print('wrote', out)


if __name__ == '__main__':
    main()
