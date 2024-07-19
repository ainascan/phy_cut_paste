# phy_cut_paste

2D Physics Based Cut-And-Paste Data Augmentation for Multiple Annotations Per Image

| Original Image | Backdrop Image | Augmented Image |
| --- | --- | --- |
| ![Original Image](/graphics/original.jpg) | ![Backdrop Image](/graphics/backdrop.jpg) | ![Augmented Image](/graphics/augmented.jpg) |


# Problem Statement

the [CUT-AND-PASTE data augmentation strategy](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf) has shown to be a strong data augmentation strategy for object detection tasks. However, most implements assume that there is only a single annotation per image. In the case of multiple annotations per image, most implementations can prove problematic as randomly pasting a mask can result in overlapping objects and invalid annotations.

# Solution

This `phy_cut_paste` codebase provides a cut-and-paste augmentation strategy that prevents data overlaps. By dropping the provided contours into a physics simulation, collision detection can ensure that no overlaps are possible. This allows for a wide range of options by being able to adjust the force vectors, number of timesteps, gravity, mass, density, center of gravity, and much more!

# How to Use

## Install

```bash
pip install phy-cut-paste
```

## Augment All Files From a Coco Dataset

All the annotations within each image will be cut and pasted into the simulation and augmented.

```python

from phy_cut_paste import simulate_coco, AugmentedCocoImage

if __name__ == "__main__":
    iterator = simulate_coco(
        coco_file='coco.json',
        image_dir='/path/to/coco/images',
        image_backdrop_path='/path/to/backdrop.jpg',
    )

    for i, a: AugmentedCocoImage in enumerate(iterator):
        cv2.imwrite('augmented_{i}.jpg', a.augmented_image)
```

## Custom Augmentation

Pass in a list of contours and they will be cropped out of the image and pasted into the simulation

```python
from phy_cut_paste import simulate

image = cv2.imread('/path/to/image.jpg')
backdrop_image = cv2.imread('/path/to/backdrop.jpg')

contours = [
    np.array([[0, 0], [0, 100], [100, 100], [100, 0]]),
    np.array([[200, 200], [200, 300], [300, 300], [300, 200]]),
]

augmented_image, augmented_contours = simulate(
    image=image,
    contours=contours,
    backdrop=backdrop_image,
)

cv2.drawContours(augmented_image, augmented_contours, -1, (0, 255, 0), 2)

cv2.imwrite('augmented.jpg', augmented_image)
```

## Custom Augmentation from Multiple Image Masks

Pass in a list of colored masks (the background is black but the object is colored and cropped) and they will be pasted into the simulation
You can use a `bitwise_and` operation to create these masks

```python
frame = cv2.imread('/path/to/frame.jpg')
contour = np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
bbox = cv2.boundingRect(contour)

# create the mask
mask = np.zeros(frame.shape[:2], dtype=np.uint8)
cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
mask = cv2.bitwise_and(frame, frame, mask=mask)

# crop the mask
mask = mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

# save to disk
cv2.imwrite('mask.jpg', mask)
```

```python
from phy_cut_paste import simulate_masks

color_masks = [cv2.imread(path) for path in masks]
backdrop_image = cv2.imread('/path/to/backdrop.jpg')

augmented_image, augmented_contours = simulate_masks(
    masks=color_masks,
    backdrop=backdrop_image,
)

cv2.drawContours(augmented_image, augmented_contours, -1, (0, 255, 0), 2)

cv2.imwrite('augmented.jpg', augmented_image)
```

# For Development

## Build
```
python3 setup.py sdist bdist_wheel
```

## Publish
```
python3 -m twine upload --skip-existing dist/*
```