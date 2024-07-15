# phy_cut_paste

2D Physics Based Cut-And-Paste Data Augmentation for Multiple Annotations Per Image

| Original Image | Backdrop Image | Augmented Image |
| --- | --- | --- |
| ![Original Image](/graphics/original.jpg) | ![Backdrop Image](/graphics/backdrop.jpg) | ![Augmented Image](/graphics/augmented.jpg) |


# Problem Statement

the ![CUT-AND-PASTE data augmentation strategy](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf) has shown to be a strong data augmentation strategy for object detection tasks. However, most implements assume that there is only a single annotation per image. In the case of multiple annotations per image, most implementations can prove problematic as randomly pasting a mask can result in overlapping objects and invalid annotations.

# Solution

This `phy_cut_paste` codebase provides a cut-and-paste augmentation strategy that prevents data overlaps. By dropping the provided contours into a physics simulation, collision detection can ensure that no overlaps are possible. This allows for a wide range of options by being able to adjust the force vectors, number of timesteps, gravity, mass, density, center of gravity, and much more!

# How to Use

## Install

```bash
pip install phy-cut-paste
```

## Augment All Files From a Coco Dataset

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