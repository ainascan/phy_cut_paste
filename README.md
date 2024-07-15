# phy_cut_paste

2D Physics Based Cut-And-Paste Data Augmentation for Multiple Annotations Per Image

| Original Image | Backdrop Image | Augmented Image |
| --- | --- | --- |
| ![Original Image](/graphics/original.jpg) | ![Backdrop Image](/graphics/backdrop.jpg) | ![Augmented Image](/graphics/augmented.jpg) |


# Problem Statement

the ![CUT-AND-PASTE data augmentation strategy](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf) has shown to be a strong data augmentation strategy for object detection tasks. However, most implements assume that there is only a single annotation per image. In the case of multiple annotations per image, most implementations can prove problematic as randomly pasting a mask can result in overlapping objects and invalid annotations.

# Solution

This `phy_cut_paste` codebase provides a cut-and-paste augmentation strategy that prevents data overlaps. By dropping the provided contours into a physics simulation, collision detection can ensure that no overlaps are possible. This allows for a wide range of options by being able to adjust the force vectors, number of timesteps, gravity, mass, density, center of gravity, and much more!