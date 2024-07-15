import os
import json
import time

import pymunk as pm
import numpy as np
import cv2


def load_coco_data(coco_file):
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
        
    images = coco_data['images']
    
    data = []
    
    for image in images:
        annotations = list(filter(lambda x: x['image_id'] == image['id'], coco_data['annotations']))
        
        if len(annotations) == 0:
            continue
        
        data.append((image, annotations))

    return data


def simulate_coco(
    coco_file: str,
    image_dir: str,
    image_backdrop_path: str,
) -> list[np.ndarray]:

    coco_data = load_coco_data(coco_file)

    backdrop = cv2.imread(image_backdrop_path)
    
    frames = []
    
    new_coco_data = []
    
    for image, annotations in coco_data:
        
        print(os.path.join(image_dir, image['file_name']))
        
        frame = cv2.imread(os.path.join(image_dir, image['file_name']))
        
        if frame.shape != backdrop.shape:
            raise ValueError(f'Backdrop image must be the same shape as the frame image. Frame shape: {frame.shape}, Backdrop shape: {backdrop.shape}')
        
        contours = []
        for annotation in annotations:
            contours.append(np.array(annotation['segmentation'][0]).reshape(-1, 2))
            
        augmented_frame, new_contours = simulate(frame, contours, backdrop)
        new_coco_data.append((augmented_frame, new_contours))

    return new_coco_data


def update_mask(mask, translation, rotation):
    """Applys a translation and rotation to a mask

    Args:
        mask (np.ndarray): The mask to apply the transformation to
        translation (np.ndarray): The 2D translation to apply
        rotation (np.ndarray): The rotation in radians to apply

    Returns:
        np.ndarray: _description_
    """
    return cv2.warpAffine(mask, np.array([
        [np.cos(rotation), -np.sin(rotation), translation[0]],
        [np.sin(rotation), np.cos(rotation), translation[1]]
    ]), (mask.shape[1], mask.shape[0]))


def update_contour(contour, translation, rotation):
    contour += translation

    # rotate the contour around it's center
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)]
    ])
    
    #contour = np.dot(contour, rotation_matrix.T)
    contour = np.dot(contour - translation, rotation_matrix.T) + translation
    
    return contour


def get_mask(image, contour):
    """Generates a mask containing the RGB values of the contour

    Args:
        image (np.ndarray): The image to generate the mask from
        contour (np.ndarray): The contour to generate the mask from

    Returns:
        np.ndarray: The mask containing the RGB values of the contour
    """
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour.astype(int)], -1, (255, 255, 255), -1)
    return cv2.bitwise_and(image, mask)


def simulate(
    image: np.ndarray,
    contours: list,
    backdrop: np.ndarray,
    force_magnitude: float = 10_000,
    force_offset: np.ndarray = 100,
    elasticitiy: float = 0.95,
    friction: float = 0.1,
    timesteps: int = 1000,
    threads: int = 4,
    iterations: int = 10,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Simulates the movement of the contours in the image and generates a new image given the provided backdrop.

    Args:
        image (np.ndarray): _description_
        contours (list): _description_
        backdrop (np.ndarray): _description_
        force_magnitude (float, optional): _description_. Defaults to 10_000.
        force_offset (np.ndarray, optional): _description_. Defaults to 100.
        elasticitiy (float, optional): _description_. Defaults to 0.95.
        friction (float, optional): _description_. Defaults to 0.1.
        timesteps (int, optional): _description_. Defaults to 1000.
        threads (int, optional): _description_. Defaults to 4.
        iterations (int, optional): _description_. Defaults to 10.

    Returns:
        tuple[np.ndarray, list[np.ndarray]]: The new image and the list of new contours
    """
    
    
    space = pm.Space(threaded=True)
    space.gravity = (0.0, 0.0)
    space.threads = threads
    space.iterations = iterations
    
    objects = []
    
    for contour in contours:
        c = contour.astype(float).tolist()

        body = pm.Body(1, pm.moment_for_poly(1, c))         
        poly = pm.Poly(body, c)
        poly.mass = 1
        poly.elasticity = elasticitiy
        poly.friction = friction

        space.add(body, poly)
        objects.append((contour, body))

        # offet center by a random amount
        center = poly.cache_bb().center()
        offset = np.random.rand(2) * force_offset - (force_offset / 2)
        center += offset
        
        force = np.random.rand(2) * force_magnitude - (force_magnitude / 2)
    
        body.apply_force_at_world_point(force.tolist(), center)
    
    # add boundaries at the image edges
    height, width = image.shape[:2]
    boundaries = [
        pm.Segment(space.static_body, (0, 0), (width, 0), 2),
        pm.Segment(space.static_body, (width, 0), (width, height), 2),
        pm.Segment(space.static_body, (width, height), (0, height), 2),
        pm.Segment(space.static_body, (0, height), (0, 0), 2),
    ]
    
    for boundary in boundaries:
        boundary.friction = friction
        boundary.elasticity = elasticitiy
    
    space.add(*boundaries)
    
    for _ in range(timesteps):
        space.step(1 / 60)

    masks = []
    contours = []
    
    for contour, body in objects:
        rotation = body.angle
        translation = np.array(body.position)

        # get the mask of the contour
        new_mask = update_mask(get_mask(image.copy(), contour), translation, rotation)

        contours.append(update_contour(contour, translation, rotation))

        masks.append(new_mask)

    # layer the masks on top of each other
    compiled = np.zeros_like(image)

    # layer them faster than a for loop
    for mask in masks:
        compiled = cv2.bitwise_or(compiled, mask)

    # get binary mask
    binary_mask = cv2.cvtColor(compiled, cv2.COLOR_BGR2GRAY)
    binary_mask = cv2.threshold(binary_mask, 0, 255, cv2.THRESH_BINARY)[1]
    
    # crop out the background
    background = cv2.bitwise_and(backdrop, backdrop, mask=cv2.bitwise_not(binary_mask))

    # combine the two images to get the final image!
    return cv2.bitwise_or(compiled, background), contours


if __name__ == "__main__":
    augmented = simulate_coco(
        coco_file='coco.json',
        image_dir='/home/jack/Mounts/CoffeeDataset/raw_images/bay_view_dead_leaves_backdrop',
        image_backdrop_path='/home/jack/Mounts/DiskOne/kona_coffee/devin_images/20240316_114709.jpg',
    )
    
    for i, (augmented_frame, contours) in enumerate(augmented):
        cv2.imwrite(f'augmented_{i}.jpg', augmented_frame)