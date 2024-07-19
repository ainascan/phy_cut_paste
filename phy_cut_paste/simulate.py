import os
import json
from typing import Iterator
from dataclasses import dataclass

import pymunk as pm
import numpy as np
import cv2


@dataclass
class AugmentedCocoImage:
    """
    A dataclass that represents the result of a coco image simulation.
    """
    
    coco_image: dict
    coco_annotations: list[dict]
    
    original_image: np.ndarray
    original_contours: list[np.ndarray]
    
    augmented_image: np.ndarray
    augmented_contours: list[np.ndarray]
    

@dataclass
class ContourSimulationConfig:
    """
    Configurations for a specific contour
    """
    mass: float = 1
    force_magnitude: float = 10_000
    force_offset: np.ndarray = np.array([100, 100])
    elasticitiy: float = 0.95
    friction: float = 0.1
    body_type: str = 'dynamic'
    

def load_coco_data(coco_file) -> list[tuple[dict, list[dict]]]:
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
        
    images = coco_data['images']
    
    data = []
    
    # build a map. Signficantly faster than searching...
    annotation_map = {}
    for annotation in coco_data['annotations']:
        if annotation['image_id'] not in annotation_map:
            annotation_map[annotation['image_id']] = []
        annotation_map[annotation['image_id']].append(annotation)
    
    for image in images:
        annotations = annotation_map.get(image['id'], [])
        
        if len(annotations) == 0:
            continue
        
        data.append((image, annotations))

    return data


def simulate_coco(
    coco_file: str,
    image_dir: str,
    image_backdrop_path: str,
) -> Iterator[AugmentedCocoImage]:
    """Provides an iterator that yields augmented images given a coco file, image directory, and backdrop image.

    Args:
        coco_file (str): coco file path
        image_dir (str): image directory path to the coco images
        image_backdrop_path (str): image path to the backdrop image

    Raises:
        ValueError: If the backdrop image is not the same shape as the coco image

    Yields:
        Iterator[AugmentedCocoImage]: An iterator that yields augmented images
    """

    coco_data = load_coco_data(coco_file)

    backdrop = cv2.imread(image_backdrop_path)
    
    for image, annotations in coco_data:
        
        print(os.path.join(image_dir, image['file_name']))
        
        frame = cv2.imread(os.path.join(image_dir, image['file_name']))
        
        if frame.shape != backdrop.shape:
            raise ValueError(f'Backdrop image must be the same shape as the frame image. Frame shape: {frame.shape}, Backdrop shape: {backdrop.shape}')
        
        contours = []
        for annotation in annotations:
            contours.append(np.array(annotation['segmentation'][0]).reshape(-1, 2))
            
        augmented_frame, new_contours = simulate(frame, contours, backdrop)
        
        yield AugmentedCocoImage(
            coco_image=image,
            coco_annotations=annotations,
            original_image=frame,
            original_contours=contours,
            augmented_image=augmented_frame,
            augmented_contours=new_contours,
        )


def update_mask(mask, translation, rotation) -> np.ndarray:
    """Applys a translation and rotation to a mask

    Args:
        mask (np.ndarray): The mask to apply the transformation to
        translation (np.ndarray): The 2D translation to apply
        rotation (np.ndarray): The rotation in radians to apply

    Returns:
        np.ndarray: The updated mask
    """
    return cv2.warpAffine(mask, np.array([
        [np.cos(rotation), -np.sin(rotation), translation[0]],
        [np.sin(rotation), np.cos(rotation), translation[1]]
    ]), (mask.shape[1], mask.shape[0]))


def update_contour(contour, translation, rotation) -> np.ndarray:
    """Applies a translation and rotation to a contour

    Args:
        contour (np.ndarray): The contour to apply the transformation to
        translation (np.ndarray): The 2D translation to apply
        rotation (float): The rotation in radians to apply

    Returns:
        np.ndarray: The updated contour
    """
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
    contour_configs: list[ContourSimulationConfig] = None,
    boundary_elasticity: float = 0.95,
    boundary_friction: float = 0.1,
    timesteps: int = 1000,
    timestep_delta: float = 1 / 60,
    threads: int = 4,
    iterations: int = 10,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Simulates the movement of the contours in the image and generates a new image given the provided backdrop.

    Args:
        image (np.ndarray): The image to simulate the contours on
        contours (list): The list of contours to simulate
        backdrop (np.ndarray): The backdrop image to layer the simulated contours on
        contour_configs (list[ContourSimulationConfig], optional): The list of configurations for each contour. Defaults to None.
        boundary_elasticitiy (float, optional): Defaults to 0.95.
        boundary_friction (float, optional): Defaults to 0.1.
        timesteps (int, optional): Timestamps in seconds. Defaults to 1000.
        timestep_delta (float, optional): Defaults to 1 / 60.
        threads (int, optional): Defaults to 4.
        iterations (int, optional): Improves accuracy of simulator. Defaults to 10.

    Returns:
        tuple[np.ndarray, list[np.ndarray]]: The new image and the list of new contours
    """

    space = pm.Space(threaded=True)
    space.gravity = (0.0, 0.0)
    space.threads = threads
    space.iterations = iterations
    
    objects = []
    
    for i, contour in enumerate(contours):
        c = contour.astype(float).tolist()
        
        configs = contour_configs[i] if contour_configs else ContourSimulationConfig()
        if configs is None:
            configs = ContourSimulationConfig()
            
        body_type = pm.Body.DYNAMIC
        if configs.body_type == 'static':
            body_type = pm.Body.STATIC

        body = pm.Body(
            configs.mass,
            pm.moment_for_poly(configs.mass, c),
            body_type
        )         
        poly = pm.Poly(body, c)
        poly.mass = configs.mass
        poly.elasticity = configs.elasticitiy
        poly.friction = configs.friction

        space.add(body, poly)
        objects.append((contour, body))

        # offet center by a random amount
        center = poly.cache_bb().center()
        offset = np.random.rand(2) * configs.force_offset - (configs.force_offset / 2)
        center += offset
        
        # apply a random force to the object
        force = np.random.rand(2) * configs.force_magnitude - (configs.force_magnitude / 2)
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
        boundary.friction = boundary_friction
        boundary.elasticity = boundary_elasticity
    
    space.add(*boundaries)
    
    for _ in range(timesteps):
        space.step(timestep_delta)

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
    for mask in masks:
        compiled = cv2.bitwise_or(compiled, mask)

    # get binary mask
    binary_mask = cv2.cvtColor(compiled, cv2.COLOR_BGR2GRAY)
    binary_mask = cv2.threshold(binary_mask, 0, 255, cv2.THRESH_BINARY)[1]
    
    # crop out the background
    background = cv2.bitwise_and(backdrop, backdrop, mask=cv2.bitwise_not(binary_mask))

    # combine the two images to get the final image!
    return cv2.bitwise_or(compiled, background), contours