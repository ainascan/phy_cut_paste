import os
import json
import time

import pymunk as pm
import pygame as pg
import numpy as np
import cv2

import pymunk.pygame_util as pmu


def load_coco_data(root_dir):
    with open('coco.json', 'r') as f:
        coco_data = json.load(f)
        
    images = coco_data['images']
    
    data = []
    
    for image in images:
        annotations = list(filter(lambda x: x['image_id'] == image['id'], coco_data['annotations']))
        
        if len(annotations) == 0:
            continue
        
        data.append({
            'image': image,
            'annotations': annotations
        })

    return data


def simulate():
    root_dir = '/home/jack/Mounts/CoffeeDataset/raw_images/bay_view_dead_leaves_backdrop'
    
    data = load_coco_data(root_dir)
    
    # display
    pg.init()
    #screen = pg.display.set_mode((1280, 720))
    #clock = pg.time.Clock()
    #draw_options = pmu.DrawOptions(screen)
    #draw_options.flags = pmu.DrawOptions.DRAW_SHAPES | pmu.DrawOptions.DRAW_CONSTRAINTS | pmu.DrawOptions.DRAW_COLLISION_POINTS
    draw_options = pm.SpaceDebugDrawOptions() # For easy printing
    
    space = pm.Space(threaded=True)
    space.gravity = (0.0, 0.0)
    space.threads = 4
    space.iterations = 10
    
    center_x = 640
    center_y = 360
    width, height = 0, 0
    
    objects = []
    
    image = None
    
    for point in data:
        
        image = point['image']
        width = point['image']['width']
        height = point['image']['height']
        
        annotations = point['annotations']
        for annotation in annotations:
            contour = np.array(annotation['segmentation'][0]).reshape(-1, 2).astype(float).tolist()

            body = pm.Body(1, pm.moment_for_poly(1, contour))        # Create a Body           
            poly = pm.Poly(body, contour) # Create a box shape and attach to body
            poly.mass = 1
            poly.elasticity = 0.95
            poly.friction = 0.1

            space.add(body, poly)
            objects.append((annotation, body, poly))

            # add random force
            force = 10000
            random_force = np.random.rand(2) * force - (force / 2)
            body.apply_force_at_world_point(random_force.tolist(), poly.cache_bb().center())

        break
    
    # add boundaries that are rectangles with 100 pixels width
    boundaries = [
        pm.Segment(space.static_body, (0, 0), (width, 0), 2),
        pm.Segment(space.static_body, (width, 0), (width, height), 2),
        pm.Segment(space.static_body, (width, height), (0, height), 2),
        pm.Segment(space.static_body, (0, height), (0, 0), 2),
    ]
    
    for boundary in boundaries:
        boundary.friction = 0.1
        boundary.elasticity = 0.95
        boundary.color = pg.Color("red")
    
    space.add(*boundaries)
        
    # translation = pm.Transform()
    # scaling = 0.1
    # rotation = 0
    timesteps = 1000
    steps = 0
    running = True
    start_time = time.time()

    while running:
        
        # zoom_in = 0
        # zoom_out = 0
        
        # for event in pg.event.get():
        #     if event.type == pg.QUIT:
        #         running = False
        #     elif event.type == pg.MOUSEWHEEL:
        #         zoom_in = event.y if event.y > 0 else 0
        #         zoom_out = 1 if event.y < 0 else 0
        
        # keys = pg.key.get_pressed()
        # left = int(keys[pg.K_LEFT])
        # up = int(keys[pg.K_UP])
        # down = int(keys[pg.K_DOWN])
        # right = int(keys[pg.K_RIGHT])

        #rotate_left = int(keys[pg.K_s])
        #rotate_right = int(keys[pg.K_x])
        
        #print(zoom_in, zoom_out)

        # translate_speed = 100
        # translation = translation.translated(
        #     translate_speed * left - translate_speed * right,
        #     translate_speed * up - translate_speed * down,
        # )

        # zoom_speed = 0.1
        # scaling *= 1 + (zoom_speed * zoom_in - zoom_speed * zoom_out)

        # rotation_speed = 0.1
        # #rotation += rotation_speed * rotate_left - rotation_speed * rotate_right

        # # to zoom with center of screen as origin we need to offset with
        # # center of screen, scale, and then offset back
        # draw_options.transform = (
        #     pm.Transform.translation(center_x, center_y)
        #     @ pm.Transform.scaling(scaling)
        #     @ translation
        #     @ pm.Transform.rotation(rotation)
        #     @ pm.Transform.translation(-center_x, -center_y)
        # )
        
        #screen.fill(pg.Color("white"))
        #space.debug_draw(draw_options)

        #pg.draw.rect(screen, pg.Color("black"), (0, 0, 200, 50))
        #font = pg.font.Font(None, 36)
        #text = font.render(f"Step: {steps}", True, pg.Color("white"))
        #screen.blit(text, (10, 10))

        
        space.step(1 / 60)


        #pg.display.flip()
        #clock.tick(50)
        
        steps += 1
        
        if steps >= timesteps:
            running = False
    
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time} seconds")
    
    from matplotlib import pyplot as plt
    
    
    # def update_contour(contour, translation, rotation):
    #     contour += translation

    #     # rotate the contour around it's center
    #     rotation_matrix = np.array([
    #         [np.cos(rotation), -np.sin(rotation)],
    #         [np.sin(rotation), np.cos(rotation)]
    #     ])
        
    #     #contour = np.dot(contour, rotation_matrix.T)
    #     contour = np.dot(contour - translation, rotation_matrix.T) + translation
        
    #     return contour


    def update_mask(mask, translation, rotation):
        return cv2.warpAffine(mask, np.array([
            [np.cos(rotation), -np.sin(rotation), translation[0]],
            [np.sin(rotation), np.cos(rotation), translation[1]]
        ]), (mask.shape[1], mask.shape[0]))


    def get_mask(image, contour):
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour.astype(int)], -1, (255, 255, 255), -1)
        return cv2.bitwise_and(image, mask)


    plt.figure(figsize=(20, 10))
    plt.axis('off')

    frame = cv2.imread(os.path.join(root_dir, image['file_name']))
    
    masks = []
    
    for annotation, body, poly in objects:
        rotation = body.angle
        translation = np.array(body.position)
        contour = np.array(annotation['segmentation'][0]).reshape(-1, 2)

        # get the mask of the contour
        old_mask = get_mask(frame.copy(), contour)
        new_mask = update_mask(old_mask.copy(), translation, rotation)

        masks.append(new_mask)

    # layer the masks on top of each other
    new_frame = np.zeros_like(frame)

    # layer them faster than a for loop
    for mask in masks:
        new_frame = cv2.bitwise_or(new_frame, mask)

    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    
    plt.subplot(1, 2, 2)
    plt.imshow(new_frame)

    # save the plot
    plt.savefig('plot.png')
        
if __name__ == "__main__":
    simulate()