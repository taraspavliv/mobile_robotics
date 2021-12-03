from camera import *
from path_planning import *

optimal_path = []


capture = cv.VideoCapture('new.mp4')

#for display
frame_limits = []
visibility_graph = []
vertices = []
dilated_obstacle_list = []
dilated_map = []

def setup():
    global capture, optimal_path, frame_limits, visibility_graph, vertices, dilated_obstacle_list, dilated_map

    valid_image = False
    while not valid_image:
        valid_image, first_image = capture.read()
    
    thymio_pos,targets_list,obstacles_list,dilated_obstacle_list,dilated_map, frame_limits = object_detection(first_image)

    visibility_graph,start_idx,targets_idx_list,vertices = vis_graph(thymio_pos,targets_list,obstacles_list,dilated_obstacle_list,dilated_map)

    distance_array, path_array = create_distance_path_matrix(visibility_graph,start_idx,[3,1])

    total_distance, targets_idx_order = shortest_path(0, np.array([0]), distance_array)

    print(total_distance)

    for i in range(len(targets_idx_order)-1):
        for j in range(len(path_array[i][i+1])):
            if i != 0 and j == 0: 
                continue
            print(path_array[i][i+1][j])
            optimal_path.append(vertices[path_array[i][i+1][j]])

setup()

print(optimal_path)

thymio_cam_pos = []

def cam_thread():
    global capture, optimal_path, frame_limits, visibility_graph, vertices, dilated_obstacle_list, dilated_map

    show_contours = False
    show_polygones = False
    show_dilated_polygones = False
    show_visibility_graph = False
    show_option = [show_contours, show_polygones, show_dilated_polygones, show_visibility_graph]
    while True:

        valid_image, frame = capture.read()
        if not valid_image:
            break

        bounded_frame = frame[frame_limits[0]: frame_limits[1], frame_limits[2]: frame_limits[3]]

        modified_frame = draw_on_frame(bounded_frame, show_option, dilated_obstacle_list, dilated_map)

        cv.imshow('Video', cv.resize(modified_frame, (960, 540)))
        key_pressed = cv.waitKey(60)
        if key_pressed == ord('q'):
            show_contours = not show_contours
        if key_pressed == ord('w'):
            show_polygones = not show_polygones
        if key_pressed == ord('e'):
            show_dilated_polygones = not show_dilated_polygones
        if key_pressed == ord('r'):
            show_visibility_graph = not show_visibility_graph
        if key_pressed == ord('d'):
            break

        show_option = [show_contours, show_polygones, show_dilated_polygones, show_visibility_graph]

cam_thread()


