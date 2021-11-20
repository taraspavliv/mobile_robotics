import numpy as np

#distance_array = np.array([[0, 1.581, 1, 2], [1.581, 0, 0.707, 0.707], [1, 0.707, 0, 1], [2, 0.707, 1, 0]])
distance_array = np.array([
    [0, 3.1, 2, 1, 7, 5, 4, 6],
    [3, 0, 1, 2, 4, 2, 1, 3],
    [2, 1, 0, 1, 5, 3, 2, 4],
    [1, 2, 1, 0, 6, 4, 3, 5],
    [7, 4, 5, 6, 0, 2, 3, 1],
    [5, 2, 3, 4, 2, 0, 1, 1],
    [4, 1, 2, 3, 3, 1, 0, 2],
    [6, 3, 4, 5, 1, 1, 2, 0],
    ])
visit_idx_list = np.array([0])

def shortest_path(start_idx, visited_idx_list):
    global distance_array
    nb_points = np.size(distance_array,0) #number of points to go through + 1 for the starting point

    distance = np.Inf #value to optimize
    idx_list = np.zeros(np.size(distance_array,0)) #saves optimal path

    if np.size(visited_idx_list) == nb_points : #end of recursion, all points have been passed through
        return 0, np.zeros(np.size(distance_array,0))

    temp_distance = 0
    temp_idx = np.zeros(np.size(distance_array,0))

    for i in range(nb_points):
        if i in visited_idx_list : #tries all points that haven't been visited
            continue

        temp_distance, temp_idx = shortest_path(i, np.append(visited_idx_list, i))
        temp_distance += distance_array[start_idx][i]
        
        if temp_distance < distance : #if found a better way, updates its optimal path and value
            distance = temp_distance
            idx_list = temp_idx
            idx_list[visited_idx_list.size] = i

    return (distance, idx_list)

print(shortest_path(0, visit_idx_list, ))