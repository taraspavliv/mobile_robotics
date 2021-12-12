import numpy as np

def dijkstra_aglorithm(idx_start, idx_end, visibility_graph):
    #calulates the shortest distance, and saves the path to have this distance
    path = []

    if idx_start == idx_end:
        return 0, np.array([])

    nb_points = len(visibility_graph)
    distance_array = np.full(nb_points, np.Inf, dtype=np.double) #creates an array to store distance to start
    distance_array[idx_start] = 0
    path_array = np.zeros(nb_points, dtype = int) #creates an array to store path from start
    explored_array = np.full(nb_points, False, dtype=np.bool) #keeps track of explored points
    explored_array[idx_start] = True
    exploring_idx = idx_start #from where we are exploring

    #find shortest path
    while(explored_array[idx_end] == False):
        for vertices in visibility_graph[exploring_idx]:
            if distance_array[exploring_idx] + vertices[1] < distance_array[vertices[0]]:
                distance_array[vertices[0]] = distance_array[exploring_idx] + vertices[1]
                path_array[vertices[0]] = exploring_idx
            
        temp_idx = np.argmin(distance_array[np.logical_not(explored_array)])
        exploring_idx = np.arange(nb_points)[np.logical_not(explored_array)][temp_idx]
        explored_array[exploring_idx] = True

    #reconstructs the shortest path
    last_idx = idx_end
    while last_idx != idx_start:
        path.insert(0,last_idx)
        last_idx = path_array[last_idx]
    path.insert(0,idx_start)
    
    return distance_array[idx_end], path


def create_distance_path_matrix(visibility_graph,start_idx,targets_idx_list):
    #we treat start as an objective, from which we start
    targets_idx_list.insert(0, start_idx)

    nb_interest_point = len(targets_idx_list)
    distance_matrix = np.zeros((nb_interest_point, nb_interest_point))
    path_matrix = np.zeros((nb_interest_point, nb_interest_point), dtype = object)
    #for all pairs of interest points, applies dijkstra algorithm
    for i in range(nb_interest_point):
        for j in range(nb_interest_point):
            if i==j:
                distance_matrix[i][j] = 0
                path_matrix[i][j] = [[targets_idx_list[i],targets_idx_list[i]]]
            elif i > j:
                distance_matrix[i][j], path_matrix[i][j] = dijkstra_aglorithm(targets_idx_list[i], targets_idx_list[j], visibility_graph)
                distance_matrix[j][i] = distance_matrix[i][j]
                path_matrix[j][i] = path_matrix[i][j][::-1]


    return distance_matrix, path_matrix


def shortest_path(start_local_idx, visited_idx_list, distance_array):
    nb_points = np.size(distance_array,0) #number of points to go through + 1 for the starting point

    distance = np.Inf #value to optimize
    idx_list = np.zeros(np.size(distance_array,0)) #saves optimal path

    if np.size(visited_idx_list) == nb_points : #end of recursion, all points have been passed through
        return 0, np.zeros(np.size(distance_array,0))

    temp_distance = 0
    temp_idx = np.zeros(np.size(distance_array,0))

    for i in range(nb_points):
        if i in visited_idx_list : #tries only points that haven't been visited
            continue
        temp_distance, temp_idx = shortest_path(i, np.append(visited_idx_list, i), distance_array)
        temp_distance += distance_array[start_local_idx][i]
        
        if temp_distance < distance : #if found a better way, updates its optimal path and value
            distance = temp_distance
            idx_list = temp_idx
            idx_list[visited_idx_list.size] = i

    return (distance, idx_list)


def reconstruct_optimal_path(path_array, targets_idx_order, vertices):
    optimal_path = []
    for i in range(len(targets_idx_order)-1):
        for j in range(len(path_array[int(targets_idx_order[i])][int(targets_idx_order[i+1])])): #use the path matrix to reconstruct
            if i != 0 and j == 0: 
                continue
            #use vertices array which gives coordinates based on index
            optimal_path.append(vertices[path_array[int(targets_idx_order[i])][int(targets_idx_order[i+1])][j]])
    return optimal_path