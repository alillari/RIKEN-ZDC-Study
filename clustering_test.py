import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import uproot as up
import math
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import scipy as sp

def distance_of_closest_approach(point, momentum, vertex):
    pdelta_r = (vertex[0] - point[0])*momentum[0] + (vertex[1] - point[1])*momentum[1] + (vertex[2] - point[2])*momentum[2]
    t0 = -(pdelta_r/(momentum[0]**2 + momentum[1]**2 + momentum[2]**2))
    closest_point = [vertex[0] + t0*momentum[0], vertex[1] + t0*momentum[1], vertex[2] + t0*momentum[2]]
    #print("Suggested closest point, ", closest_point)
    dist = math.sqrt((point[0] - vertex[0] - t0*momentum[0])**2 + (point[1] - vertex[1] - t0*momentum[1])**2 + (point[2] - vertex[2] - t0*momentum[2])**2)
    return dist

def find_particle_index(pdg_list, parent_list, pid, true_parent, instance):
    number_found = 0
    for particle in range(len(pdg_list)):
        if(pdg_list[particle] == pid and pdg_list[parent_list[particle-1]] == true_parent):
            number_found = number_found + 1 
            if(number_found == instance):
                return particle

    #obviously out of range error, method failed
    return 9999

def two_vectors_closest_approach(vertex1, vertex2, momentum1, momentum2):
    A = np.array([[(momentum1.T)@(vertex2-vertex1)], [(momentum2.T)@(vertex2-vertex1)]])
    B = np.linalg.inv(np.array([[(momentum1.T)@momentum1 , -(momentum1.T)@momentum2],[-(momentum1.T)@momentum2,(momentum2.T)@momentum2]]))
    st = B@A
    point1 = vertex1 + st[0]*momentum1
    point2 = vertex2 + st[1]*momentum2

    point_of_closest_approach = (point1+point2)/2

    return point_of_closest_approach

def make_MC_vector_for_particle(pid, instance, pdg_list, parent_list, MC_vert, MC_mom):
    if(pid == 22):
        true_parent = 111
    elif(pid == 111 or pid == 2112):
        true_parent = 3122

    particle_index = find_particle_index(pdg_list, parent_list, pid, true_parent, instance)

    particle_vertex = [MC_vert[particle_index, 0], MC_vert[particle_index, 1], MC_vert[particle_index, 2]]
    particle_momentum = [MC_mom[particle_index, 0], MC_mom[particle_index, 1], MC_mom[particle_index, 2]]

    return particle_vertex, particle_momentum

def get_weighted_average(cluster):
    cluster_energy_sum = np.sum(cluster[:,3])

    cluster_ave = np.average(cluster, axis=0, weights=cluster[:, 3]/cluster_energy_sum)

    return cluster_ave

def make_pred_vector_from_ave(peak_ecal_cluster, peak_hcal_cluster):
    peak_ecal_energy_sum = np.sum(peak_ecal_cluster[:,3])
    peak_hcal_energy_sum = np.sum(peak_hcal_cluster[:,3])

    peak_ecal_ave = np.average(peak_ecal_cluster[:,:3], axis=0, weights=peak_ecal_cluster[:,3]/peak_ecal_energy_sum)
    peak_hcal_ave = np.average(peak_hcal_cluster[:,:3], axis=0, weights=peak_hcal_cluster[:,3]/peak_hcal_energy_sum)

    peak_mom = peak_hcal_ave - peak_ecal_ave

    return peak_hcal_ave, peak_mom

def pred_energy_gamma_cluster(ecal_cluster, hcal_cluster):
    true_energy = np.sum(ecal_cluster[:,3]) + np.sum(hcal_cluster[:,3])*(1/.02044)
    return true_energy

def clustering_check(peak1_hcal_ave, peak2_hcal_ave, gamma1_vert, gamma2_vert, gamma1_mom, gamma2_mom):
    peak1_from_gamma1 = distance_of_closest_approach(peak1_hcal_ave, gamma1_mom, gamma1_vert)
    peak1_from_gamma2 = distance_of_closest_approach(peak1_hcal_ave, gamma2_mom, gamma2_vert)
    peak2_from_gamma1 = distance_of_closest_approach(peak2_hcal_ave, gamma1_mom, gamma1_vert)
    peak2_from_gamma2 = distance_of_closest_approach(peak2_hcal_ave, gamma2_mom, gamma2_vert)

    if(peak1_from_gamma1 < peak1_from_gamma2):
        peak1_nearest_gamma = 1
        peak1_nearest_dist = peak1_from_gamma1
    else:
        peak1_nearest_gamma = 2
        peak1_nearest_dist = peak2_from_gamma1

    if(peak2_from_gamma1 < peak2_from_gamma2):
        peak2_nearest_gamma = 1
        peak2_nearest_dist = peak2_from_gamma1
    else:
        peak2_nearest_gamma = 2
        peak2_nearest_dist = peak2_from_gamma2

    if(peak1_nearest_dist < 30 and peak2_nearest_dist < 30 and peak1_nearest_gamma != peak2_nearest_gamma):
        return True

    return False



def looping_through_events(indices, ecal_hits_x, ecal_hits_y, ecal_hits_z, hcal_hits_x, hcal_hits_y, hcal_hits_z, MC_vert_x, MC_vert_y, MC_vert_z, MC_mom_x, MC_mom_y, MC_momz, MC_pdg, MC_parents):
    good_event_counter = 0

    for event in indices:
        looping_rotated_ecal_hits = rotate_hits(np.column_stack((ecal_hits_x[event],ecal_hits_y[event],ecal_hits_z[event])))
        looping_rotated_hcal_hits = rotate_hits(np.column_stack((hcal_hits_x[event],hcal_hits_y[event],hcal_hits_z[event])))

        looping_ecal_data = np.column_stack((looping_rotated_ecal_hits, ecal_hits_energy[event].T)) 
        looping_hcal_data = np.column_stack((looping_rotated_hcal_hits, hcal_hits_energy[event].T))

        looping_ecal_data = energy_cut(looping_ecal_data, .005)
        looping_hcal_data = energy_cut(looping_hcal_data, .0005)

        if looping_ecal_data.shape[0] == 0 or looping_hcal_data.shape[0] == 0:
            print("Event ", event, " failed due to no hits")
            continue

        looping_ecal_peaks = ECal_peak_finding(looping_ecal_data)
        looping_hcal_peaks = HCal_peak_finding(looping_hcal_data, 80)
        if filter_based_on_ECal_peaks(looping_ecal_peaks) and filter_based_on_HCal_peaks(looping_hcal_data[looping_hcal_data[:,2] < 36050]):
            rotated_verts = rotate_hits(np.column_stack((MC_vert_x[event],MC_vert_y[event],MC_vert_z[event])))
            rotated_mom = rotate_hits(np.column_stack((MC_mom_x[event],MC_mom_y[event],MC_mom_z[event])))

            gamma1 = find_particle_index(MC_pdg[event], MC_parents[event], 22, 111, 1)
            gamma2 = find_particle_index(MC_pdg[event], MC_parents[event], 22, 111, 2)

            gamma1_vertex, gamma1_mom = make_MC_vector_for_particle(22, 1, MC_pdg[event], MC_parents[event], rotated_verts, rotated_mom)
            gamma2_vertex, gamma2_mom = make_MC_vector_for_particle(22, 2, MC_pdg[event], MC_parents[event], rotated_verts, rotated_mom)

            MC_energy_1 = math.sqrt((rotated_mom[gamma1, 0])**2 + (rotated_mom[gamma1,1])**2 + (rotated_mom[gamma1,2])**2)
            MC_energy_2 = math.sqrt((rotated_mom[gamma2, 0])**2 + (rotated_mom[gamma2,1])**2 + (rotated_mom[gamma2,2])**2)

            proto_clusters, neutron_check = peak_merging(looping_ecal_peaks, looping_hcal_peaks)
            proto_gamma_clusters = filter_cluster(proto_clusters, neutron_check)

            if(len(proto_gamma_clusters) != 2):
                print("Event ",event," failed due to failing to find two gamma clusters.")
                continue

            if(len(proto_gamma_clusters[0]) != 2 or len(proto_gamma_clusters[1]) != 2):
                print("Event ", event, " failed due to lack of associated ECal or HCal hit.")
                continue

            peak1_hcal_cluster = expand_cluster(proto_gamma_clusters[0], looping_ecal_data, looping_hcal_data, 37, .01, 300, 70)
            peak2_hcal_cluster = expand_cluster(proto_gamma_clusters[1], looping_ecal_data, looping_hcal_data, 37, .01, 300, 70)

            peak1_hcal_ave = get_weighted_average(peak1_hcal_cluster)
            peak2_hcal_ave = get_weighted_average(peak2_hcal_cluster)

            if(clustering_check(peak1_hcal_ave, peak2_hcal_ave, gamma1_vertex, gamma2_vertex, gamma1_mom, gamma2_mom) == False):
                print("Event ", event, " failed in clustering.")
                continue

            peak1_ecal_cluster = expand_ECal_cluster(proto_gamma_clusters[0][0].reshape(1,4), looping_ecal_data, 90)
            peak2_ecal_cluster = expand_ECal_cluster(proto_gamma_clusters[1][0].reshape(1,4), looping_ecal_data, 90)

            peak1_hcal_ave = get_weighted_average(peak1_hcal_cluster)
            peak2_hcal_ave = get_weighted_average(peak2_hcal_cluster)

            pred_energy_1 = pred_energy_gamma_cluster(peak1_ecal_cluster, peak1_hcal_cluster)
            pred_energy_2 = pred_energy_gamma_cluster(peak2_ecal_cluster, peak2_hcal_cluster)

            #print("Predicted energy 1: ", pred_energy_1)
            #print("Predicted energy 2: ", pred_energy_2)

            ++good_event_counter
        else:
            print("Event ", event, " failed due to lack of peaks")

    print("Total number of good events: ",good_event_counter)

def find_gamma_one_event(event, ecal_hits_x, ecal_hits_y, ecal_hits_z, hcal_hits_x, hcal_hits_y, hcal_hits_z, MC_vert_x, MC_vert_y, MC_vert_z, MC_mom_x, MC_mom_y, MC_momz, MC_pdg, MC_parents):
    rotated_good_event_ecal_hits = rotate_hits(np.column_stack((ecal_hits_x[event],ecal_hits_y[event],ecal_hits_z[event])))
    rotated_good_event_hcal_hits = rotate_hits(np.column_stack((hcal_hits_x[event],hcal_hits_y[event],hcal_hits_z[event])))

    rotated_verts = rotate_hits(np.column_stack((MC_vert_x[event],MC_vert_y[event],MC_vert_z[event])))
    rotated_mom = rotate_hits(np.column_stack((MC_mom_x[event],MC_mom_y[event],MC_mom_z[event])))

    neutron = find_particle_index(MC_pdg[event], MC_parents[event], 2112, 3122, 1)
    gamma1 = find_particle_index(MC_pdg[event], MC_parents[event], 22, 111, 1)
    gamma2 = find_particle_index(MC_pdg[event], MC_parents[event], 22, 111, 2)

    neutron_vertex, neutron_mom = make_MC_vector_for_particle(2112, 1, MC_pdg[event], MC_parents[event], rotated_verts, rotated_mom)
    gamma1_vertex, gamma1_mom = make_MC_vector_for_particle(22, 1, MC_pdg[event], MC_parents[event], rotated_verts, rotated_mom)
    gamma2_vertex, gamma2_mom = make_MC_vector_for_particle(22, 2, MC_pdg[event], MC_parents[event], rotated_verts, rotated_mom)
    pion_vertex, pion_mom = make_MC_vector_for_particle(111, 1, MC_pdg[event], MC_parents[event], rotated_verts, rotated_mom)

    good_hcal_data = np.column_stack((rotated_good_event_hcal_hits, hcal_hits_energy[event].T))
    good_ecal_data = np.column_stack((rotated_good_event_ecal_hits, ecal_hits_energy[event].T))

    good_hcal_data = energy_cut(good_hcal_data, .0005)
    good_ecal_data = energy_cut(good_ecal_data, .005)

    ecal_peaks = ECal_peak_finding(good_ecal_data)
    hcal_peaks = HCal_peak_finding(good_hcal_data, 80)

    proto_clusters, neutron_check = peak_merging(ecal_peaks, hcal_peaks)
    proto_gamma_clusters = filter_cluster(proto_clusters, neutron_check)
    peak1_hcal_cluster = expand_cluster(proto_gamma_clusters[0], good_ecal_data, good_hcal_data, 37, .01, 300, 70)
    peak2_hcal_cluster = expand_cluster(proto_gamma_clusters[1], good_ecal_data, good_hcal_data, 37, .01, 300, 70)
    
    peak1_ecal_cluster = expand_ECal_cluster(proto_gamma_clusters[0][0].reshape(1,4), good_ecal_data, 90)
    peak2_ecal_cluster = expand_ECal_cluster(proto_gamma_clusters[1][0].reshape(1,4), good_ecal_data, 90)

    peak1_ecal_ave = get_weighted_average(peak1_ecal_cluster)
    peak2_ecal_ave = get_weighted_average(peak2_ecal_cluster)
    
    peak1_hcal_ave = get_weighted_average(peak1_hcal_cluster)
    peak2_hcal_ave = get_weighted_average(peak2_hcal_cluster)

    peak1_ave, peak1_mom = make_pred_vector_from_ave(peak1_ecal_cluster, peak1_hcal_cluster) 
    peak2_ave, peak2_mom = make_pred_vector_from_ave(peak2_ecal_cluster, peak2_hcal_cluster)

    predicted_pion_point = two_vectors_closest_approach(peak1_ave, peak2_ave, peak1_mom, peak2_mom) 

    pred_energy_1 = pred_energy_gamma_cluster(peak1_ecal_cluster, peak1_hcal_cluster)
    pred_energy_2 = pred_energy_gamma_cluster(peak2_ecal_cluster, peak2_hcal_cluster)

    MC_energy_1 = math.sqrt((rotated_mom[gamma1, 0])**2 + (rotated_mom[gamma1,1])**2 + (rotated_mom[gamma1,2])**2)
    MC_energy_2 = math.sqrt((rotated_mom[gamma2, 0])**2 + (rotated_mom[gamma2,1])**2 + (rotated_mom[gamma2,2])**2)

    clustering_check_flag = clustering_check(peak1_hcal_ave, peak2_hcal_ave, gamma1_vertex, gamma2_vertex, gamma1_mom, gamma2_mom)

    print("Clustering check: ", clustering_check_flag)

    print("Predicted pion vertex: ", predicted_pion_point)
    print("MC pion vertex: ", pion_vertex)

    print("Reconstructed Energy 1: ", pred_energy_1)
    print("Reconstructed Energy 2: ", pred_energy_2)
    print("MC energy 1: ", MC_energy_1)
    print("MC energy 2: ", MC_energy_2)

    rescaled_peak1_hcal_cluster = peak1_hcal_cluster.copy()
    rescaled_peak2_hcal_cluster = peak2_hcal_cluster.copy()

    rescaled_peak1_hcal_cluster[:, 3] = rescaled_peak1_hcal_cluster[:, 3]*(1/.02044)
    rescaled_peak2_hcal_cluster[:, 3] = rescaled_peak2_hcal_cluster[:, 3]*(1/.02044)

    peak1_cluster_with_ecal = np.append(rescaled_peak1_hcal_cluster, peak1_ecal_cluster, axis=0)
    peak2_cluster_with_ecal = np.append(rescaled_peak2_hcal_cluster, peak2_ecal_cluster, axis=0)

    both_gamma_clusters = np.append(peak1_cluster_with_ecal, peak2_cluster_with_ecal, axis=0)
    
    visualize(good_hcal_data, both_gamma_clusters, "HCal data 25mrad rotated", "Expanded HCal gamma clusters")

    visualize_with_lines(good_hcal_data, "HCal early peaks", rotated_verts, rotated_mom, neutron, gamma1, gamma2)

    visualize_with_provided_lines(both_gamma_clusters, "Testing vertex finding", peak1_ave, peak2_ave, peak1_mom, peak2_mom)
    
    

    visualize_with_pred_and_true(np.vstack((peak1_hcal_ave, peak2_hcal_ave, peak1_ecal_ave, peak2_ecal_ave)), "Average hits in ECal and HCal", peak1_ave, peak2_ave, gamma1_vertex, gamma2_vertex, peak1_mom, peak2_mom, gamma1_mom, gamma2_mom)

def draw_one_event(event, ecal_hits_x, ecal_hits_y, ecal_hits_z, hcal_hits_x, hcal_hits_y, hcal_hits_z, MC_vert_x, MC_vert_y, MC_vert_z, MC_mom_x, MC_mom_y, MC_momz, MC_pdg, MC_parents):
    rotated_good_event_ecal_hits = rotate_hits(np.column_stack((ecal_hits_x[event],ecal_hits_y[event],ecal_hits_z[event])))
    rotated_good_event_hcal_hits = rotate_hits(np.column_stack((hcal_hits_x[event],hcal_hits_y[event],hcal_hits_z[event])))

    rotated_verts = rotate_hits(np.column_stack((MC_vert_x[event],MC_vert_y[event],MC_vert_z[event])))
    rotated_mom = rotate_hits(np.column_stack((MC_mom_x[event],MC_mom_y[event],MC_mom_z[event])))

    neutron = find_particle_index(MC_pdg[event], MC_parents[event], 2112, 3122, 1)
    gamma1 = find_particle_index(MC_pdg[event], MC_parents[event], 22, 111, 1)
    gamma2 = find_particle_index(MC_pdg[event], MC_parents[event], 22, 111, 2)

    good_hcal_data = np.column_stack((rotated_good_event_hcal_hits, hcal_hits_energy[event].T))
    good_ecal_data = np.column_stack((rotated_good_event_ecal_hits, ecal_hits_energy[event].T))

    good_hcal_data = energy_cut(good_hcal_data, .0005)
    good_ecal_data = energy_cut(good_ecal_data, .005)

    #combined_data = np.concatenate((good_ecal_data, good_hcal_data), axis=0)

    ecal_peaks = ECal_peak_finding(good_ecal_data)
    hcal_peaks = HCal_peak_finding(good_hcal_data, 80)

    visualize_2D(good_ecal_data, ecal_peaks, "ECal data 25 mrad rotated, energy cut", "ECal Peak Finding")
    visualize(good_hcal_data, hcal_peaks[hcal_peaks[:,2] < 35950], "HCal data 25 mrad rotated, energy cut", "HCal Early Peak")
    visualize_with_lines(good_hcal_data, "HCal data 25 mrad rotated, energy cut", rotated_verts, rotated_mom, neutron, gamma1, gamma2)

def filter_based_on_ECal_peaks(ecal_peaks):
    if(ecal_peaks.shape[0] < 2):
        print("Too few ECal peaks.")
        return False
    if(ecal_peaks.shape[0] > 3):
        print("Too many ECal peaks.")
        return True
    return True

def filter_based_on_HCal_peaks(hcal_peaks):
    if(hcal_peaks.shape[0] < 2):
        return False
    return True


def dbscan_to_find_neutron(ecal_hits, hcal_hits):
    clusters = DBSCAN(eps=4, min_samples=4).fit(hcal_hits[:, :3])
    return clusters

def energy_cut(data, energy_threshold):
    return data[data[:, 3] > energy_threshold] 

def expand_ECal_cluster(seed_peak, ecal_hits, max_distance_xy):
    added_hits = set()

    cluster = seed_peak
    seed_position = cluster[:, :3]

    ecal_neighbors_info = NearestNeighbors(radius=50).fit(ecal_hits[:, :3])

    while True:
        new_hits = []
        for hit in cluster:
            ecal_neighbors = ecal_neighbors_info.radius_neighbors(hit[:3].reshape(1,-1), return_distance=False)[0]
            for idx in ecal_neighbors:
                neighbor_hit = ecal_hits[idx].reshape(1, -1)
                if tuple(neighbor_hit[0]) in added_hits:
                    continue
               
                neighbor_position = neighbor_hit[0, :3]
                neighbor_energy = neighbor_hit[0, 3]
                average_energy = np.mean(cluster[:, 3])
             
                distance_xy = np.linalg.norm(neighbor_position[:2] - seed_position[0,:2])
                
                if neighbor_energy <= average_energy and distance_xy <= max_distance_xy:
                    new_hits.append(neighbor_hit)
                    added_hits.add(tuple(neighbor_hit[0]))

        if not new_hits:
            break
        
        cluster = np.vstack([cluster] + new_hits)

    return cluster

def expand_cluster(seed_cluster, ecal_hits, hcal_hits, distance_cutoff, energy_jump_threshold, max_distance_z, max_distance_xy):
    added_hits = set()

    if(len(seed_cluster) == 1):
        cluster = seed_cluster[0].reshape(1,-1)
        added_hits.add(tuple(seed_cluster[0]))
    else:
        if(len(seed_cluster) == 2):
            cluster = seed_cluster[1].reshape(1, -1)
            added_hits.add(tuple(seed_cluster[1]))
        else:
            search_cluster = np.array(seed_cluster[1:])
            max_energy_index = np.argmax(search_cluster[:, 3])
            cluster = search_cluster[max_energy_index].reshape(1, -1)
    

    seed_position = cluster[:, :3][0]

    hcal_neighbors_info = NearestNeighbors(radius=distance_cutoff).fit(hcal_hits[:, :3])

    while True:
        new_hits = []
        for hit in cluster:
            hcal_neighbors = hcal_neighbors_info.radius_neighbors(hit[:3].reshape(1, -1), return_distance=False)[0]
            for idx in hcal_neighbors:
                mod_energy_jump_threshold = energy_jump_threshold
                neighbor_hit = hcal_hits[idx].reshape(1, -1)
                if tuple(neighbor_hit[0]) in added_hits:
                    continue
               
                neighbor_position = neighbor_hit[0, :3]
                neighbor_energy = neighbor_hit[0, 3]
                average_energy = np.mean(cluster[:, 3])
              
                distance_z = abs(neighbor_position[2] - seed_position[2])
                distance_xy = np.linalg.norm(neighbor_position[:2] - seed_position[:2])
                
                #TO-DO: add z distance into consideration here
                if distance_xy < 50:
                    mod_energy_jump_threshold = 3*energy_jump_threshold 

                if neighbor_energy <= average_energy + mod_energy_jump_threshold and distance_z <= max_distance_z and distance_xy <= max_distance_xy:
                    new_hits.append(neighbor_hit)
                    added_hits.add(tuple(neighbor_hit[0]))

        if not new_hits:
            break
        
        cluster = np.vstack([cluster] + new_hits)

    return cluster

def filter_cluster(clusters, deep_neutron_check):

    if(deep_neutron_check is True):
        if(len(clusters) > 2):
            print("Something went wrong, there are more than two gamma clusters")

        return clusters

    cluster_energies = []
    for cluster in clusters:
        total_energy = sum(point[3] for point in cluster[1:])  # Assumes energy is the 4th column
        cluster_energies.append(total_energy)

    most_energetic_index = np.argmax(cluster_energies)

    filtered_clusters = [cluster for i, cluster in enumerate(clusters) if i != most_energetic_index]

    if(len(filtered_clusters) > 2):
        print("Something went wrong, there are more than two gamma clusters")

    return filtered_clusters


def peak_merging(ecal_peaks, hcal_peaks):

    depth_cut = 35950

    clusters = []
    early_hcal_peaks = hcal_peaks[hcal_peaks[:,2] < depth_cut]

    deep_neutron_event = False

    if(ecal_peaks.shape[0] == 2 and hcal_peaks[np.argmax(hcal_peaks[:,3]),2] > depth_cut):
        deep_neutron_event = True

    distances_xy = sp.spatial.distance.cdist(early_hcal_peaks[:, :2], ecal_peaks[:, :2])

    xy_distance_cutoff = 80

    xy_distance_mask = distances_xy > xy_distance_cutoff
    rows_above_threshold = np.all(xy_distance_mask, axis=1)

    isolated_hcal_peak = early_hcal_peaks[rows_above_threshold,:]

    if isolated_hcal_peak.shape[0] > 0:
        #print("Isolated HCal peak event")
        #print(isolated_hcal_peak)
        clusters.append(isolated_hcal_peak)
        early_hcal_peaks = np.delete(early_hcal_peaks, rows_above_threshold, axis=0)
    
    distances = sp.spatial.distance.cdist(early_hcal_peaks[:, :3], ecal_peaks[:, :3])
    closest_ecal_indices = np.argmin(distances, axis=1)
    hcal_to_ecal_pairs = np.column_stack((np.arange(len(early_hcal_peaks)), closest_ecal_indices))

    ecal_to_hcal = {}
    for hcal_idx, ecal_idx in hcal_to_ecal_pairs:
        if ecal_idx in ecal_to_hcal:
            ecal_to_hcal[ecal_idx].append(hcal_idx)
        else:
            ecal_to_hcal[ecal_idx] = [hcal_idx]

    for ecal_idx, hcal_indices in ecal_to_hcal.items():
        cluster = [ecal_peaks[ecal_idx]]
        for hcal_idx in hcal_indices:
            cluster.append(early_hcal_peaks[hcal_idx])
        clusters.append(cluster)

    return clusters, deep_neutron_event

def HCal_peak_finding(hcal_data, radius):
    coordinates = hcal_data[:,:3]

    nbrs = NearestNeighbors(radius=radius, n_jobs=4).fit(coordinates)

    distances, indices = nbrs.radius_neighbors(coordinates)

    local_maxima = []
    local_maxima_neighbor_energy = []
    for i, neighbors in enumerate(indices):
        if all(hcal_data[i,3] > hcal_data[neighbor,3] for neighbor in neighbors if neighbor != i):
            local_maxima.append(hcal_data[i,:])
            local_maxima_neighbor_energy.append(np.sum(hcal_data[neighbors,3]))

    local_maxima = np.array(local_maxima)
    local_maxima_neighbor_energy = np.array(local_maxima_neighbor_energy)
    
    peak_energy_threshold = .005

    energy_cut_mask = local_maxima[:,3] > peak_energy_threshold
    local_maxima_neighbor_energy = local_maxima_neighbor_energy[energy_cut_mask]
    local_maxima = local_maxima[energy_cut_mask]

    neighbor_energy_threshold = .015
    local_maxima = local_maxima[local_maxima_neighbor_energy > neighbor_energy_threshold]

    return local_maxima

def ECal_peak_finding(ecal_data):
    coordinates = ecal_data[:,:3]

    nbrs = NearestNeighbors(radius=43).fit(coordinates)

    distances, indices = nbrs.radius_neighbors(coordinates)

    local_maxima = []
    local_maxima_neighbor_energy = []
    for i, neighbors in enumerate(indices):
        if all(ecal_data[i,3] > ecal_data[neighbor,3] for neighbor in neighbors if neighbor != i):
            local_maxima.append(ecal_data[i,:])
            local_maxima_neighbor_energy.append(np.sum(ecal_data[neighbors,3]))

    local_maxima = np.array(local_maxima)
    local_maxima_neighbor_energy = np.array(local_maxima_neighbor_energy)

    peak_energy_threshold = .1

    energy_cut_mask = local_maxima[:,3] > peak_energy_threshold
    local_maxima_neighbor_energy = local_maxima_neighbor_energy[energy_cut_mask]
    local_maxima = local_maxima[energy_cut_mask]

    neighbor_energy_threshold = .3
    local_maxima = local_maxima[local_maxima_neighbor_energy > neighbor_energy_threshold]

    return local_maxima

def rotate_hits(data):
    theta = .025
    rotation_mat = np.array([[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0], [math.sin(theta), 0, math.cos(theta)]])
    rotated_data = data@rotation_mat
    return rotated_data

def visualize_2D(original, reconstructed, title1, title2):
    fig = plt.figure(figsize=(20, 8))

    ax1 = fig.add_subplot(121)
    x_original = original[:, 0]
    y_original = original[:, 1]
    energy_original = original[:, 3]

    x_reconstructed = reconstructed[:, 0]
    y_reconstructed = reconstructed[:, 1]
    energy_reconstructed = reconstructed[:, 3]

    energy_min = min(np.min(energy_original), np.min(energy_reconstructed))
    energy_max = max(np.max(energy_original), np.max(energy_reconstructed))

    sc1 = ax1.scatter(x_original, y_original, c=energy_original, s=energy_original*100, cmap='viridis', alpha=0.6, vmin=energy_min, vmax=energy_max)
    ax1.set_title(title1)
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Energy [GeV]')

    ax2 = fig.add_subplot(122)
    
    sc2 = ax2.scatter(x_reconstructed, y_reconstructed, c=energy_reconstructed, s=energy_reconstructed*100, cmap='viridis', alpha=0.6, vmin=energy_min, vmax=energy_max)
    ax2.set_title(title2)
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Y [mm]')
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label('Energy [GeV]')

    x_min = min(np.min(x_original), np.min(x_reconstructed))
    x_max = max(np.max(x_original), np.max(x_reconstructed))
    y_min = min(np.min(y_original), np.min(y_reconstructed))
    y_max = max(np.max(y_original), np.max(y_reconstructed))

    ax1.set_xlim(-300, 300)
    ax1.set_ylim(-300, 300)
    ax2.set_xlim(-300, 300)
    ax2.set_ylim(-300, 300)

    plt.show()


def visualize(original, reconstructed, title1, title2):
    fig = plt.figure(figsize=(20, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    x_original = original[:, 0]
    y_original = original[:, 1]
    z_original = original[:, 2]
    energy_original = original[:, 3]

    x_reconstructed = reconstructed[:, 0]
    y_reconstructed = reconstructed[:, 1]
    z_reconstructed = reconstructed[:, 2]
    energy_reconstructed = reconstructed[:, 3]

    energy_min = min(np.min(energy_original), np.min(energy_reconstructed))
    energy_max = max(np.max(energy_original), np.max(energy_reconstructed))

    sc1 = ax1.scatter(x_original, y_original, z_original, c=energy_original, s=energy_original*100, cmap='viridis', alpha=0.6, vmin=energy_min, vmax=energy_max)
    ax1.set_title(title1)
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_zlabel('Z [mm]')
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Unscaled Energy [GeV]')

    ax2 = fig.add_subplot(122, projection='3d')

    sc2 = ax2.scatter(x_reconstructed, y_reconstructed, z_reconstructed, c=energy_reconstructed, s=energy_reconstructed*100, cmap='viridis', alpha=0.6, vmin=energy_min, vmax=energy_max)
    ax2.set_title(title2)
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Y [mm]')
    ax2.set_zlabel('Z [mm]')
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label('Unscaled Energy [GeV]')

    x_min = min(np.min(x_original), np.min(x_reconstructed))
    x_max = max(np.max(x_original), np.max(x_reconstructed))
    y_min = min(np.min(y_original), np.min(y_reconstructed))
    y_max = max(np.max(y_original), np.max(y_reconstructed))
    z_min = min(np.min(z_original), np.min(z_reconstructed))
    z_max = max(np.max(z_original), np.max(z_reconstructed))

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_zlim(z_min, z_max)

    plt.show()

def visualize_with_lines(original, title, verts, mom, neutron, gamma1, gamma2):
    fig = plt.figure(figsize=(20, 8))

    ax = fig.add_subplot(111, projection='3d')
    x_original = original[:, 0]
    y_original = original[:, 1]
    z_original = original[:, 2]
    energy_original = original[:, 3]

    t = np.linspace(0, 10000, 10000)
    neutron_line_x = mom[neutron, 0]*t + verts[neutron, 0]
    neutron_line_y = mom[neutron, 1]*t + verts[neutron, 1]
    neutron_line_z = mom[neutron, 2]*t + verts[neutron, 2]

    gamma1_line_x = mom[gamma1, 0]*t + verts[gamma1, 0]
    gamma1_line_y = mom[gamma1, 1]*t + verts[gamma1, 1]
    gamma1_line_z = mom[gamma1, 2]*t + verts[gamma1, 2]

    gamma2_line_x = mom[gamma2, 0]*t + verts[gamma2, 0]
    gamma2_line_y = mom[gamma2, 1]*t + verts[gamma2, 1]
    gamma2_line_z = mom[gamma2, 2]*t + verts[gamma2, 2]


    energy_min = min(np.min(energy_original), np.min(energy_original))
    energy_max = max(np.max(energy_original), np.max(energy_original))

    sc = ax.scatter(x_original, y_original, z_original, c=energy_original, s=energy_original*100, cmap='viridis', alpha=0.6, vmin=energy_min, vmax=energy_max)
    ax.set_title(title)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Unscaled Energy [GeV]')

    ax.plot(neutron_line_x, neutron_line_y, neutron_line_z, color='r', linewidth=2, label="Neutron Path")
    ax.plot(gamma1_line_x, gamma1_line_y, gamma1_line_z, color='b', linewidth=2, label="First Photon Path")
    ax.plot(gamma2_line_x, gamma2_line_y, gamma2_line_z, color='g', linewidth=2, label="Second Photon Path")

    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_zlim(35800, 37000)

    plt.legend()
    plt.show()

def visualize_with_provided_lines(original, title, vertex1, vertex2, mom1, mom2):
    fig = plt.figure(figsize=(20, 8))

    ax = fig.add_subplot(111, projection='3d')
    x_original = original[:, 0]
    y_original = original[:, 1]
    z_original = original[:, 2]
    energy_original = original[:, 3]

    t = np.linspace(0, 10000, 10000)
    line_1_x = mom1[0]*t + vertex1[0]
    line_1_y = mom1[1]*t + vertex1[1]
    line_1_z = mom1[2]*t + vertex1[2]

    line_2_x = mom2[0]*t + vertex2[0]
    line_2_y = mom2[1]*t + vertex2[1]
    line_2_z = mom2[2]*t + vertex2[2]

    energy_min = min(np.min(energy_original), np.min(energy_original))
    energy_max = max(np.max(energy_original), np.max(energy_original))

    sc = ax.scatter(x_original, y_original, z_original, c=energy_original, s=energy_original*100, cmap='viridis', alpha=0.6, vmin=energy_min, vmax=energy_max)
    ax.set_title(title)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Unscaled Energy [GeV]')

    ax.plot(line_1_x, line_1_y, line_1_z, color='r', linewidth=2, label="Line 1")
    ax.plot(line_2_x, line_2_y, line_2_z, color='b', linewidth=2, label="Line 2")

    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_zlim(35800, 37000)

    plt.legend()
    plt.show()

def visualize_with_pred_and_true(original, title, vertex1, vertex2, true_vertex1, true_vertex2, mom1, mom2, true_mom1, true_mom2):
    fig = plt.figure(figsize=(20, 8))

    ax = fig.add_subplot(111, projection='3d')
    x_original = original[:, 0]
    y_original = original[:, 1]
    z_original = original[:, 2]
    energy_original = original[:, 3]

    t = np.linspace(0, 10000, 10000)
    line_1_x = mom1[0]*t + vertex1[0]
    line_1_y = mom1[1]*t + vertex1[1]
    line_1_z = mom1[2]*t + vertex1[2]

    line_2_x = mom2[0]*t + vertex2[0]
    line_2_y = mom2[1]*t + vertex2[1]
    line_2_z = mom2[2]*t + vertex2[2]

    true_line_1_x = true_mom1[0]*t + true_vertex1[0]
    true_line_1_y = true_mom1[1]*t + true_vertex1[1]
    true_line_1_z = true_mom1[2]*t + true_vertex1[2]

    true_line_2_x = true_mom2[0]*t + true_vertex2[0]
    true_line_2_y = true_mom2[1]*t + true_vertex2[1]
    true_line_2_z = true_mom2[2]*t + true_vertex2[2]

    energy_min = min(np.min(energy_original), np.min(energy_original))
    energy_max = max(np.max(energy_original), np.max(energy_original))

    sc = ax.scatter(x_original, y_original, z_original, c=energy_original, s=energy_original*100, cmap='viridis', alpha=0.6, vmin=energy_min, vmax=energy_max)
    ax.set_title(title)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Unscaled Energy [GeV]')

    ax.plot(line_1_x, line_1_y, line_1_z, color='r', linewidth=2, label="Line 1")
    ax.plot(line_2_x, line_2_y, line_2_z, color='b', linewidth=2, label="Line 2")
    ax.plot(true_line_1_x, true_line_1_y, true_line_1_z, color='g', linewidth=2, label="True Line 1")
    ax.plot(true_line_2_x, true_line_2_y, true_line_2_z, color='y', linewidth=2, label="True Line 2")

    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_zlim(35800, 37000)

    plt.legend()
    plt.show()

infile="/home/alessio/RIKENSUMMER/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r100.root"
#infile="/home/alessio/RIKENSUMMER/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r211.root"
#infile = "/home/alessio/RIKENSUMMER/angle_data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r303.root"
#infile = "/home/alessio/RIKENSUMMER/angle_data/*.root"

#used for using a file with event numbers
log_file = '/home/alessio/RIKENSUMMER/myscripts/r100_good_events_log.txt'
#log_file = '/home/alessio/RIKENSUMMER/myscripts/all_angular_data_log.txt'

indices = []

with open(log_file, 'r') as file:
    for line in file:
        index = int(line.strip())

        indices.append(index)

events_tree = up.open(infile)["events"]

hcal_hits_x = events_tree["HcalFarForwardZDCHits.position.x"].array(library="np")
hcal_hits_y = events_tree["HcalFarForwardZDCHits.position.y"].array(library="np")
hcal_hits_z = events_tree["HcalFarForwardZDCHits.position.z"].array(library="np")
hcal_hits_energy = events_tree["HcalFarForwardZDCHits.energy"].array(library="np")

ecal_hits_x = events_tree["EcalFarForwardZDCHits.position.x"].array(library="np")
ecal_hits_y = events_tree["EcalFarForwardZDCHits.position.y"].array(library="np")
ecal_hits_z = events_tree["EcalFarForwardZDCHits.position.z"].array(library="np")
ecal_hits_energy = events_tree["EcalFarForwardZDCHits.energy"].array(library="np")

MC_vert_x = events_tree["MCParticles.vertex.x"].array(library="np")
MC_vert_y = events_tree["MCParticles.vertex.y"].array(library="np")
MC_vert_z = events_tree["MCParticles.vertex.z"].array(library="np")

MC_mom_x = events_tree["MCParticles.momentum.x"].array(library="np")
MC_mom_y = events_tree["MCParticles.momentum.y"].array(library="np")
MC_mom_z = events_tree["MCParticles.momentum.z"].array(library="np")

MC_pdg =  events_tree["MCParticles.PDG"].array(library="np")
MC_parents = events_tree["_MCParticles_parents.index"].array(library="np")

looping_through_events(indices, ecal_hits_x, ecal_hits_y, ecal_hits_z, hcal_hits_x, hcal_hits_y, hcal_hits_z, MC_vert_x, MC_vert_y, MC_vert_z, MC_mom_x, MC_mom_y, MC_mom_z, MC_pdg, MC_parents)
#draw_one_event(120, ecal_hits_x, ecal_hits_y, ecal_hits_z, hcal_hits_x, hcal_hits_y, hcal_hits_z, MC_vert_x, MC_vert_y, MC_vert_z, MC_mom_x, MC_mom_y, MC_mom_z, MC_pdg, MC_parents)
#find_gamma_one_event(177, ecal_hits_x, ecal_hits_y, ecal_hits_z, hcal_hits_x, hcal_hits_y, hcal_hits_z, MC_vert_x, MC_vert_y, MC_vert_z, MC_mom_x, MC_mom_y, MC_mom_z, MC_pdg, MC_parents)
