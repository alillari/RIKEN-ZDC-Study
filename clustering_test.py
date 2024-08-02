import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpld3
import uproot as up
import math
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import hdbscan
import scipy as sp

def hdbscan_on_one_event(event, ecal_hits_x, ecal_hits_y, ecal_hits_z, hcal_hits_x, hcal_hits_y, hcal_hits_z):
    clusterer  = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=1)

    rotated_good_event_ecal_hits = rotate_hits(np.column_stack((ecal_hits_x[event],ecal_hits_y[event],ecal_hits_z[event])))
    rotated_good_event_hcal_hits = rotate_hits(np.column_stack((hcal_hits_x[event],hcal_hits_y[event],hcal_hits_z[event])))

    good_hcal_data = np.column_stack((rotated_good_event_hcal_hits, hcal_hits_energy[event].T))
    good_ecal_data = np.column_stack((rotated_good_event_ecal_hits, ecal_hits_energy[event].T))

    good_hcal_data = energy_cut(good_hcal_data, .0005)
    good_ecal_data = energy_cut(good_ecal_data, .005)

    cluster_labels = clusterer.fit_predict(good_hcal_data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (cluster_labels == k)
        xyz = good_hcal_data[class_member_mask]

        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[col], marker='o', s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of HCal Hits with Cluster Colors')

    plt.show()

def looping_through_events(indices, ecal_hits_x, ecal_hits_y, ecal_hits_z, hcal_hits_x, hcal_hits_y, hcal_hits_z):
    for event in indices:
        looping_rotated_ecal_hits = rotate_hits(np.column_stack((ecal_hits_x[event],ecal_hits_y[event],ecal_hits_z[event])))
        looping_rotated_hcal_hits = rotate_hits(np.column_stack((hcal_hits_x[event],hcal_hits_y[event],hcal_hits_z[event])))

        looping_ecal_data = np.column_stack((looping_rotated_ecal_hits, ecal_hits_energy[event].T)) 
        looping_hcal_data = np.column_stack((looping_rotated_hcal_hits, hcal_hits_energy[event].T))

        looping_ecal_data = energy_cut(looping_ecal_data, .005)
        looping_hcal_data = energy_cut(looping_hcal_data, .0005)

        if looping_ecal_data.shape[0] == 0 or looping_hcal_data.shape[0] == 0:
            continue

        looping_ecal_peaks = ECal_peak_finding(looping_ecal_data)
        looping_hcal_peaks = HCal_peak_finding(looping_hcal_data, 80)
        if filter_based_on_ECal_peaks(looping_ecal_peaks) and filter_based_on_HCal_peaks(looping_hcal_data[looping_hcal_data[:,2] < 36050]):
            print(event)

def find_gamma_one_event(event, ecal_hits_x, ecal_hits_y, ecal_hits_z, hcal_hits_x, hcal_hits_y, hcal_hits_z):
    rotated_good_event_ecal_hits = rotate_hits(np.column_stack((ecal_hits_x[event],ecal_hits_y[event],ecal_hits_z[event])))
    rotated_good_event_hcal_hits = rotate_hits(np.column_stack((hcal_hits_x[event],hcal_hits_y[event],hcal_hits_z[event])))

    good_hcal_data = np.column_stack((rotated_good_event_hcal_hits, hcal_hits_energy[event].T))
    good_ecal_data = np.column_stack((rotated_good_event_ecal_hits, ecal_hits_energy[event].T))

    good_hcal_data = energy_cut(good_hcal_data, .0005)
    good_ecal_data = energy_cut(good_ecal_data, .005)

    ecal_peaks = ECal_peak_finding(good_ecal_data)
    hcal_peaks = HCal_peak_finding(good_hcal_data, 80)

    proto_clusters, neutron_check = peak_merging(ecal_peaks, hcal_peaks)
    proto_gamma_clusters = filter_cluster(proto_clusters, neutron_check)
    gamma0_cluster = expand_cluster(proto_gamma_clusters[0], good_ecal_data, good_hcal_data, 37, .01, 400, 80)
    gamma1_cluster = expand_cluster(proto_gamma_clusters[1], good_ecal_data, good_hcal_data, 37, .01, 400, 80)
    both_gamma_clusters = np.append(gamma0_cluster, gamma1_cluster, axis=0)
    visualize(good_hcal_data, both_gamma_clusters, "HCal data 25mrad rotated", "Expanded HCal gamma clusters")

def draw_one_event(event, ecal_hits_x, ecal_hits_y, ecal_hits_z, hcal_hits_x, hcal_hits_y, hcal_hits_z):
    rotated_good_event_ecal_hits = rotate_hits(np.column_stack((ecal_hits_x[event],ecal_hits_y[event],ecal_hits_z[event])))
    rotated_good_event_hcal_hits = rotate_hits(np.column_stack((hcal_hits_x[event],hcal_hits_y[event],hcal_hits_z[event])))

    good_hcal_data = np.column_stack((rotated_good_event_hcal_hits, hcal_hits_energy[event].T))
    good_ecal_data = np.column_stack((rotated_good_event_ecal_hits, ecal_hits_energy[event].T))

    good_hcal_data = energy_cut(good_hcal_data, .0005)
    good_ecal_data = energy_cut(good_ecal_data, .005)

    combined_data = np.concatenate((good_ecal_data, good_hcal_data), axis=0)

    ecal_peaks = ECal_peak_finding(good_ecal_data)
    hcal_peaks = HCal_peak_finding(good_hcal_data, 80)

    visualize_2D(good_ecal_data, ecal_peaks, "ECal data 25 mrad rotated, energy cut", "ECal Peak Finding")
    visualize(good_hcal_data, hcal_peaks[hcal_peaks[:,2] < 36050], "HCal data 25 mrad rotated, energy cut", "HCal Early Peak")


def filter_based_on_ECal_peaks(ecal_peaks):
    if(ecal_peaks.shape[0] < 2):
        #print("Too few ECal peaks.")
        return False
    if(ecal_peaks.shape[0] > 3):
        print("Too many ECal peaks.")
        return True
    return True

def filter_based_on_HCal_peaks(hcal_peaks):
    if(hcal_peaks.shape[0] < 2):
        return False
    return True


def define_core_points(hcal_hits, epsilon, min_neighbors):
    neighbors_info = NearestNeighbors(radius=epsilon).fit(hcal_hits[:, :3])
    core_points = set()

    for idx, point in enumerate(hcal_hits):
        neighbors = neighbors_info.radius_neighbors(point[:3].reshape(1, -1), return_distance=False)[0]
        if len(neighbors) > min_neighbors:
            core_points.add(tuple(point))
    
    return core_points

def dbscan_to_find_neutron(ecal_hits, hcal_hits):
    clusters = DBSCAN(eps=4, min_samples=4).fit(hcal_hits[:, :3])
    return clusters

def energy_cut(data, energy_threshold):
    return data[data[:, 3] > energy_threshold] 

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
                if distance_xy < 30:
                    mod_energy_jump_threshold = 2*energy_jump_threshold 

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

    depth_cut = 36050

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
    
    peak_energy_threshold = .01

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


infile="/home/alessio/RIKENSUMMER/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r100.root"

log_file = '/home/alessio/RIKENSUMMER/myscripts/r100_good_events_log.txt'

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

#looping_through_events(indices, ecal_hits_x, ecal_hits_y, ecal_hits_z, hcal_hits_x, hcal_hits_y, hcal_hits_z)
#draw_one_event(9997, ecal_hits_x, ecal_hits_y, ecal_hits_z, hcal_hits_x, hcal_hits_y, hcal_hits_z)
find_gamma_one_event(14, ecal_hits_x, ecal_hits_y, ecal_hits_z, hcal_hits_x, hcal_hits_y, hcal_hits_z)
