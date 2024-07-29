import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import uproot as up
import math
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import scipy as sp

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
    cluster = seed_cluster[1].reshape(1, -1)
    seed_position = cluster[:, :3][0]

    hcal_neighbors_info = NearestNeighbors(radius=distance_cutoff).fit(hcal_hits[:, :3])
    
    added_hits = set()
    added_hits.add(tuple(seed_cluster[1]))

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

    clusters = []
    early_hcal_peaks = hcal_peaks[hcal_peaks[:,2] < 36050]

    deep_neutron_event = False

    if(ecal_peaks.shape[0] == 2 and hcal_peaks[np.argmax(hcal_peaks[:,3]),2] > 36050):
        deep_neutron_event = True

    distances = sp.spatial.distance.cdist(early_hcal_peaks[:, :3], ecal_peaks[:, :3])
    distances_xy = sp.spatial.distance.cdist(early_hcal_peaks[:, :2], ecal_peaks[:, :2])

    xy_distance_cutoff = 90

    xy_distance_mask = distances_xy > xy_distance_cutoff
    rows_above_threshold = np.all(xy_distance_mask, axis=1)

    isolated_hcal_peak = early_hcal_peaks[rows_above_threshold,:]

    if isolated_hcal_peak.shape[0] > 0:
        print("Isolated HCal peak event")

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

    nbrs = NearestNeighbors(radius=radius).fit(coordinates)

    distances, indices = nbrs.radius_neighbors(coordinates)

    local_maxima = []
    local_maxima_neighbor_energy = []
    for i, neighbors in enumerate(indices):
        if all(hcal_data[i,3] > hcal_data[neighbor,3] for neighbor in neighbors if neighbor != i):
            local_maxima.append(hcal_data[i,:])
            local_maxima_neighbor_energy.append(np.sum(hcal_data[neighbors,3]))

    local_maxima = np.array(local_maxima)
    local_maxima = local_maxima[local_maxima[:,3] > .01]

    #TO-DO: epxeriment with adding a neighboring energy cut

    return local_maxima

def ECal_peak_finding(ecal_data):
    coordinates = ecal_data[:,:3]

    nbrs = NearestNeighbors(radius=43).fit(coordinates)

    distances, indices = nbrs.radius_neighbors(coordinates)

    local_maxima = []
    for i, neighbors in enumerate(indices):
        if all(ecal_data[i,3] > ecal_data[neighbor,3] for neighbor in neighbors if neighbor != i):
            local_maxima.append(ecal_data[i,:])

    local_maxima = np.array(local_maxima)
    local_maxima = local_maxima[local_maxima[:,3] > .15]

    return np.array(local_maxima)

def rotate_hits(data):
    theta = .025
    rotation_mat = np.array([[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0], [math.sin(theta), 0, math.cos(theta)]])
    rotated_data = data@rotation_mat
    return rotated_data

def visualize_2D(original, reconstructed):
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
    ax1.set_title('ECal 25mrad filtered hits')
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Energy [GeV]')

    ax2 = fig.add_subplot(122)
    
    sc2 = ax2.scatter(x_reconstructed, y_reconstructed, c=energy_reconstructed, s=energy_reconstructed*100, cmap='viridis', alpha=0.6, vmin=energy_min, vmax=energy_max)
    ax2.set_title('ECal Peaks')
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


def visualize(original, reconstructed):
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
    ax1.set_title('HCal 25mrad rotated filtered hits')
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_zlabel('Z [mm]')
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Unscaled Energy [GeV]')

    ax2 = fig.add_subplot(122, projection='3d')

    sc2 = ax2.scatter(x_reconstructed, y_reconstructed, z_reconstructed, c=energy_reconstructed, s=energy_reconstructed*100, cmap='viridis', alpha=0.6, vmin=energy_min, vmax=energy_max)
    ax2.set_title('HCal peak clustering')
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

#infile="" 
infile="/home/alessio/RIKENSUMMER/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r100.root"

events_tree = up.open(infile)["events"]

hcal_hits_x = events_tree["HcalFarForwardZDCHits.position.x"].array(library="np")
hcal_hits_y = events_tree["HcalFarForwardZDCHits.position.y"].array(library="np")
hcal_hits_z = events_tree["HcalFarForwardZDCHits.position.z"].array(library="np")
hcal_hits_energy = events_tree["HcalFarForwardZDCHits.energy"].array(library="np")

ecal_hits_x = events_tree["EcalFarForwardZDCHits.position.x"].array(library="np")
ecal_hits_y = events_tree["EcalFarForwardZDCHits.position.y"].array(library="np")
ecal_hits_z = events_tree["EcalFarForwardZDCHits.position.z"].array(library="np")
ecal_hits_energy = events_tree["EcalFarForwardZDCHits.energy"].array(library="np")

one_good_event = 5

rotated_good_event_ecal_hits = rotate_hits(np.column_stack((ecal_hits_x[one_good_event],ecal_hits_y[one_good_event],ecal_hits_z[one_good_event])))
rotated_good_event_hcal_hits = rotate_hits(np.column_stack((hcal_hits_x[one_good_event],hcal_hits_y[one_good_event],hcal_hits_z[one_good_event])))

good_hcal_data = np.column_stack((rotated_good_event_hcal_hits, hcal_hits_energy[one_good_event].T))
good_ecal_data = np.column_stack((rotated_good_event_ecal_hits, ecal_hits_energy[one_good_event].T))

good_hcal_data = energy_cut(good_hcal_data, .0005)
good_ecal_data = energy_cut(good_ecal_data, .005)

#combined_data = np.concatenate((good_ecal_data, good_hcal_data), axis=0)

ecal_peaks = ECal_peak_finding(good_ecal_data)
hcal_peaks = HCal_peak_finding(good_hcal_data, 80)

visualize_2D(good_ecal_data, ecal_peaks)
visualize(good_hcal_data, hcal_peaks)

proto_clusters, neutron_check = peak_merging(ecal_peaks, hcal_peaks)
proto_gamma_clusters = filter_cluster(proto_clusters, neutron_check)
gamma0_cluster = expand_cluster(proto_gamma_clusters[0], good_ecal_data, good_hcal_data, 37, .02, 400, 80)
gamma1_cluster = expand_cluster(proto_gamma_clusters[1], good_ecal_data, good_hcal_data, 37, .02, 400, 80)
both_gamma_clusters = np.append(gamma0_cluster, gamma1_cluster, axis=0)
visualize(good_hcal_data, both_gamma_clusters)
