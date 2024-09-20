#TODO: use the centroid of the ECal clusters to find the cores for KMeans and force it to break merged 
#clusters with core locations before certain depth in the ZDC
#TODO: fix the extract angles function (no need to extract data, just calculate everything)
#TODO: fix the rotate function 

import numpy as np # type: ignore
import matplotlib # type: ignore
import matplotlib.pyplot as plt # type: ignore
from collections import Counter # type: ignore
from mpl_toolkits.mplot3d import Axes3D # type: ignore
from dbscan_mod import DBSCAN_S1, DBSCAN_S2
from dbscan import DBSCAN
import statistics as stat 

#Rotate the hit data to the particle's reference frame 
def rotate(event, angle=0.025): 
    rot_lst = []  
    for hit in event:  
        x = hit[0] * np.cos(angle) + hit[2] * np.sin(angle)
        z = -hit[0] * np.sin(angle) + hit[2] * np.cos(angle)
        if len(hit) == 4:
            rot_lst.append((x, hit[1], z, hit[3]))
        else: 
            rot_lst.append((x, hit[1], z))
    return rot_lst


#Define the ECal plane and HCal plane
#ECal
ZDC_norm = np.array([np.sin(-0.025), 0., np.cos(0.025)])
ecal_pt = np.array([35500*np.sin(-0.025), 0., 35500*np.cos(0.025)])
#HCal
hcal_pt0 = np.array([36607.5*np.sin(-0.025), 0., 36607.5*np.cos(0.025)])
hcal_pts = [35800] 
for i in range(1, 64):
    lay_thick = (37400 - 35800) / 64
    r0 = 36607.5 
    lay_loc = 35800 + lay_thick*i
    hcal_pts.append(lay_loc)


#Count the total number of events
def total_count(chain):
    count = 0 
    for event in chain: 
        count += 1
    return count


#Find the euclidean distance between two points in 2D
def euclidean_dist_2D(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1])**2)


#Find the euclidean distance between two points in 3D
def euclidean_dist_3D(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)
        

#Merge clusters from step 1 and 2
def merge_dicts_unique_keys(dict1, dict2):
    merged_dict = dict1.copy()  # Start with a copy of the first dictionary
    for key, value in dict2.items():
        new_key = key
        # Ensure the key is unique
        while new_key in merged_dict:
            new_key = str(new_key) + '_1'  # Append '_1' to the key until it's unique
        merged_dict[new_key] = value
    return merged_dict


#Get particle specific (angle, hit and distance) data from root files
#Output is a list of events. Each event is a list of angles and distance between particles
def extract_angles(chain): 
    e_num = 0 #Counter for event tracking
    event_data = [] 
    good_evt_list = []

    for event in chain:
        MCParticles_vec = event.MCParticles
        temp_list = [] 
        
        for i in range(MCParticles_vec.size()):
            particle = MCParticles_vec.at(i)
            pid = particle.PDG
            mom_vec = [particle.momentum.x, particle.momentum.y, particle.momentum.z]
            mag_mom_vec = np.linalg.norm(mom_vec)
            vertex_vec = [particle.vertex.x, particle.vertex.y, particle.vertex.z]
            if (pid == 22 or pid == 2112 or pid == 3122):  #Photon, neutron and Lambda check 
                temp_list.append((pid , mom_vec/mag_mom_vec, vertex_vec))
            else: 
                None

        if len(temp_list) == 4: #CUT 1: Double photon, neutron events only
            #angle between Lambda and vector running through center of ZDC
            #l_angle  = np.arccos(np.dot(temp_list[0][1], ecal_norm))
            #angle between neutron and Lambda
            n_l_angle = np.arccos(np.dot(temp_list[0][1], temp_list[1][1]))

            #angle and distance between neutron and closest photon in ECal plane
            inter_ptn = intersection(ZDC_norm, ecal_pt, temp_list[1][1], temp_list[1][2])
            inter_ptg1 = intersection(ZDC_norm, ecal_pt, temp_list[2][1], temp_list[2][2])
            inter_ptg2 = intersection(ZDC_norm, ecal_pt, temp_list[3][1], temp_list[3][2])
            inter_list = [inter_ptn, inter_ptg1, inter_ptg2] 
            #make sure the hit is acutally in ZDC 
            for pt in inter_list:
                if (-1200 < pt[0] < -590) and (-300 < pt[1] < 300): 
                    dist_n_g1 = np.linalg.norm(inter_ptn - inter_ptg1)
                    dist_n_g2 = np.linalg.norm(inter_ptn - inter_ptg2)
                    if dist_n_g1 < dist_n_g2:
                        n_g_angle = np.arccos(np.dot(temp_list[1][1], temp_list[2][1]))
                        dist = dist_n_g1
                    else: 
                        n_g_angle = np.arccos(np.dot(temp_list[1][1], temp_list[3][1]))
                        dist = dist_n_g1

        event_data.append((n_l_angle, n_g_angle, dist))
        good_evt_list.append(e_num)
        e_num += 1        
        
    else:
        None

    return (event_data, good_evt_list)


def merge_clusters(clusters, distance_threshold):
    def compute_centroid(points):
        return np.mean([p[:3] for p in points], axis=0)

    def is_within_threshold(centroid1, centroid2):
        return np.linalg.norm(centroid1 - centroid2) <= distance_threshold

    def find_closest_cluster(centroid, cluster_centroids):
        distances = {k: np.linalg.norm(centroid - c) for k, c in cluster_centroids.items()}
        closest_cluster = min(distances, key=distances.get)
        return closest_cluster

    #Step 1: Filter out clusters with keys starting with "-1"
    valid_clusters = {k: v for k, v in clusters.items() if not str(k).startswith('-1')}
    ignored_clusters = {k: v for k, v in clusters.items() if str(k).startswith('-1')}

    #Step 2: Separate large and small clusters
    large_clusters = {k: v for k, v in valid_clusters.items() if len(v) > 4}
    small_clusters = {k: v for k, v in valid_clusters.items() if len(v) <= 4}

    #Step 3: Merge large clusters based on centroid distance
    merged_clusters = []
    used_keys = set()

    for key1, points1 in large_clusters.items():
        if key1 in used_keys:
            continue

        centroid1 = compute_centroid(points1)
        new_cluster = points1

        for key2, points2 in large_clusters.items():
            if key1 != key2 and key2 not in used_keys:
                centroid2 = compute_centroid(points2)
                if is_within_threshold(centroid1, centroid2):
                    new_cluster.extend(points2)
                    used_keys.add(key2)

        used_keys.add(key1)
        merged_clusters.append(new_cluster)

    #Compute centroids for the merged large clusters
    merged_cluster_centroids = {i: compute_centroid(cluster) for i, cluster in enumerate(merged_clusters)}

    #Step 4: Merge remaining small clusters with the closest large cluster
    for small_points in small_clusters.values():
        small_centroid = compute_centroid(small_points)
        if merged_cluster_centroids:
            closest_cluster_idx = find_closest_cluster(small_centroid, merged_cluster_centroids)
            merged_clusters[closest_cluster_idx].extend(small_points)
        else:
            #If there are no large clusters, treat the small clusters as merged clusters
            merged_clusters.append(small_points)

    #Step 5: Combine merged clusters with ignored clusters
    final_clusters = {i: cluster for i, cluster in enumerate(merged_clusters)}
    for key, points in ignored_clusters.items():
        final_clusters[key] = points

    return final_clusters


def step_1_check(clusters_1):
    #Find the largest cluster
    largest_cluster_key = max(clusters_1, key=lambda k: len(clusters_1[k]) if k != -1 else 0)
    largest_cluster = clusters_1[largest_cluster_key]

    #Compute the core for all clusters aside from the largest one and the one with key -1
    cores = {}
    keys_to_remove = []  # List to keep track of clusters to be removed

    for key, cluster in clusters_1.items():
        if key != largest_cluster_key and key != -1:
            core = np.mean([p[:3] for p in cluster], axis=0)
            cores[key] = core

    #Merge clusters with cores where the z position is greater than 36000 into the largest cluster
    for key, core in cores.items():
        if core[2] > 36000:  # z position
            largest_cluster.extend(clusters_1[key])
            keys_to_remove.append(key)

    #Remove the clusters identified for removal
    for key in keys_to_remove:
        del clusters_1[key]

    #Add the largest cluster back to the dictionary
    clusters_1[largest_cluster_key] = largest_cluster
    return clusters_1


def calculate_weighted_center(cluster):
    total_weight = sum(hit[3] for hit in cluster)  #Sum of energies
    weighted_x = sum(hit[0] * hit[3] for hit in cluster) / total_weight  #weighted average of x
    weighted_y = sum(hit[1] * hit[3] for hit in cluster) / total_weight  #weighted average of y
    weighted_z = sum(hit[2] * hit[3] for hit in cluster) / total_weight  #weighted average of z
    point = [weighted_x, weighted_y, weighted_z]
    return point

#Orthogonal distance regression - used to find line of best fot for photons using HCal hits
def odr_4d(cluster): 
    #Prep the data
    x = np.array([hit[0] for hit in cluster])
    y = np.array([hit[1] for hit in cluster])
    z = np.array([hit[2] for hit in cluster])
    energy = np.array([hit[3] for hit in cluster])
    data = np.concatenate((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis], energy[:, np.newaxis]), axis=1)
    #data = np.concatenate((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), axis=1)

    #Calculate the mean 
    datamean = data.mean(axis=0)

    #Do a SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(data - datamean)
    #uu, dd, vv = np.linalg.svd(data - datamean, full_matrices=False)

    #Find points on the line (just need 2)
    linepts = vv[0] * np.mgrid[-100:100:2j][:, np.newaxis]
    linepts += datamean

    return linepts


def error_bars(layer):
    error = [] 
    # Using zip to split list by item of interest
    x, y, z, energy, tag = zip(*layer)
    x = list(x)
    x = np.array(x)
    y = list(y)
    y = np.array(y)
    z = list(z)
    z = np.array(z)
    energy = list(energy)

    if len(energy) == 1: 
        x_err = 0 
        y_err = 0
        z_err = 0
    else: 
        #energy = scale_energy(energy)
        #energy = np.array(energy)
        #weighted_mean_x = np.average(x, weights=energy)
        #x_err = np.average((x - weighted_mean_x)**2, weights=energy)
        #weighted_mean_y = np.average(y, weights=energy)
        #y_err = np.average((y - weighted_mean_y)**2, weights=energy)
        #weighted_mean_z = np.average(z, weights=energy)
        #z_err = np.average((z - weighted_mean_z)**2, weights=energy)
        x_err = np.std(x)
        y_err = np.std(y)
        z_err = np.std(z)


    error = [x_err, y_err, z_err]
    return error


def split_by_layer(cluster, hcal_pts):
    #Initialize lists for each layer
    layers = [[] for _ in range(len(hcal_pts) - 1)]
    for hit in cluster:
        z_position = hit[2]
        #Check which layer the hit belongs to
        for i in range(len(hcal_pts) - 1):
            if hcal_pts[i] <= z_position < hcal_pts[i + 1]:
                layers[i].append(hit)
                break
    #Filter out empty lists
    layers = [layer for layer in layers if layer]
    return layers
 
 
#Combine the photon clusters in the ECal and HCal 
def pair_clusters(clusters_mod):
    #Split photon clusters by ECal and HCal and copy the original seperate clusters into another dictionary 
    h_clusters = [] 
    e_clusters = [] 
    clusters_g = {} 
    for (key, cluster) in list(clusters_mod.items()):
        #Ignore the noise clusters
        if (isinstance(key, str) and key.startswith('-1')) or key == -1: 
            continue
        else: 
            core = np.array(np.mean([p[:3] for p in cluster], axis=0))
            #Ignore the neutron clusters
            if core[2] >= 36100:
                continue
            else: 
                hit = cluster[0]
                if hit[-1] == 'h': 
                    h_clusters.append(cluster)
                else: 
                    e_clusters.append(cluster)
                clusters_g[key] = clusters_mod.pop(key)

    #Combine the closest ECal and HCal clusters
    combined = [] 
    for h_cluster in h_clusters:
        h_x = h_cluster[3][0]
        closest_e_cluster = min(e_clusters, key=lambda e_cluster: abs(e_cluster[3][0] - h_x))
        temp_list = h_cluster + closest_e_cluster
        combined.append(temp_list)
        e_clusters.remove(closest_e_cluster)

    #Add the combined clusters back into clusters mod 
    clusters_mod[90] = combined[0]
    clusters_mod[91] = combined[1]

    return clusters_mod


def find_points_odr(clusters_mod): 
    clusters_mod = pair_clusters(clusters_mod)
    #Find points needed to make photon vectors
    cluster_points_n = [] 
    cluster_points_g = []
    #Find the photon clusters and seperate them in ECal and HCal 
    for key, cluster in clusters_mod.items():
        temp_list_n = []
        temp_list_g = [] 
        #ignore the noise clusters
        if (isinstance(key, str) and key.startswith('-1')) or key == -1: 
            continue
        else: 
            core = np.array(np.mean([p[:3] for p in cluster], axis=0))
            #Neutron 
            if core[2] >= 36100:
                point = odr_4d(cluster)
                temp_list_n = [key, "n", "h", point] 
            #Photons
            else: 
                point = odr_4d(cluster)
                temp_list_g = [key, "g", "e+h", point]
                
            cluster_points_g.append(temp_list_g)
            cluster_points_n.append(temp_list_n)

    cluster_points_n = [x for x in cluster_points_n if x]
    cluster_points_g = [x for x in cluster_points_g if x]
    return (cluster_points_n, cluster_points_g)


def find_points(clusters_mod): 
    #Find points needed to make photon vectors
    cluster_points_n = [] 
    cluster_points_g = []
    error_list_n = []
    error_list_g = []
    #Find the photon clusters and seperate them in ECal and HCal 
    for i, (key, cluster) in enumerate(clusters_mod.items()):
        temp_list_n = []
        temp_list_g = [] 
        #ignore the noise clusters
        if (isinstance(key, str) and key.startswith('-1')) or key == -1: 
            None
        else: 
            Photon, Neutron, Ecal, Hcal = False, False, False, False
            core = np.array(np.mean([p[:3] for p in cluster], axis=0))
            if core[2] >= 36100:
                Neutron = True
                for hit in cluster: 
                    if hit[4] == 'h': 
                        Hcal = True
                    else: 
                        Ecal = True
            else: 
                Photon = True
                for hit in cluster: 
                    if hit[4] == 'h': 
                        Hcal = True
                    else: 
                        Ecal = True

            #ECal resolution is not good so ignore the ECal cluster
            if (Photon == True) and (Ecal == True):
                None 

            #Find points for line of best fit of photon hits in HCal
            elif (Photon == True) and (Ecal != True): 
                point = odr_4d(cluster)
                #print(key,"\n", point, "\n****\n")
                #cluster = [list(elem) for elem in cluster]
                temp_list_g = [key, "g", "h", point]
            
            #Split neutron hits by layer in Hcal and find the weighted average position
            elif (Neutron == True) and (Hcal == True): 
                temp_list_n = [key, "n", "h"] 
                temp_error = [] 
                hits_by_layer = split_by_layer(cluster, hcal_pts)
                for layer in hits_by_layer: 
                    #print(layer)
                    point = calculate_weighted_center(layer)
                    error = error_bars(layer)
                    temp_list_h = [point]
                    temp_list_n.extend(temp_list_h)
                    temp_error.append(error)
                
            cluster_points_g.append(temp_list_g)
            cluster_points_n.append(temp_list_n)
            error_list_n.append(temp_error)

    cluster_points_n = [x for x in cluster_points_n if x]
    cluster_points_n = cluster_points_n[0]
    cluster_points_g = [x for x in cluster_points_g if x]
    return (cluster_points_n, cluster_points_g, error_list_n, error_list_g)


def find_vertex(cluster_points_g, error_g):
    #Separate the HCal and ECal clusters
    for (i, cluster) in enumerate(cluster_points_g): 
        cluster.append(error_g[i])
    h_clusters = [cluster for cluster in cluster_points_g if cluster[2] == 'h']
    e_clusters = [cluster for cluster in cluster_points_g if cluster[2] == 'e']

    #Find the clusters that are closest to each other and pair their points to define the lines 
    vecs = [] 
    vecs_err = []
    for h_cluster in h_clusters:
        h_x = h_cluster[3][0]
        closest_e_cluster = min(e_clusters, key=lambda e_cluster: abs(e_cluster[3][0] - h_x))
        vec = [closest_e_cluster[3], h_cluster[3]]
        vec_err = [closest_e_cluster[4],h_cluster[4]]
        vecs.append(vec)
        vecs_err.append(vec_err)
        e_clusters.remove(closest_e_cluster)
    
    #Define the lines and compute their unit vectors
    XA0 = np.array(vecs[0][0])  
    XA1 = np.array(vecs[0][1])  
    XB0 = np.array(vecs[1][0])  
    XB1 = np.array(vecs[1][1])  
    UA = (XA1 - XA0) / np.linalg.norm(XA1 - XA0)  
    UB = (XB1 - XB0) / np.linalg.norm(XB1 - XB0)  

    #Find the unit vector for line C which is perpendicular to both A and B
    UC = np.cross(UB, UA)
    UC /= np.linalg.norm(UC)

    #Solve the system of equations
    RHS = XB0 - XA0
    LHS = np.array([UA, -UB, UC]).T
    try:
        #Solve for the parameters t1, t2, t3
        vals = np.linalg.solve(LHS, RHS)
    except np.linalg.LinAlgError:
        #In case LHS is singular (e.g., lines are parallel), use least squares
        vals = np.linalg.lstsq(LHS, RHS, rcond=None)[0]

    #The intersection point (approximate by using the midpoint between the closest points on the two lines)
    P_A = XA0 + vals[0] * UA  #Closest point on line A
    P_B = XB0 + vals[1] * UB  #Closest point on line B
    vertex = (P_A + P_B) / 2  #Midpoint of closest approach
        
    return (vecs, vertex)
                

def cluster_energy(clusters_mod):
    e_list = [] 
    for (key, cluster) in clusters_mod.items():
        temp_list = [] 
        #ignore the noise clusters for now 
        if (isinstance(key, str) and key.startswith('-1')) or key == -1: 
            None
        #particle clusters
        else: 
            for hit in cluster: 
                #particle tag
                core = np.mean([p[:3] for p in cluster], axis=0)
                if core[2] < 36100:
                    particle_tag = 'g'
                else: 
                    particle_tag = 'n'

                #detector tag
                total_energy = sum(hit[3] for hit in cluster)
                if hit[4] == "e":
                    detector_tag = "e"
                else:
                    detector_tag = "h"
                    if particle_tag == "g":
                        total_energy = total_energy / 0.02
                    else: 
                        total_energy = total_energy / 0.017
            temp_list = [key, particle_tag, detector_tag, total_energy, core]

        e_list.append(temp_list)
        e_list = [x for x in e_list if x]

    return e_list
            
        
def photon_energy(e_list):
    g_hcal = []
    g_ecal = [] 
    new_e_list = [] 
    
    # Separate 'g' clusters with 'e' and 'h' detector_tag and remove them from e_list
    for cluster in e_list:
        if (cluster[1] == "g") and (cluster[2] == "e"):
            g_ecal.append(cluster)
        elif (cluster[1] == "g") and (cluster[2] == "h"):
            g_hcal.append(cluster)
            cluster[2] = "e+h"
        else: 
            new_e_list.append(cluster)
    
    # Merge the g_hcal and g_ecal lists based on closest x coordinates
    if len(g_hcal) == 2 and len(g_ecal) == 2:
        x_hcal = [cluster[4][0] for cluster in g_hcal]
        x_ecal = [cluster[4][0] for cluster in g_ecal]
        
        # Calculate differences and find the closest pairs
        diffs = np.abs(np.array(x_hcal).reshape(-1, 1) - np.array(x_ecal).reshape(1, -1))
        pairs = []
        used_hcal = set()
        used_ecal = set()
        
        for _ in range(len(g_hcal)):
            min_diff = np.min(diffs)
            min_idx = np.unravel_index(np.argmin(diffs), diffs.shape)
            hcal_idx, ecal_idx = min_idx
            if hcal_idx not in used_hcal and ecal_idx not in used_ecal:
                pairs.append((g_hcal[hcal_idx], g_ecal[ecal_idx]))
                used_hcal.add(hcal_idx)
                used_ecal.add(ecal_idx)
            diffs[hcal_idx, :] = np.inf
            diffs[:, ecal_idx] = np.inf
        
        merged_clusters = []
        for hcal, ecal in pairs:
            # Merge logic here; for example, combine energy
            merged_cluster = [
                hcal[0],  # key (from hcal)
                hcal[1],  # particle_tag (from hcal)
                hcal[2],  # detector_tag "e+h"
                hcal[3] + ecal[3],  # combined energy
                hcal[4]  # position (from hcal)
            ]
            merged_clusters.append(merged_cluster)
        
        return new_e_list + merged_clusters
    
    return new_e_list


if __name__ == "__main__":

    data = np.load('extracted_data.npz', allow_pickle=True)
    event_data_hits = data['event_data_hits']
    event_data_particles = data['event_data_particles']
    good_event_list = data['good_event_list']

    #print(good_event_list[0],"\n******\n", event_data_hits[0], "\n******\n", event_data_particles[0])
    #print(good_event_list[0:100])

    #Prep data
    event_id = 13
    event = np.array([(hit[0], hit[1], hit[2], hit[3]) for hit in event_data_hits[event_id]])
    #print(good_event_list[:10])
    #print("Event number:", good_event_list[event_id])
    event = rotate(event)
    event_list = [list(hit) for hit in event]
    for idx, hit in enumerate(event_list):
        hit.append(event_data_hits[event_id][idx][4])
    event = [tuple(hit) for hit in event_list]

    event_filt = [] 

    #Make noise cuts (different for HCal and ECal hits)
    for hit in event: 
        if hit[4] == "h":
            if hit[3] >= 0.001:
                event_filt.append(hit)
        else: 
            if hit[3] >= 0.005: 
                event_filt.append(hit)

    #Scale up HCal energy (sampling fraction is different in ECal and HCal)
    scaled_event_filt = []
    for hit in event_filt:
        hit = list(hit)
        if hit[4] == 'h':
            hit[3] = hit[3] * (200)
        scaled_event_filt.append(tuple(hit))

    #CLUSTERING PREP
    #Check photon vertex, closest photon distance, and closest photon momentum for ideal clustering candidates
    """
    event_data_filt = [] 
    good_event_list_filt = [] 

    for index, event in enumerate(event_data_particles):
        neutron = event[1]
        g1 = event[2]
        g2 = event[3]
        
        # Make sure the photons have the same vertex
        if g1[2] == g2[2]:
            pos = np.array(neutron[3])
            
            # Check if both photons land in the ZDC
            g1_condition = (-1200 < g1[3][0] < -590) and (-300 < g1[3][1] < 300) and (35500 < g1[3][2] < 37500)
            g2_condition = (-1200 < g2[3][0] < -590) and (-300 < g2[3][1] < 300) and (35500 < g2[3][2] < 37500)
            
            if g1_condition and g2_condition:
                d1 = euclidean_dist_3D(neutron[3], g1[3])
                d2 = euclidean_dist_3D(neutron[3], g2[3])
                
                if d1 < d2:
                    z_diff = pos[-1] - g1[3][2]
                    gmom = np.linalg.norm(g1[1])
                else:
                    z_diff = pos[-1] - g2[3][2]
                    gmom = np.linalg.norm(g2[1])
                
                # Append the calculated z_diff and gmom to the event
                event_data_particles[index].append((z_diff, gmom))
            else:
                event_data_particles[index].append("ignore")

    for index in range(min(50, len(event_data_particles))):
        event = event_data_particles[index]
        print(event[-1])
    print(good_event_list[:50])
    """

    #Clustering
    #Parameters for Steps 1 and 2 
    epsilon_s1 = 60
    min_pts_s1 = 3
    epsilon_s2 = 50
    min_pts_s2 = 4

    #Step 1: 
    dbscan_1 = DBSCAN_S1(euclidean_dist_3D, epsilon_s1, min_pts_s1, energy_threshold=0.01)
    clusters_1 = dbscan_1.cluster(event_filt)
    print("\n*******","\nInitial number of clusters:", len(clusters_1))
    clusters_1 = step_1_check(clusters_1)
    print("Number of clusters after post-processing:", len(clusters_1))
    
    #Step 2: 
    if len(clusters_1) < 5:
        key_largest_clust = max(clusters_1, key=lambda k: len(clusters_1[k]))
        clust_test = clusters_1.pop(key_largest_clust)
        dbscan_2 = DBSCAN_S2(euclidean_dist_3D, epsilon_s2, min_pts_s2, distance_threshold_mm=100)
        clusters_2 = dbscan_2.cluster(clust_test)
        #Merge the clusters if there are too many produced after the second step 
        if len(clusters_2) > 5: 
            print("Clustering Step: 2", "\n*******")
            clusters_2_merged = merge_clusters(clusters_2, distance_threshold=25)
        #Merge the results from steps 1 and 2
        clusters_mod = merge_dicts_unique_keys(clusters_1, clusters_2_merged)
    else: 
        clusters_mod = clusters_1
        print("Clustering Step: 1", "\n*******")

    #print(clusters_mod)
    #print("\n*******", "\n",clusters_mod)
    #Step 3: 


    """
    e_list = cluster_energy(clusters_mod)
    print(e_list, "\n********\n")
    result = photon_energy(e_list)
    print(result)
    """

    #Reconstruction of Neutron and Photon vectors
    cluster_points_n, cluster_points_g = find_points_odr(clusters_mod)
    pn = cluster_points_n[0][3:][0]
    pg1 = cluster_points_g[0][3:][0]
    pg2 = cluster_points_g[1][3:][0]


    #Rotate true position vectors
    gd1 = list(event_data_particles[event_id][2])
    gd1 = [gd1[2], gd1[3]]
    gd1 = rotate(gd1)
    gd2 = list(event_data_particles[event_id][3])
    gd2 = [gd2[2], gd2[3]]
    gd2 = rotate(gd2)
    nd = list(event_data_particles[event_id][1])
    nd = [nd[2], nd[3]]
    nd = rotate(nd)


    #Neutron reconstruction plot 
    fig1 = plt.figure(figsize=(16, 8))
    ax0 = fig1.add_subplot(121, projection='3d')
    sc = ax0.scatter([pn[0][0], pn[1][0]], [pn[0][1], pn[1][1]], [pn[0][2], pn[1][2]], color="black", alpha=1, s=10, label="Reconstructed hits")
    ax0.plot([nd[0][0], nd[1][0]], [nd[0][1], nd[1][1]], [nd[0][2], nd[1][2]], color="green", alpha=1, label="True Vec n")
    ax0.set_title("ZDC Neutron Reconstruction")
    ax0.set_xlabel("X [mm]")
    ax0.set_ylabel("Y [mm]")
    ax0.set_zlabel("Z [mm]")
    ax0.legend()

    #Photon reconstruction plot 
    ax1 = fig1.add_subplot(122, projection='3d')
    sc = ax1.scatter([pg1[0][0], pg1[1][0]], [pg1[0][1], pg1[1][1]], [pg1[0][2], pg1[1][2]], color="blue", alpha=1, s=10, label="Reconstructed hits g2")
    sc = ax1.scatter([pg2[0][0], pg2[1][0]], [pg2[0][1], pg2[1][1]], [pg2[0][2], pg2[1][2]], color="red", alpha=1, s=10, label="Reconstructed hits g1")
    ax1.plot([gd1[0][0], gd1[1][0]], [gd1[0][1], gd1[1][1]], [gd1[0][2], gd1[1][2]], color="red", alpha=1, linestyle='-', label="True Vec g1")
    ax1.plot([gd2[0][0], gd2[1][0]], [gd2[0][1], gd2[1][1]], [gd2[0][2], gd2[1][2]], color="blue", alpha=1, linestyle='-', label="True Vec g2")
    ax1.set_title("ZDC Photon Reconstruction")
    ax1.set_xlabel("X [mm]")
    ax1.set_ylabel("Y [mm]")
    ax1.set_zlabel("Z [mm]")
    ax1.legend()
    


    #Create Clustering figure and define color list
    colors = ['blue', 'red', 'green', 'orange', 'pink', 'magenta', 'black', 'cyan', 'violet']
    fig2 = plt.figure(figsize=(16, 8))

    #Raw hit data plot
    ax2 = fig2.add_subplot(121, projection='3d')
    x_raw = [hit[0] for hit in event_filt]
    y_raw = [hit[1] for hit in event_filt]
    z_raw = [hit[2] for hit in event_filt]
    weights_raw = [hit[3] for hit in event_filt]

    #Scale sizes based on energy values
    energy_min, energy_max = np.min(weights_raw), np.max(weights_raw)
    sizes = 40 * (weights_raw - energy_min) / (energy_max - energy_min + 1e-6)  # Adjust scaling factor as needed

    sc = ax2.scatter(x_raw, y_raw, z_raw, c=weights_raw, cmap='viridis', s=sizes, alpha = 0.5)
    ax2.set_title("ZDC 3D Hitmap HCal (Raw)")
    ax2.set_xlabel("X [mm]")
    ax2.set_ylabel("Y [mm]")
    ax2.set_zlabel("Z [mm]")
    plt.colorbar(sc, ax=ax2, label="Energy")

    #Adjust view for raw hit data plot
    ax2.view_init(elev=15, azim=20)
    x_min, x_max = np.min(x_raw), np.max(x_raw)
    y_min, y_max = np.min(y_raw), np.max(y_raw)
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    ax2.set_zlim([35600, 37500])

    #Modified Clustered data plot
    ax3 = fig2.add_subplot(122, projection='3d')
    for i, (key, event) in enumerate(clusters_mod.items()):
        x_clust = [hit[0] for hit in event]
        y_clust = [hit[1] for hit in event]
        z_clust = [hit[2] for hit in event]
        #Set color to grey if key begins with -1
        if isinstance(key, str) and key.startswith('-1'):
            color = 'white'
        elif key == -1:
            color = 'grey'
        else:
            color = colors[i % len(colors)]
            #color = 'grey' if key == -1 else colors[i % len(colors)]
        ax3.scatter(x_clust, y_clust, z_clust, c=color, label=f'Cluster {key}', s=10, alpha = 0.5)

    #Adjust view and set axis scales
    ax3.view_init(elev=15, azim=20)
    ax3.set_xlim([x_min, x_max])
    ax3.set_ylim([y_min, y_max])
    ax3.set_zlim([35600, 37500])
    ax3.legend()

    ax3.set_title("ZDC 3D Hitmap HCal Clusters (Mod)")
    ax3.set_xlabel("X [mm]")
    ax3.set_ylabel("Y [mm]")
    ax3.set_zlabel("Z [mm]")

    #plt.savefig("histogram_3d_test.png")
    #print("Histogram saved")

    plt.show()

    
    #Keep running script until user closes it 
    input("Press Enter to exit")