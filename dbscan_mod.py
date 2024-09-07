import numpy as np # type: ignore

#distinct cluster means no overlap between two clusters 
#First step clustering attempt: All three particles leave distinct high energy clusters that are far apart 
#ex. 36
class DBSCAN_S1:
    UNDEFINED = -2
    NOISE = -1

    class __ClusterPoint:
        def __init__(self, point, cluster):
            self.point = point
            self.cluster = cluster

    def __init__(self, dist_func, epsilon, min_neighbors, energy_threshold):
        self.dist_func = dist_func
        self.epsilon = epsilon
        self.min_neighbors = min_neighbors
        self.energy_threshold = energy_threshold

    def cluster(self, points):
        clustered = self.__dbscan(points)
        splitted = self.__split_by_cluster(clustered)
        return splitted

    def __dbscan(self, points):
        point_class_list = [self.__ClusterPoint(x, self.UNDEFINED) for x in points]
        current_cluster = 0

        for current_point in point_class_list:
            if current_point.cluster != self.UNDEFINED:
                continue

            # Check the new noise condition
            if current_point.point[2] > 36500 and current_point.point[3] < self.energy_threshold:
                current_point.cluster = self.NOISE
                continue

            neighbors = self.__range_query(point_class_list, current_point)

            if len(neighbors) < self.min_neighbors or not self.energy_check(neighbors):
                current_point.cluster = self.NOISE
                continue

            current_point.cluster = current_cluster

            if current_point in neighbors:
                neighbors.remove(current_point)

            seeds = neighbors

            while seeds:
                q_point = seeds.pop(0)

                if q_point.cluster == self.NOISE:
                    q_point.cluster = current_cluster
                elif q_point.cluster != self.UNDEFINED:
                    continue

                q_point.cluster = current_cluster
                q_neighbors = self.__range_query(point_class_list, q_point)

                if len(q_neighbors) >= self.min_neighbors:
                    seeds.extend(q_neighbors)

            current_cluster += 1

        return point_class_list

    def __range_query(self, db, current_point):
        def is_point_in_range(other_point): 
            return self.dist_func(current_point.point, other_point.point) <= self.epsilon

        return list(filter(is_point_in_range, db))

    def __split_by_cluster(self, clustered):
        splitted = {}
        for p in clustered:
            current_cluster = p.cluster
            if current_cluster not in splitted:
                splitted[current_cluster] = []
            splitted[current_cluster].append(p.point)

        return splitted

    def energy_check(self, neighbors):
        # Check if any neighboring point has energy above the threshold
        for neighbor in neighbors:
            if neighbor.point[3] > self.energy_threshold:
                return True
        return False
    
#Second step clustering attempt: All three particles leave distinct high energy clusters that are closer together 
#ex. 14
class DBSCAN_S2:
    UNDEFINED = -2
    NOISE = -1

    class __ClusterPoint:
        def __init__(self, point, cluster):
            self.point = point
            self.cluster = cluster

    def __init__(self, dist_func, epsilon, min_neighbors, distance_threshold_mm=100):
        self.dist_func = dist_func
        self.epsilon = epsilon
        self.min_neighbors = min_neighbors
        self.distance_threshold = distance_threshold_mm / 1000  # Convert mm to meters

    def cluster(self, points):
        clustered = self.__dbscan(points)
        splitted = self.__split_by_cluster(clustered)
        processed_clusters = self.__process_clusters(splitted)
        return processed_clusters

    def __dbscan(self, points):
        point_class_list = [self.__ClusterPoint(x, self.UNDEFINED) for x in points]
        current_cluster = 0

        for current_point in point_class_list:
            if current_point.cluster != self.UNDEFINED:
                continue

            neighbors = self.__range_query(point_class_list, current_point)

            if len(neighbors) < self.min_neighbors:
                current_point.cluster = self.NOISE
                continue

            current_point.cluster = current_cluster

            if current_point in neighbors:
                neighbors.remove(current_point)

            seeds = neighbors

            while seeds:
                q_point = seeds.pop(0)

                if q_point.cluster == self.NOISE:
                    q_point.cluster = current_cluster
                elif q_point.cluster != self.UNDEFINED:
                    continue

                q_point.cluster = current_cluster
                q_neighbors = self.__range_query(point_class_list, q_point)

                if len(q_neighbors) >= self.min_neighbors:
                    seeds.extend(q_neighbors)

            current_cluster += 1

        return point_class_list

    def __range_query(self, db, current_point):
        def is_point_in_range(other_point): 
            return self.dist_func(current_point.point, other_point.point) <= self.epsilon

        return list(filter(is_point_in_range, db))

    def __split_by_cluster(self, clustered):
        splitted = {}
        for p in clustered:
            current_cluster = p.cluster
            if current_cluster not in splitted:
                splitted[current_cluster] = []
            splitted[current_cluster].append(p.point)

        return splitted

    def __process_clusters(self, clusters):
        new_clusters = {}
        current_cluster_id = 0

        for cluster_id, points in clusters.items():
            if cluster_id == self.NOISE:
                new_clusters[cluster_id] = points
                continue

            # Compute energy density for each cluster
            energy_density = self.compute_energy_density(points)

            # Check if the distance between points exceeds the threshold
            if self.check_distance_exceeds_threshold(points):
                split_clusters = self.__check_energy_density(points, energy_density)
                for split_cluster in split_clusters:
                    new_clusters[current_cluster_id] = split_cluster
                    current_cluster_id += 1
            else:
                new_clusters[cluster_id] = points

        return new_clusters

    def __check_energy_density(self, points, original_energy_density):
        # Sort points based on distance from the centroid
        centroid = np.mean([p[:3] for p in points], axis=0)
        points_sorted = sorted(points, key=lambda p: self.dist_func(p[:3], centroid))

        current_cluster = []
        split_clusters = []
        previous_energy_density = original_energy_density

        for point in points_sorted:
            current_cluster.append(point)
            current_energy_density = self.compute_energy_density(current_cluster)

            if current_energy_density > previous_energy_density:
                if len(split_clusters) < 2 and self.check_distance_exceeds_threshold(current_cluster):
                    # Save the current cluster and start a new one
                    split_clusters.append(current_cluster[:-1])
                    current_cluster = [current_cluster[-1]]

            previous_energy_density = current_energy_density

        if current_cluster:
            split_clusters.append(current_cluster)

        # Cap the number of splits to a maximum of 5 clusters
        if len(split_clusters) > 3:
            split_clusters = split_clusters[:3]

        return split_clusters

    def check_distance_exceeds_threshold(self, points):
        max_distance = 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                # Calculate 2D distance using only x and y coordinates
                distance = np.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                if distance > max_distance:
                    max_distance = distance
        return max_distance > self.distance_threshold

    def compute_energy_density(self, points):
        if not points:
            return 0
        total_energy = sum(p[3] for p in points)
        volume = (self.epsilon * len(points)) ** 3  # Adjust volume calculation as needed
        return total_energy / volume