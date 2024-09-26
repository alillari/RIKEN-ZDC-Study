import ROOT  # type: ignore
import numpy as np  # type: ignore

# Load the data
data = np.load('singlephotondata.npz', allow_pickle=True)
event_data_hcal = data['event_data_hcal']
#event_data_ecal = data['event_data_ecal']
event_data_particles = data['event_data_particles']
#print(event_data_hcal[0])

e_num = 2

#Define HCAL layers positions
hcalpts = [35800]
for i in range(1, 65):  
    lay_thick = (37400 - 35800) / 64
    lay_loc = 35800 + lay_thick * i
    hcalpts.append(lay_loc)

#Separate the data into HCAL layers for a single event 
event_layers = [[] for _ in range(64)]  #Initialize a list of 64 empty lists
for hit in event_data_hcal[e_num]: 
    hit = list(hit)
    z = -hit[0] * np.sin(0.025) + hit[2] * np.cos(0.025)
    for i in range(len(hcalpts) - 1):
        if hcalpts[i] < z <= hcalpts[i + 1]:
            event_layers[i].append(hit)
            break

#Calculate normalized energy-weighted average x and y positions for each layer
def average_hit(event_layers):
    energy_weighted_averages = []
    for layer_hits in event_layers:
        total_energy = 0
        weighted_x_sum = 0
        weighted_y_sum = 0
        
        #Calculate total energy for the layer
        for hit in layer_hits:
            total_energy += hit[3]
        
        #Calculate normalized weighted sums for x and y
        if total_energy > 0:  # Avoid division by zero
            for hit in layer_hits:
                normalized_energy = hit[3] / total_energy  # Normalize the energy
                weighted_x_sum += hit[0] * normalized_energy
                weighted_y_sum += hit[1] * normalized_energy

            #Calculate energy-weighted averages
            avg_x = weighted_x_sum
            avg_y = weighted_y_sum
        else:
            avg_x = 0  # Handle layers with zero total energy
            avg_y = 0  # Handle layers with zero total energy
        
        energy_weighted_averages.append((avg_x, avg_y))
    return energy_weighted_averages


def average_error(event_layers, average_layers):
    error_layers = []
    
    for i in range(len(event_layers)):  #Loop through each layer
        count = len(event_layers[i])
        
        if count == 0:
            error_layers.append((0, 0))  #Skip empty layers
            continue
        
        sum_wi = 0
        sum_wi2 = 0
        sum_wixi2_x = 0
        sum_wixi2_y = 0
        total_energy = 0
        
        #Calculate total energy for the layer
        for hit in event_layers[i]:
            total_energy += hit[3]
        
        #Calculate weighted sums for x2 and y2
        for hit in event_layers[i]:
            weight = hit[3] / total_energy  #Normalized energy (weight)
            sum_wi += weight
            sum_wi2 += weight**2
            sum_wixi2_x += weight * hit[0]**2
            sum_wixi2_y += weight * hit[1]**2
        
        # Retrieve the pre-calculated average positions for this layer
        xav, yav = average_layers[i]
        
        # Calculate errors
        if sum_wi > 0 and sum_wi**2 != sum_wi2:  # Guard against division by zero
            # Error in x
            x1 = (sum_wixi2_x / sum_wi) - xav**2
            x2 = sum_wi2 / (sum_wi**2 - sum_wi2)
            errx = np.sqrt(x1 * x2)
            
            # Error in y
            y1 = (sum_wixi2_y / sum_wi) - yav**2
            y2 = sum_wi2 / (sum_wi**2 - sum_wi2)
            erry = np.sqrt(y1 * y2)
        else:
            errx = 0  # Handle division by zero case
            erry = 0  # Handle division by zero case
        
        error_layer = (errx, erry)
        error_layers.append(error_layer)

    return error_layers

average_layers = average_hit(event_layers)
error_layers = average_error(event_layers, average_layers)

#Output
#print("Error for layer 0:", error_layers[0])
#print("Average hit for layer 0:", average_layers[0])

#Plot the data on an XY scatterplot in ROOT with error bars
n_points = len(average_layers)
x = np.array([hit[0] for hit in average_layers], dtype=float)
y = np.array([hit[1] for hit in average_layers], dtype=float)
errx = np.array([error[0] for error in error_layers], dtype=float)
erry = np.array([error[1] for error in error_layers], dtype=float)

# Create the TGraphErrors object
graph = ROOT.TGraphErrors(n_points, x, y, errx, erry)

# Customize the graph
graph.SetTitle("Average HCal Hit Position; X [mm] ; Y [mm]")  # Set plot title and axis labels
graph.SetMarkerStyle(20)  # Set marker style (e.g., filled circle)
graph.SetMarkerColor(ROOT.kBlue)  # Set marker color
graph.SetLineColor(ROOT.kBlue)  # Set line color for the error bars
graph.GetXaxis().SetLimits(-1000, -800)
graph.GetYaxis().SetRangeUser(-120, 120)

# Create a canvas to draw the graph
c1 = ROOT.TCanvas("c1", "Average Hit Positions", 1000, 800)

# Draw the graph
graph.Draw("AP")  # "A" draws axis, "P" draws points with errors

# Display the canvas
c1.Update()
c1.Draw()
input("Press Enter to exit")
