import numpy as np  # type: ignore
import ROOT  # type: ignore

# Load the data
data = np.load('singlephotondata.npz', allow_pickle=True)
event_data_hcal = data['event_data_hcal']

# Set up the canvas and histograms
canvas4 = ROOT.TCanvas("canvas4", "HCal X Hits", 800, 600)
canvas4.Divide(5, 2)  # Divide the canvas into 2x5 grid
canvas5 = ROOT.TCanvas("canvas5", "HCal XY Hits", 800, 600)
canvas5.Divide(5, 2)  # Divide the canvas into 2x5 grid

histograms4 = []
for i in range(1, 11):
    hist = ROOT.TH1F(f"PhotonHits{i}", f"Number of Photon Hits HCal_L{i}; Hits", 20, 0, 50)
    histograms4.append(hist)

histograms5 = []
for i in range(1, 11):
    hist = ROOT.TH2F(f"XYHits{i}", f"Number of Photon Hits HCal_L{i}; X [mm]; Y [mm]", 20, -1100, -600, 20, -300, 300)
    histograms5.append(hist)

# Define HCAL layers positions
hcalpts = [35800]
for i in range(1, 64):
    lay_thick = (37400 - 35800) / 64
    lay_loc = 35800 + lay_thick * i
    hcalpts.append(lay_loc)

# Count hits for each event in each layer
counts = []
xhits = [] 
yhits = [] 
for event in event_data_hcal:
    temp_list = [0] * 10  # Initialize hit counts for 10 layers
    temp_listx = [0] * 10
    temp_listy = [0] * 10
    for hit in event:
        z = -hit[0] * np.sin(0.025) + hit[2] * np.cos(0.025)
        if z < hcalpts[1]:
            temp_list[0] += 1
            temp_listx[0] = hit[0]
            temp_listy[0] = hit[1]
        elif hcalpts[1] < z < hcalpts[2]:
            temp_list[1] += 1
            temp_listx[1] = hit[0]
            temp_listy[1] = hit[1]
        elif hcalpts[2] < z < hcalpts[3]:
            temp_list[2] += 1
            temp_listx[2] = hit[0]
            temp_listy[2] = hit[1]
        elif hcalpts[3] < z < hcalpts[4]:
            temp_list[3] += 1
            temp_listx[3] = hit[0]
            temp_listy[3] = hit[1]
        elif hcalpts[4] < z < hcalpts[5]:
            temp_list[4] += 1
            temp_listx[4] = hit[0]
            temp_listy[4] = hit[1]
        elif hcalpts[5] < z < hcalpts[6]:
            temp_list[5] += 1
            temp_listx[5] = hit[0]
            temp_listy[5] = hit[1]
        elif hcalpts[6] < z < hcalpts[7]:
            temp_list[6] += 1
            temp_listx[6] = hit[0]
            temp_listy[6] = hit[1]
        elif hcalpts[7] < z < hcalpts[8]:
            temp_list[7] += 1
            temp_listx[7] = hit[0]
            temp_listy[7] = hit[1]
        elif hcalpts[8] < z < hcalpts[9]:
            temp_list[8] += 1
            temp_listx[8] = hit[0]
            temp_listy[8] = hit[1]
        elif hcalpts[9] < z < hcalpts[10]:
            temp_list[9] += 1
            temp_listx[9] = hit[0]
            temp_listy[9] = hit[1]
    
    counts.append(temp_list)
    xhits.append(temp_listx)
    yhits.append(temp_listy)

# Fill the histograms with the counts
for count in counts:
    for i in range(10):
        histograms4[i].Fill(count[i])

for i, (xhit, yhit, count) in enumerate(zip(xhits, yhits, counts)):
    for j in range(10):
        histograms5[j].Fill(xhit[j], yhit[j], count[j])

# Draw the histograms and format the canvas
for i, hist in enumerate(histograms4):
    canvas4.cd(i + 1)
    hist.Draw()
    hist.GetXaxis().SetLabelSize(0.04)
    hist.GetXaxis().SetTitleSize(0.04)

# Draw the 2D histograms
for i, hist in enumerate(histograms5):
    canvas5.cd(i + 1)
    hist.Draw("COLZ")
    hist.GetXaxis().SetLabelSize(0.04)
    hist.GetXaxis().SetTitleSize(0.04)

# Update the canvas and keep it open
canvas4.Update()
canvas5.Update()
input("Press Enter to exit")