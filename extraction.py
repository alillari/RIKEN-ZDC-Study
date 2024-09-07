import ROOT # type: ignore
import numpy as np # type: ignore
from collections import Counter # type: ignore

#Define the ECal plane and HCal plane
#ECal
ZDC_norm = np.array([np.sin(-0.025), 0., np.cos(0.025)])
ecal_pt = np.array([35500*np.sin(-0.025), 0., 35500*np.cos(0.025)])
#HCal
hcal_pt0 = np.array([36607.5*np.sin(-0.025), 0., 36607.5*np.cos(0.025)])
hcal_pts = [hcal_pt0] 
for i in range(1, 64):
    lay_thick = 24.9 
    r0 = 36607.5 
    lay_loc = ZDC_norm * (lay_thick*i + r0)
    hcal_pts.append(lay_loc)

#Function that determines the intersection between the momentum vector and Ecal plane
def intersection(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = np.dot(planeNormal,rayDirection)
    if abs(ndotu) > epsilon:
        w = rayPoint - planePoint
        t = -np.dot(planeNormal,w) / ndotu
        intersectionPoint = rayPoint + t * rayDirection
        return intersectionPoint
    return None 

#Isolate ideal events (both photons and neutron hit ZDC)
#PID: photon 22, pi0 111, neutron 2112, lambda 3122
def interesting_event(lst):
    required_pids = {22: 2, 111: 1, 2112: 1, 3122: 1}
    counts = Counter(lst)
    for number, required_count in required_pids.items():
        if counts[number] < required_count:
            return False
    return True


#Get hit data from root files
#Ouput is a list of events. In each event is another list with the x,y,z hit position and energy 
def extract_hits(chain): 
    e_num = 0 #Counter for event tracking
    event_data_hits = [] 
    event_data_particles = []
    good_evt_list = []

    for event in chain:
        MCParticles_vec = event.MCParticles
        hcal_hits = event.HcalFarForwardZDCHits
        ecal_hits = event.EcalFarForwardZDCHits
        pid_list = []
        temp_list = []  
        hit_list = []

        for i in range(MCParticles_vec.size()):
            particle = MCParticles_vec.at(i)
            pid = particle.PDG
            pid_list.append(pid)
            mom_vec = [particle.momentum.x, particle.momentum.y, particle.momentum.z]
            mag_mom_vec = np.linalg.norm(mom_vec)
            vertex_vec = [particle.vertex.x, particle.vertex.y, particle.vertex.z]
            end_vec = [particle.endpoint.x, particle.endpoint.y, particle.endpoint.z]
            if (pid == 22 or pid == 2112 or pid == 3122): #ignore pi0 information
                temp_list.append((particle.PDG, mom_vec/mag_mom_vec, vertex_vec, end_vec, mom_vec))

            #CUT 1.0: Double photon, neutron events only
            check = interesting_event(pid_list)
            if((check == True) and (len(pid_list) == 5)): 
                #Both photons and neutron hit the ECal plane
                inter_ptn = intersection(ZDC_norm, ecal_pt, temp_list[1][1], temp_list[1][2])
                inter_ptg1 = intersection(ZDC_norm, ecal_pt, temp_list[2][1], temp_list[2][2])
                inter_ptg2 = intersection(ZDC_norm, ecal_pt, temp_list[3][1], temp_list[3][2])
                inter_list = [inter_ptn, inter_ptg1, inter_ptg2]
                for pt in inter_list:
                    if (-1200 < pt[0] < -590) and (-300 < pt[1] < 300):
                        save_data = True
                    else: 
                        save_data = False
                if save_data == True: 
                    #Replace the normalized momentum vector with the actual momentum vector of the particle
                    for idx, particle in enumerate(temp_list):
                        particle = list(particle)  
                        particle[1] = particle[-1]  
                        particle.pop()  
                        temp_list[idx] = tuple(particle)  
                    for hit in hcal_hits:
                            hit_list.append((hit.position.x, hit.position.y, hit.position.z, hit.energy, "h"))
                    for hit in ecal_hits:
                            hit_list.append((hit.position.x, hit.position.y, hit.position.z, hit.energy, "e"))
                    event_data_hits.append(hit_list)
                    event_data_particles.append(temp_list)
                    good_evt_list.append(e_num)
                #event_data.append([e_num, pid_lst])
            else: 
                None

        e_num += 1   
    return (event_data_hits, event_data_particles, good_evt_list)

#Create a TChain and add ROOT files
chain = ROOT.TChain("events")
chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r100.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r101.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r200.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r201.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r202.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r203.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r204.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r206.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r207.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r208.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r209.root")
#chain.Add("/Users/gursimran/eic/Lambda/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r210.root")

event_data_hits, event_data_particles, good_event_list = extract_hits(chain)

#Convert the lists to numpy arrays
event_data_hits = np.array(event_data_hits, dtype=object)
event_data_particles = np.array(event_data_particles, dtype=object)
good_event_list = np.array(good_event_list)

#Save the arrays to a .npz file
np.savez('extracted_data.npz', event_data_hits=event_data_hits, event_data_particles=event_data_particles, good_event_list=good_event_list)
print("Extraction complete. Data saved in extracted_data.npz")