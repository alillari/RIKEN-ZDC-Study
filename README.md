# RIKEN-ZDC-Study

DDSIM STEERING:
### ddsim --compactFile $DETECTOR_PATH/$DETECTOR_CONFIG.xml -G -N 1000 --gun.thetaMin "-0.034" --gun.thetaMax "0.034" --gun.phiMin "-0.018" --gun.phiMax "-0.036" --gun.distribution "uniform" --gun.momentumMin "1*GeV" --gun.momentumMax "50*GeV" --gun.particle "gamma" --outputFile gammatest.edm4hep.root

DBSCAN Clustering background: 
The algorithm currently works only for one event at a time. Next update will fix this as well as how rotate function works. 

DBSCAN Clustering Instructions: 
1. Need analysis, and dbscan_mod files to use the algorithm
2. Only need to make changes in the main() function (might need to modify rotate() funtion as well - up to you). All lines in main() upto "Scale up HCal Energy" comment can be commented out - they are just there to get data from my file and organize it correctly. 
4. All you need to do is create a list called "event" that contains tuples with the following hit info, (x, y, z, energy, detector_tag). For example "[(179.97497898503605, 54.213191986083984, 36370.049736540124, 6.828349432907999e-05, 'h'), (133.02504894552294, 81.31978607177734, 36370.05148807682, 1.615391556697432e-05, 'h'), ..., (287.8499636913565, -287.8500061035156, 35736.99907480968, 0.0012800407130271196, 'e'), (227.24995291926393, 15.149999618530273, 35736.99829202678, 0.00030324480030685663, 'e')]". The detector tag just needs to be "e" or "h" depending on whether the hit was in the ecal or hcal for energy scaling.
5. I would also reccomend rotating the data before clustering. There is a function already in the analysis file but it needs to be modified to use it since it currently is looking for a list of lists that each contain "[x, y, z, energy]" as the input not a list of tuples. That being said clustering will work just fine even if you dont rotate the data.
6. From here just run the analysis file! 

Photon steering file

import math

from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import cm, mm, GeV, MeV, radian, m
SIM = DD4hepSimulation()

SIM.numberOfEvents = 100000
SIM.enableGun = True
SIM.outputFile = "photon_fullcoverage_1to50GeV.edm4hep.root"

SIM.gun.particle = "gamma"
SIM.gun.momentumMin = 1*GeV
SIM.gun.momentumMax = 50*GeV

SIM.gun.phiMin = .025
SIM.gun.phiMax = .03
SIM.gun.thetaMin = -.034
SIM.gun.thetaMax = .034
SIM.gun.distribution = "uniform"
