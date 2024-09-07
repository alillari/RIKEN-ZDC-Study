# RIKEN-ZDC-Study

Bing Bong!

DBSCAN Clustering background: 
The algorithm currently works only for one event at a time. Next update will fix this as well as how rotate function works. 

DBSCAN Clustering Instructions: 
1. Need analysis, and dbscan_mod files to use the algorithm
2. Only need to make changes in the main() function (might need to modify rotate() funtion as well - up to you). All lines in main() upto "Scale up HCal Energy" comment can be commented out - they are just there to get data from my file and organize it correctly. 
4. All you need to do is create a list called "event" that contains tuples with the following hit info, (x, y, z, energy, detector_tag). For example "[(179.97497898503605, 54.213191986083984, 36370.049736540124, 6.828349432907999e-05, 'h'), (133.02504894552294, 81.31978607177734, 36370.05148807682, 1.615391556697432e-05, 'h'), ..., (287.8499636913565, -287.8500061035156, 35736.99907480968, 0.0012800407130271196, 'e'), (227.24995291926393, 15.149999618530273, 35736.99829202678, 0.00030324480030685663, 'e')]"

