#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TF1.h>
#include <utility>
#include <vector>
#include <TMatrixD.h>
#include <TVectorD.h>

std::vector<double> loc_in_ecal(const TTreeReaderArray<double>& ver_x,const TTreeReaderArray<double>& ver_y,const TTreeReaderArray<double>& ver_z,const TTreeReaderArray<float>& mom_x,const TTreeReaderArray<float>& mom_y,const TTreeReaderArray<float>& mom_z, int which_index) {
	double ecal_x = 35500*std::sin(-.025);
	double ecal_z = 35500*std::cos(.025);
	double ecal_y = 0;

	
	TVectorD vec(3);
	vec(0) = ver_x[which_index] - ecal_x;
	vec(1) = ver_y[which_index] - ecal_y;
	vec(2) = ver_z[which_index] - ecal_z;

	TMatrixD mat(3, 3);

	mat(0, 0) = 0;
	mat(1, 0) = 1;
	mat(2, 0) = 0;
	mat(0, 1) = cos(.025);
	mat(1, 1) = 0;
	mat(2, 1) = sin(.025);
	mat(0, 2) = -mom_x[which_index];
	mat(1, 2) = -mom_y[which_index];
	mat(2, 2) = -mom_z[which_index];

	TVectorD result = mat.Invert() * vec;

	std::vector<double> position = {ver_x[which_index] + result(2)*mom_x[which_index], ver_y[which_index] + result(2)*mom_y[which_index], ver_z[which_index] + result(2)*mom_z[which_index]};
	return position;
}

int find_particle_index(int particle_id, const TTreeReaderArray<int>& pdg, int which_instance, const TTreeReaderArray<int>& parent_id){
	int correct_parent;
	switch(particle_id){
		case 111:
			correct_parent = 3122;
			break;
		case 2112:
			correct_parent = 3122;
			break;
		case 22:
			correct_parent = 111;
			break;
	}
	for(int i = 1; i < pdg.GetSize(); ++i){
		if(pdg[i] == particle_id){
			if(pdg[parent_id[i-1]] == correct_parent){
				if(which_instance != 1){
					which_instance = which_instance - 1;
					continue;
				}
				return i;
			}

		}
	}
	//std::cout<<"Couldn't find that particle, "<< std::to_string(particle_id) <<" with instance, " << std::to_string(which_instance) <<" . Event "<< std::to_string(row_id[0])<<std::endl;
	return 0;
}

std::pair<std::vector<float>, std::vector<float>> rotate_arrays(TTreeReaderArray<float>& x_hits, TTreeReaderArray<float>& z_hits){
	std::vector<float> rot_x_hits;
	std::vector<float> rot_z_hits;
	float cosTheta = cos(.025);
	float sinTheta = sin(.025);

	for (size_t j = 0; j < x_hits.GetSize(); ++j) {
            float rot_x = x_hits[j] * cosTheta + z_hits[j] * sinTheta;
            float rot_z = - x_hits[j] * sinTheta + z_hits[j] * cosTheta;

            rot_x_hits.push_back(rot_x);
            rot_z_hits.push_back(rot_z);
	}

	std::pair<std::vector<float>, std::vector<float>> return_pair(rot_x_hits, rot_z_hits);

	return return_pair;

}



int main(){
	TChain* my_chain = new TChain("events");
	//my_chain->Add("../lambda_20to220GeV.edm4hep.root");
	//my_chain->Add("/home/alessio/RIKENSUMMER/data/Lambda_allGeV_ZDC_lyso_sipm.edm4hep_r100.root");
	//my_chain->Add("../photon_sampling_fraction.edm4hep.root");
	my_chain->Add("single_photons_only_5to70GeV.edm4hep.root");
	TTreeReader my_reader(my_chain);

	TFile* output = new TFile("photon_study.root", "RECREATE");

	TTreeReaderArray<float> ecal_ener(my_reader, "EcalFarForwardZDCHits.energy");
	TTreeReaderArray<float> ecal_hit_x(my_reader, "EcalFarForwardZDCHits.position.x");
	TTreeReaderArray<float> ecal_hit_y(my_reader, "EcalFarForwardZDCHits.position.y");
	TTreeReaderArray<float> ecal_hit_z(my_reader, "EcalFarForwardZDCHits.position.z");

	TTreeReaderArray<float> hcal_ener(my_reader, "HcalFarForwardZDCHits.energy");
	TTreeReaderArray<float> hcal_hit_x(my_reader, "HcalFarForwardZDCHits.position.x");
	TTreeReaderArray<float> hcal_hit_y(my_reader, "HcalFarForwardZDCHits.position.y");
	TTreeReaderArray<float> hcal_hit_z(my_reader, "HcalFarForwardZDCHits.position.z");

	TTreeReaderArray<int> hit_true_id(my_reader, "_EcalFarForwardZDCHitsContributions_particle.index");

	TTreeReaderArray<float> mom_x(my_reader, "MCParticles.momentum.x");
	TTreeReaderArray<float> mom_y(my_reader, "MCParticles.momentum.y");
	TTreeReaderArray<float> mom_z(my_reader, "MCParticles.momentum.z");

	//TTreeReaderArray<double> vert_x(my_reader, "MCParticles.vertex.x");
	//TTreeReaderArray<double> vert_y(my_reader, "MCParticles.vertex.y");
	//TTreeReaderArray<double> vert_z(my_reader, "MCParticles.vertex.z");

	//TTreeReaderArray<int> pdg(my_reader, "MCParticles.PDG");
	//TTreeReaderArray<int> parent_id(my_reader, "_MCParticles_parents.index");

	TTreeReaderArray<int> row_id(my_reader, "EventHeader.eventNumber");

	//TH2F* one_event_hit_map_ecal = new TH2F("one_event_hit_map_ecal", "Hit map for one event in Ecal, rotated frame; x axis [mm]; y axis [mm]", 100, -300, 300, 100, -300, 300);
	//TH3F* one_event_hit_map_hcal = new TH3F("one_event_hit_map_hcal","Hit map for one event in Hcal, rotated frame; x axis [mm]; y axis [mm]; z axis [mm]", 100, -300, 300, 100, -300, 300, 100, 35500, 37500);
	//TH3F* one_event_hit_map_hcal_no_weight = new TH3F("one_event_hit_map_hcal_no_weight", "Hit map for one event in Hcal, rotated frame and no weights; x axis [mm]; y axis [mm]; z axis [mm]", 100, -300, 300, 100, -300, 300, 100, 35500, 37500);

	//TH3F* one_event_hit_map_zdc = new TH3F("one_event_hit_map_zdc", "Hit mpa of one event for whole ZDC, rotated frame; x axis [mm]; y axis [mm]; z axis [mm]", 100, -300, 300, 100, -300, 300, 100, 35500, 37500);
	
	TH1F* photon_energy = new TH1F("photon_energy", "#gamma momentum; Momentum [GeV]; Count", 100, 1, 75);
	TH1F* photon_angle_dist = new TH1F("photon_angle_dist", "#gamma angle from 25mrad line; Angle [rad]; Count", 100, -10, 10);

	TH2F* photon_energy_hit_count_ecal = new TH2F("photon_energy_hit_count_ecal", "#gamma momentum vs. number of hits in ECal; #gamma momentum [GeV]; Number of hits in ECal", 100, 1, 75, 100, 0, 500);
	TH2F* photon_energy_hit_count_hcal = new TH2F("photon_energy_hit_count_hcal", "#gamma momentum vs. number of hits in HCal; #gamma momentum [GeV]; Number of hits in HCal", 100, 1, 75, 100, 0, 1000);
	
	TH2F* photon_energy_energy_in_hcal = new TH2F("photon_energy_energy_in_hcal", "#gamma momentum vs. energy of hits in HCal; #gamma momentum [GeV]; Energy of HCal hit [GeV]", 100, 1, 75, 100, 0, .5);
	TH2F* photon_energy_energy_in_ecal = new TH2F("photon_energy_energy_in_ecal", "#gamma momentum vs. energy of hits in ECal; #gamma momentum [GeV]; Energy of ECal hit [GeV]", 100, 1, 75, 100, 0, 15);

	TH1F* z_vertex_hcal_hit = new TH1F("z_vertex_hcal_hit", "Unrotated Z vertex of HCal Hits; unrotated Z vertex of HCal [mm]; Counts", 100, 36000, 37000);
	//TH2F* photon_momentum_z_vertex_ecal_hit = new TH1F("");

	TH1F* energy_in_ecal_low = new TH1F("energy_in_ecal_low", "Average energy of Ecal hit below .002", 100, 0, .002);
	TH1F* energy_in_hcal_low = new TH1F("energy_in_hcal_low", "Average energy of Hcal hit below .002", 100, 0, .002);
	TH1F* ecal_low_energy_counts = new TH1F("ecal_low_energy_counts", "Number of hits with deposited energy below .002", 100, 0, 500);
	TH1F* hcal_low_energy_counts = new TH1F("hcal_low_energy_counts", "Number of hits with deposited energy below .002", 100, 0, 500);
	TH1F* peak_energy_per_event_ecal = new TH1F("peak_energy_per_event_ecal", "Maximum energy deposited in single hit per event in Ecal; Energy [GeV]; Counts", 100, 0, 10);
	TH1F* peak_energy_per_event_hcal = new TH1F("peak_energy_per_event_hcal", "Maximum energy deposited in single hit per event in Hcal; Energy [GeV]; Counts", 100, 0, .3);
	TH1F* peak_energy_per_event_position_hcal = new TH1F("peak_energy_per_event_position_hcal", "Z Position of maximum hit energy in HCal; Position [mm]; Counts", 100, 35600, 36200);
	
	TH2F* xy_spread_in_HCal = new TH2F("xy_spread_in_HCal", "X and Y hit information in rotated frame; X axis [mm]; Y axis[mm]", 200, -300, 300, 200, -300, 300);

	TH1F* distance_in_the_hcal = new TH1F("distance_in_the_hcal", "Distribution of z-hits in Hcal for all events, all photon energies, all hit energies, 25mrad rotated frame; Z Hit position [mm]; Counts", 100, 35500, 37500);

	TH2F* distance_in_the_hcal_vs_hit_energy = new TH2F("distance_in_the_hcal_vs_hit_energy", "Distribution of z-hits vs. energy of those hits for all events and all photon energies, 25mrad rotated frame; Z Hit position [mm]; Energy deposited [GeV]", 100, 35500, 37500, 100, 0, .3);

	TH2F* energy_of_photon_vs_distance_in_the_hcal = new TH2F("energy_of_photon_vs_distance_in_the_hcal", "Energy of the photon vs. distribution of z-hits, 25mrad rotated frame; Energy of photon [GeV]; Z Hit position [mm]", 25, 0, 55, 100, 35500, 37500);



	while(my_reader.Next()){
		std::pair<std::vector<float>, std::vector<float>> ecal_x_z_hits = rotate_arrays(ecal_hit_x, ecal_hit_z);
		std::pair<std::vector<float>, std::vector<float>> hcal_x_z_hits = rotate_arrays(hcal_hit_x, hcal_hit_z);

		if(ecal_x_z_hits.first.size() == 0 || hcal_x_z_hits.first.size() == 0){continue;}

		float initial_energy = std::sqrt(std::pow(mom_x[0], 2) + std::pow(mom_y[0], 2) + std::pow(mom_z[0], 2));
		float photon_angle = std::sqrt((mom_x[0]*sin(-.025)) + (mom_z[0]*cos(-.025)))/std::sqrt(std::pow(mom_x[0],2) + std::pow(mom_y[0],2) + std::pow(mom_z[0],2));

		photon_energy->Fill(initial_energy);
		photon_angle_dist->Fill(photon_angle);

		photon_energy_hit_count_ecal->Fill(initial_energy, ecal_hit_x.GetSize());
		photon_energy_hit_count_hcal->Fill(initial_energy, hcal_hit_x.GetSize());

		for(size_t i = 0; i < ecal_hit_x.GetSize(); ++i){
			photon_energy_energy_in_ecal->Fill(initial_energy,ecal_ener[i]);	
		}
		for(size_t i = 0; i < hcal_hit_x.GetSize(); ++i){
			photon_energy_energy_in_hcal->Fill(initial_energy,hcal_ener[i]);
			z_vertex_hcal_hit->Fill(hcal_hit_z[i]);
		}

		//if(row_id[0] != 8){continue;}
		int ecal_count = 0;
		double ecal_energy_sum = 0;
		int hcal_count = 0;
		double hcal_energy_sum = 0;
		double ecal_max_energy = 0;
		double hcal_max_energy = 0;
		int hcal_max_energy_index = 0;
		for(size_t i = 0; i < ecal_x_z_hits.first.size(); ++i){
			//one_event_hit_map_ecal->Fill(ecal_x_z_hits.first[i], ecal_hit_y[i], ecal_ener[i]);
			//one_event_hit_map_zdc->Fill(ecal_x_z_hits.first[i], ecal_hit_y[i], ecal_x_z_hits.second[i], ecal_ener[i]);
			if(ecal_ener[i] < .002){
				ecal_energy_sum += ecal_ener[i];
				++ecal_count;
			}
			if(ecal_max_energy < ecal_ener[i]){
				ecal_max_energy = ecal_ener[i];
			}

		}
		for(size_t i = 0; i < hcal_x_z_hits.first.size(); ++i){
			//one_event_hit_map_hcal->Fill(hcal_x_z_hits.first[i], hcal_hit_y[i], hcal_x_z_hits.second[i], hcal_ener[i]);
			//one_event_hit_map_zdc->Fill(hcal_x_z_hits.first[i], hcal_hit_y[i], hcal_x_z_hits.second[i], hcal_ener[i]);
			//one_event_hit_map_hcal_no_weight->Fill(hcal_x_z_hits.first[i], hcal_hit_y[i], hcal_x_z_hits.second[i]);
			if(hcal_ener[i] < .002){
				hcal_energy_sum += hcal_ener[i];
				++hcal_count;
			}
			distance_in_the_hcal->Fill(hcal_x_z_hits.second[i]);
			distance_in_the_hcal_vs_hit_energy->Fill(hcal_x_z_hits.second[i], hcal_ener[i]);
			energy_of_photon_vs_distance_in_the_hcal->Fill(initial_energy, hcal_x_z_hits.second[i]);
			xy_spread_in_HCal->Fill(hcal_x_z_hits.first[i], hcal_hit_y[i]);
			if(hcal_max_energy < hcal_ener[i]){
				hcal_max_energy = hcal_ener[i];
				hcal_max_energy_index = i;
			}
		}
		energy_in_ecal_low->Fill(ecal_energy_sum/ecal_count);
		energy_in_hcal_low->Fill(hcal_energy_sum/hcal_count);
		ecal_low_energy_counts->Fill(ecal_count);
		hcal_low_energy_counts->Fill(hcal_count);
		peak_energy_per_event_ecal->Fill(ecal_max_energy);
		peak_energy_per_event_hcal->Fill(hcal_max_energy);
		peak_energy_per_event_position_hcal->Fill(hcal_x_z_hits.second[hcal_max_energy_index]);
	}

	photon_energy->Write();
	photon_angle_dist->Write();
	photon_energy_hit_count_ecal->Write();
	photon_energy_hit_count_hcal->Write();
	photon_energy_energy_in_ecal->Write();
	photon_energy_energy_in_hcal->Write();
	z_vertex_hcal_hit->Write();
	
	//one_event_hit_map_ecal->Write();
	//one_event_hit_map_hcal->Write();
	//one_event_hit_map_zdc->Write();

	energy_in_ecal_low->Write();
	energy_in_hcal_low->Write();
	ecal_low_energy_counts->Write();
	hcal_low_energy_counts->Write();
	peak_energy_per_event_ecal->Write();
	peak_energy_per_event_hcal->Write();
	peak_energy_per_event_position_hcal->Write();
	//one_event_hit_map_hcal_no_weight->Write();

	distance_in_the_hcal->Write();
	distance_in_the_hcal_vs_hit_energy->Write();
	energy_of_photon_vs_distance_in_the_hcal->Write();

	xy_spread_in_HCal->Write();

	output->Close();
	return 1;
}
