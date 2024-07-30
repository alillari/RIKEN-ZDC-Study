#include <TTree.h>
#include <TFile.h>
#include <TChain.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TMatrixD.h>
#include <TVectorD.h>
#include <vector>
#include <iostream>
#include <cmath>

std::vector<double> rotate_point(std::vector<double> point){
	float theta = .025;
	std::vector<double> rotated_point = {cos(theta)*point[0] + sin(theta)*point[2], point[1], -sin(theta)*point[0] + cos(theta)*point[2]};
	return rotated_point;
}

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

float distance_neutron_closest_gamma(std::vector<double> neutron_vec, std::vector<double> gamma1_vec, std::vector<double> gamma2_vec){
	float neutron_dist_from_gamma1 = std::sqrt(std::pow(neutron_vec[0]-gamma1_vec[0], 2) + std::pow(neutron_vec[1]-gamma1_vec[1], 2) + std::pow(neutron_vec[2]-gamma1_vec[2], 2));
	float neutron_dist_from_gamma2 = std::sqrt(std::pow(neutron_vec[0]-gamma2_vec[0], 2) + std::pow(neutron_vec[1]-gamma2_vec[1], 2) + std::pow(neutron_vec[2]-gamma2_vec[2], 2));

	if(neutron_dist_from_gamma1 > neutron_dist_from_gamma2){return neutron_dist_from_gamma2;}
	return neutron_dist_from_gamma2;
}


int main(){
	TChain* my_chain = new TChain("events");
	my_chain->Add("../lambda_20to220GeV.edm4hep.root");
	//my_chain->Add("/home/alessio/RIKENSUMMER/data/*.root");
	//my_chain->Add("/home/alessio/RIKENSUMMER/angle_data/*.root");

	TFile* output = new TFile("lambda_study_output.root", "RECREATE");

	TTreeReader my_reader(my_chain);

	TH1F* lambda_momentum_dist = new TH1F("lambda_momentum_dist","Distribution of #Lambda momentum; #Lambda momentum [GeV]; Counts", 25, 0, 270);
	TH1F* lambda_decay_position_dist = new TH1F("lambda_decay_position_dist", "Distribution of #Lambda decay length; #Lambda decay length [mm]; Counts", 25, 0, 35500);
	TH1F* lambda_angle_dist = new TH1F("lambda_angle_dist", "Distribution of #Lambda angle calculated from proton beam direction; Angle [rad]; Counts", 25, 0, .022);
	TH2F* lambda_momentum_decay_position = new TH2F("lambda_momentum_decay_position", "#Lambda momentum vs. decay position; momentum [GeV]; decay position [mm]", 25, 20, 270, 25, 0, 35500);
	TH2F* lambda_momentum_angle = new TH2F("lambda_momentum_angle", "#Lambda momentum vs. #Lambda angle with proton beam; #Lambda momentum [GeV]; #Lambda angle [rad]", 25, 0, 270, 25, 0, .022);
	TH2F* lambda_decay_position_angle = new TH2F("lambda_decay_position_angle", "#Lambda decay position vs. #Lambda angle with proton beam; Decay position [mm]; #Lambda angle [rad]", 25, 0, 35500, 25, 0, .022);
	TH1F* neutron_momentum_dist = new TH1F("neutron_momentum_dist", "Distribution of neutron momentum; neutron momentum [GeV]; Counts", 100, 0, 270);
	TH1F* pion_momentum_dist = new TH1F("pion_momentum_dist", "Distribution of #pi^{0} momentum; #pi^{0} momentum [GeV]; Counts", 100, 0, 100);
	TH2F* lambda_neutron_momentum_dist = new TH2F("lambda_neutron_momentum_dist", "#Lambda momentum vs neutron momentum; #Lambda momentum [GeV]; neutron momentum [GeV]", 100, 20, 270, 100, 0, 270);
	TH2F* lambda_pion_momentum_dist = new TH2F("lambda_pion_momentum_dist", "#Lambda momentum vs #pi^{0} momentum; #Lambda momentum [GeV]; #pi^{0} [GeV]", 100, 20, 270, 100, 0, 100);
	TH2F* lambda_decay_pion_momentum = new TH2F("lambda_decay_pion_momentum", "#Lambda decay distance vs #pi^{0} momentum; #Lambda decay distance [mm]; #pi^{0} [GeV]", 100, 0, 35500, 100, 0, 100);
	TH2F* lambda_decay_neutron_momentum = new TH2F("lambda_decay_neutron_momentum", "#Lambda decay distance vs neutron momentum; #Lambda decay distance [mm]; neutron momentum [GeV]", 100, 0, 35500, 100, 0, 270);
	TH1F* neutron_gamma_closest_distance_dist = new TH1F("neutron_gamma_closest_distance_dist", "Distance in Ecal of neutron and closest #gamma; Distance [mm]; Counts", 25, 0, 850);
	TH2F* lambda_momentum_closest_distance = new TH2F("lambda_momentum_closest_distance", "#Lambda momentum vs. closest distance of neutron and #gamma; #Lambda momentum [GeV]; n #gamma distance [mm]", 25, 20, 270, 25, 0, 850);
	TH2F* lambda_decay_dist_closest_distance = new TH2F("lambda_decay_dist_closest_distance", "#Lambda Decay distance vs closest distance of neutron and #gamma; #Lambda decay distance [mm]; n #gamma distance [mm]", 100, 0, 35500, 100, 0, 850);
	TH2F* lambda_angle_neutron_angle = new TH2F("lambda_angle_neutron_angle", "#Lambda angle with proton beam vs. neutron angle with proton beam; #Lambda angle [rad]; neutron angle [rad]", 100, 0, .022, 100, 0, .05);
	TH1F* neutron_gamma_endpoint_closest_distance_dist = new TH1F("neutron_gamma_endpoint_closest_distance_dist", "Distribution of the distance between neutron and closest gamma for all events; Distance [mm]; Counts", 200, 0, 1600);
	TH2F* lambda_momentum_vs_neutron_gamma_endpoint_closest_distance_dist = new TH2F("lambda_momentum_vs_neutron_gamma_endpoint_closest_distance_dist", "#Lambda momentum vs. closest neutron to #gamma distance; #Lambda momentum [GeV]; Distance [mm]", 100, 20, 270, 200, 0, 1600);

	TH1F* hit_lambda_momentum_dist = new TH1F("hit_lambda_momentum_dist", "Distribution of #Lambda momentum of all particles in ZDC; #Lambda momentum [GeV]; Counts", 25, 0, 270);
	TH1F* hit_lambda_decay_position_dist = new TH1F("hit_lambda_decay_position_dist", "Distribution of #Lambda decay length of all particles in ZDC; #Lambda decay length [mm]; Counts", 25, 0, 35500);
	TH1F* hit_lambda_angle_dist = new TH1F("hit_lambda_angle_dist", "Distribution of #Lambda angle with proton beam for all particles in Ecal; Angle [rad]; Counts", 25, 0, .022);
	TH2F* hit_lambda_momentum_decay_position = new TH2F("hit_lambda_momentum_decay_position", "#Lambda momentum vs. decay position of all particle in ZDC; momentum [GeV]; decay distance [mm]", 25, 20, 270, 25, 0, 35500);
	TH2F* hit_lambda_momentum_angle = new TH2F("hit_lambda_momentum_angle", "#Lambda momentum vs. #Lambda angle with proton beam for all particles in Ecal; #Lambda momentum [GeV]; #Lambda angle [rad]", 25, 0, 270, 25, 0, .022);
	TH2F* hit_lambda_decay_position_angle = new TH2F("hit_lambda_decay_position_angle", "#Lambda decay position vs. #Lambda angle with proton beam for all particles in Ecal; Decay position [mm]; #Lambda angle [rad]", 25, 0, 35500, 25, 0, .022);
	TH1F* hit_neutron_momentum_dist = new TH1F("hit_neutron_momentum_dist", "Distribution of neutron momentum of all particle in ZDC; neutron momentum [GeV]; Counts", 100, 0, 270);
	TH1F* hit_pion_momentum_dist = new TH1F("hit_pion_momentum_dist", "Distribution of #pi^{0} momentum of all particle in ZDC; #pi^{0} momentum [GeV]; Counts", 100, 0, 100);
	TH2F* hit_lambda_neutron_momentum_dist = new TH2F("hit_lambda_neutron_momentum_dist", "#Lambda momentum vs neutron momentum of all particle in ZDC; #Lambda momentum [GeV]; neutron momentum [GeV]", 100, 20, 220, 100, 0, 270);
	TH2F* hit_lambda_pion_momentum_dist = new TH2F("hit_lambda_pion_momentum_dist", "#Lambda momentum vs #pi^{0} momentum of all particle in ZDC; #Lambda momentum [GeV]; #pi^{0} [GeV]", 100, 20, 270, 100, 0, 100);
	TH2F* hit_lambda_decay_pion_momentum = new TH2F("hit_lambda_decay_pion_momentum", "#Lambda decay distance vs #pi^{0} momentum for all particle in ZDC; #Lambda decay distance [mm]; #pi^{0} momentum [GeV]", 100, 0, 35500, 100, 0, 100);
	TH2F* hit_lambda_decay_neutron_momentum = new TH2F("hit_lambda_decay_neutron_momentum", "#Lambda decay distance vs neutron momentum; #Lambda decay distance [mm]; neutron momentum [GeV]", 100, 0, 35500, 100, 0, 270);
	TH1F* hit_neutron_gamma_closest_distance_dist = new TH1F("hit_neutron_gamma_closest_distance_dist", "Distance in Ecal of neutron and closest #gamma for all particles in ZDC; Distance [mm]; Counts", 25, 0, 850);
	TH2F* hit_lambda_momentum_closest_distance = new TH2F("hit_lambda_momentum_closest_distance", "#Lambda momentum vs. closest distance of neutron and #gamma for all particles in ZDC; #Lambda momentum [GeV]; n #gamma distance [mm]", 100, 20, 270, 100, 0, 450);
	TH2F* hit_lambda_decay_dist_closest_distance = new TH2F("hit_lambda_decay_dist_closest_distance", "#Lambda Decay distance vs closest distance of neutron and #gamma for all particles in ZDC; #Lambda decay distance [mm]; n #gamma distance [mm]", 100, 0, 35500, 100, 0, 450);
	TH2F* hit_lambda_angle_neutron_angle = new TH2F("hit_lambda_angle_neutron_angle", "#Lambda angle with proton beam vs. neutron angle with proton beam for all particles in Ecal; #Lambda angle [rad]; neutron angle [rad]", 100, 0, .022, 100, 0, .05);
	TH1F* hit_neutron_gamma_endpoint_closest_distance_dist = new TH1F("hit_neutron_gamma_endpoint_closest_distance_dist", "Distribution of the distance between neutron and closest gamma for all particles in ZDC; Distance [mm]; Counts", 200, 0, 1600);
	TH2F* hit_lambda_momentum_vs_neutron_gamma_endpoint_closest_distance_dist = new TH2F("hit_lambda_momentum_vs_neutron_gamma_endpoint_closest_distance_dist", "#Lambda momentum vs. closest neutron to #gamma distance for all particles in ZDC; #Lambda momentum [GeV]; Distance [mm]", 100, 20, 270, 200, 0, 1600);

	TH1F* two_gamma_angle_hist = new TH1F("two_gamma_angle_hist", "Angle between two #gamma for all events; Angle (rad); Counts", 50, 0, 1);
	TH1F* neutron_angle_hist = new TH1F("neutron_angle_hist", "Angle of neutron from #Lambda direction for all events; Angle (rad); Counts", 25, 0, .01);
	TH2F* two_gamma_vs_neutron_angle = new TH2F("two_gamma_vs_neutron_angle", "Angle between two #gamma vs. neutron angle from #Lambda direction for all events; Two #gamma angle (rad); Neutron angle (rad)", 50, 0, 1, 50, 0, .01);
	TH2F* momentum_vs_two_gamma_angle = new TH2F("momentum_vs_two_gamma_angle", "Momentum of #Lambda vs. angle of two #gamma for all events; #Lambda momentum [GeV]; angle of two gamma (rad)", 100, 20, 270, 50, 0, 1);
	TH2F* momentum_vs_neutron_angle = new TH2F("momentum_vs_neutron_angle", "Momentum of #Lambda vs. angle of neutron from #Lambda for all events; #Lambda momentum [GeV]; Neutron angle (rad)", 100, 20, 270, 50, 0, .01);

	TH1F* hit_two_gamma_angle_hist = new TH1F("hit_two_gamma_angle_hist", "Angle between two #gamma for hit events; Angle (rad); Counts", 50, 0, 1);
	TH1F* hit_neutron_angle_hist = new TH1F("hit_neutron_angle_hist", "Angle of neutron from #Lambda direction for hit events; Angle (rad); Counts", 25, 0, .01);
	TH2F* hit_two_gamma_vs_neutron_angle = new TH2F("hit_two_gamma_vs_neutron_angle", "Angle between two #gamma vs. neutron angle from #Lambda direction for hit events; 2 #gamma angle (rad); Neutron angle (rad)", 50, 0, 1, 50, 0, 1);
	TH2F* hit_momentum_vs_two_gamma_angle = new TH2F("hit_momentum_vs_two_gamma", "Momentum of #Lambda vs. angle of two #gamma for hit events; #Lambda momentum [GeV]; two #gamma angle (rad)", 100, 20, 270, 50, 0, 1);
	TH2F* hit_momentum_vs_neutron_angle = new TH2F("hit_momentum_vs_neutron_angle", "Momentum of #Lambda vs. angle of neutron with #Lambda direction for hit events; #Lambda momentum [GeV]; Neutron angle (rad)", 100, 20, 270, 50, 0, .01);

	TH2F* neutron_hit_map = new TH2F("neutron_hit_map", "Hit map for neutron in Ecal, rotated frame; X axis [mm]; Y axis [mm]", 20, -300, 300, 20, -300, 300);
	TH2F* gamma_hit_map = new TH2F("gamma_hit_map", "Hit map for 2 gamma in Ecal, rotated frame; X axis [mm]; Y axis [mm]", 20, -300, 300, 20, -300, 300);


	TTreeReaderArray<int> pdg(my_reader, "MCParticles.PDG");
	TTreeReaderArray<double> vert_x(my_reader, "MCParticles.vertex.x");
	TTreeReaderArray<double> vert_y(my_reader, "MCParticles.vertex.y");
	TTreeReaderArray<double> vert_z(my_reader, "MCParticles.vertex.z");

	TTreeReaderArray<float> mom_x(my_reader, "MCParticles.momentum.x");
	TTreeReaderArray<float> mom_y(my_reader, "MCParticles.momentum.y");
	TTreeReaderArray<float> mom_z(my_reader, "MCParticles.momentum.z");	

	TTreeReaderArray<double> endpoint_x(my_reader, "MCParticles.endpoint.x");
	TTreeReaderArray<double> endpoint_y(my_reader, "MCParticles.endpoint.y");
	TTreeReaderArray<double> endpoint_z(my_reader, "MCParticles.endpoint.z");

	TTreeReaderArray<int> row_id(my_reader, "EventHeader.eventNumber");

	TTreeReaderArray<int> parent_id(my_reader, "_MCParticles_parents.index");

	while(my_reader.Next()){
		float mom_mag = std::sqrt(std::pow(mom_x[0], 2) + std::pow(mom_y[0], 2) + std::pow(mom_z[0], 2));
		float decay_dist = std::sqrt(std::pow(vert_x[1], 2) + std::pow(vert_y[1], 2) + std::pow(vert_z[1], 2));
		float lambda_angle = std::acos((mom_x[0]*std::sin(-.025) + mom_z[0]*std::cos(-.025))/(mom_mag));
		//std::cout<<"Lambda angle: "<<std::to_string((mom_x[0]*(-.025) + mom_z[0])/(mom_mag))<<std::endl;

		int gamma1_index = find_particle_index(22, pdg, 1, parent_id);
		int gamma2_index = find_particle_index(22, pdg, 2, parent_id);
		int pion_index = find_particle_index(111, pdg, 1, parent_id);
		int neutron_index = find_particle_index(2112, pdg, 1, parent_id);
		if(gamma1_index == 0 || gamma2_index == 0 || pion_index == 0 || neutron_index == 0){continue;}
		float neutron_mom_mag = std::sqrt(std::pow(mom_x[neutron_index], 2) + std::pow(mom_y[neutron_index], 2) + std::pow(mom_z[neutron_index], 2));
		if(pdg.GetSize() > 15 || neutron_mom_mag < 5){
			//std::cout<<"Low neutron momentum, event id: "<<std::to_string(row_id[0])<<std::endl;		
			continue;
		}

		float pion_mom_mag = std::sqrt(std::pow(mom_x[pion_index], 2) + std::pow(mom_y[pion_index], 2) + std::pow(mom_z[pion_index], 2));
		
		lambda_momentum_dist->Fill(mom_mag);
		lambda_decay_position_dist->Fill(decay_dist);
		lambda_momentum_decay_position->Fill(mom_mag, decay_dist);
		lambda_angle_dist->Fill(lambda_angle);
		lambda_momentum_angle->Fill(mom_mag, lambda_angle);
		lambda_decay_position_angle->Fill(decay_dist, lambda_angle);

		neutron_momentum_dist->Fill(neutron_mom_mag);
		pion_momentum_dist->Fill(pion_mom_mag);
		lambda_neutron_momentum_dist->Fill(mom_mag, neutron_mom_mag);
		lambda_pion_momentum_dist->Fill(mom_mag, pion_mom_mag);
		lambda_decay_pion_momentum->Fill(decay_dist, pion_mom_mag);
		lambda_decay_neutron_momentum->Fill(decay_dist, neutron_mom_mag);

		float gamma1_mom_mag = std::sqrt(std::pow(mom_x[gamma1_index], 2) + std::pow(mom_y[gamma1_index], 2) + std::pow(mom_z[gamma1_index], 2));
		float gamma2_mom_mag = std::sqrt(std::pow(mom_x[gamma2_index], 2) + std::pow(mom_y[gamma2_index], 2) + std::pow(mom_z[gamma2_index], 2));
		float two_gamma_angle = std::acos((mom_x[gamma1_index]*mom_x[gamma2_index]+mom_y[gamma1_index]*mom_y[gamma2_index]+mom_z[gamma1_index]*mom_z[gamma2_index])/(gamma1_mom_mag*gamma2_mom_mag));
		float neutron_angle = std::acos((mom_x[neutron_index]*std::sin(-.025)+mom_z[neutron_index]*std::cos(-.025))/(neutron_mom_mag));

		two_gamma_angle_hist->Fill(two_gamma_angle);
		neutron_angle_hist->Fill(neutron_angle);
		two_gamma_vs_neutron_angle->Fill(two_gamma_angle, neutron_angle);
		momentum_vs_two_gamma_angle->Fill(mom_mag, two_gamma_angle);
		momentum_vs_neutron_angle->Fill(mom_mag, neutron_angle);
		lambda_angle_neutron_angle->Fill(lambda_angle, neutron_angle);

		std::vector<double> gamma1_pos = loc_in_ecal(vert_x, vert_y, vert_z, mom_x, mom_y, mom_z, gamma1_index);
		std::vector<double> gamma2_pos = loc_in_ecal(vert_x, vert_y, vert_z, mom_x, mom_y, mom_z, gamma2_index);
		std::vector<double> neutron_pos = loc_in_ecal(vert_x, vert_y, vert_z, mom_x, mom_y, mom_z, neutron_index);

		float neutron_gamma_dist = distance_neutron_closest_gamma(neutron_pos, gamma1_pos, gamma2_pos);

		neutron_gamma_closest_distance_dist->Fill(neutron_gamma_dist);
		lambda_momentum_closest_distance->Fill(mom_mag, neutron_gamma_dist);
		lambda_decay_dist_closest_distance->Fill(decay_dist, neutron_gamma_dist);

		bool is_gamma1_in_ecal = gamma1_pos[0] < -600 && gamma1_pos[0] > -1200 && gamma1_pos[1] < 300 && gamma1_pos[1] > -300;
		bool is_gamma2_in_ecal = gamma2_pos[0] < -600 && gamma2_pos[0] > -1200 && gamma2_pos[1] < 300 && gamma2_pos[1] > -300;
		bool is_neutron_in_ecal = neutron_pos[0] < -600 && neutron_pos[0] > -1200 && neutron_pos[1] < 300 && neutron_pos[1] > -300;

		std::vector<double> gamma1_endpoint({endpoint_x[gamma1_index], endpoint_y[gamma1_index], endpoint_z[gamma1_index]});
		std::vector<double> gamma2_endpoint({endpoint_x[gamma2_index], endpoint_y[gamma2_index], endpoint_z[gamma2_index]});
		std::vector<double> neutron_endpoint({endpoint_x[neutron_index], endpoint_y[neutron_index], endpoint_z[neutron_index]});
		double neutron_gamma_endpoint_dist = distance_neutron_closest_gamma(neutron_endpoint, gamma1_endpoint, gamma2_endpoint);

		
		
		if(is_gamma1_in_ecal && is_gamma2_in_ecal && is_neutron_in_ecal){
			//std::cout<< std::to_string(row_id[0])<<std::endl;
			hit_lambda_momentum_dist->Fill(mom_mag);
			hit_lambda_decay_position_dist->Fill(decay_dist);
			hit_lambda_angle_dist->Fill(lambda_angle);
			hit_lambda_momentum_decay_position->Fill(mom_mag, decay_dist);
			hit_lambda_momentum_angle->Fill(mom_mag, lambda_angle);
			hit_lambda_decay_position_angle->Fill(decay_dist, lambda_angle);
			hit_neutron_momentum_dist->Fill(neutron_mom_mag);
			hit_pion_momentum_dist->Fill(pion_mom_mag);
			hit_lambda_neutron_momentum_dist->Fill(mom_mag, neutron_mom_mag);
			hit_lambda_pion_momentum_dist->Fill(mom_mag, pion_mom_mag);
			hit_lambda_decay_pion_momentum->Fill(decay_dist, pion_mom_mag);
			hit_lambda_decay_neutron_momentum->Fill(decay_dist, neutron_mom_mag);
			hit_neutron_gamma_closest_distance_dist->Fill(neutron_gamma_dist);
			hit_lambda_momentum_closest_distance->Fill(mom_mag, neutron_gamma_dist);
			hit_lambda_decay_dist_closest_distance->Fill(decay_dist, neutron_gamma_dist);
			hit_lambda_angle_neutron_angle->Fill(lambda_angle, neutron_angle);

			hit_two_gamma_angle_hist->Fill(two_gamma_angle);
			hit_neutron_angle_hist->Fill(neutron_angle);
			hit_two_gamma_vs_neutron_angle->Fill(two_gamma_angle, neutron_angle);
			hit_momentum_vs_two_gamma_angle->Fill(mom_mag, two_gamma_angle);
			hit_momentum_vs_neutron_angle->Fill(mom_mag, neutron_angle);

			std::vector<double> rotated_neutron_pos = rotate_point(neutron_pos);
			std::vector<double> rotated_gamma1_pos = rotate_point(gamma1_pos);
			std::vector<double> rotated_gamma2_pos = rotate_point(gamma2_pos);

			neutron_hit_map->Fill(rotated_neutron_pos[0], rotated_neutron_pos[1]);
			gamma_hit_map->Fill(rotated_gamma1_pos[0], rotated_gamma1_pos[1]);
			gamma_hit_map->Fill(rotated_gamma2_pos[0], rotated_gamma2_pos[1]);
		
		}
	}

	TH2F* efficiency_plot = new TH2F(*hit_lambda_momentum_decay_position);
	TH2F* efficiency_mom_angle_plot = new TH2F(*hit_lambda_momentum_angle);
	TH1F* efficiency_lambda_decay_distance = new TH1F(*hit_lambda_decay_position_dist);
	TH1F* efficiency_lambda_momentum = new TH1F(*hit_lambda_momentum_dist);

	efficiency_plot->SetName("efficiency_plot");
	efficiency_plot->SetTitle("Efficiency plot of #Lambda momentum vs. #Lambda decay distance");
	efficiency_plot->GetXaxis()->SetTitle("#Lambda momentum [GeV]");
	efficiency_plot->GetYaxis()->SetTitle("#Lambda decay distance [mm]");

	efficiency_mom_angle_plot->SetName("efficiency_mom_angle_plot");
	efficiency_mom_angle_plot->SetTitle("Efficiency plot of #Lambda momentum vs. #Lambda angle with proton beam");
	efficiency_mom_angle_plot->GetXaxis()->SetTitle("#Lambda momentum [GeV]");
	efficiency_mom_angle_plot->GetYaxis()->SetTitle("#Lambda angle [rad]");

	efficiency_lambda_decay_distance->SetName("efficiency_lambda_decay_distance");
	efficiency_lambda_decay_distance->SetTitle("Efficiency plot of #Lambda decay distance");
	efficiency_lambda_decay_distance->GetXaxis()->SetTitle("#Lambda decay distance [mm]");

	efficiency_lambda_momentum->SetName("efficiency_lambda_momentum");
	efficiency_lambda_momentum->SetTitle("Efficiency plot of #Lambda momentum");
	efficiency_lambda_momentum->GetXaxis()->SetTitle("#Lambda momentum [GeV]");

	efficiency_plot->Divide(lambda_momentum_decay_position);
	efficiency_lambda_decay_distance->Divide(lambda_decay_position_dist);
	efficiency_lambda_momentum->Divide(lambda_momentum_dist);
	efficiency_mom_angle_plot->Divide(lambda_momentum_angle);

	lambda_momentum_dist->Write();
	lambda_decay_position_dist->Write();
	lambda_angle_dist->Write();
	lambda_momentum_decay_position->Write();
	lambda_momentum_angle->Write();
	lambda_decay_position_angle->Write();
	neutron_momentum_dist->Write();
	pion_momentum_dist->Write();
	lambda_neutron_momentum_dist->Write();
	lambda_pion_momentum_dist->Write();
	lambda_decay_pion_momentum->Write();
	lambda_decay_neutron_momentum->Write();
	neutron_gamma_closest_distance_dist->Write();
	lambda_momentum_closest_distance->Write();
	lambda_decay_dist_closest_distance->Write();
	lambda_angle_neutron_angle->Write();

	hit_lambda_momentum_dist->Write();
	hit_lambda_decay_position_dist->Write();
	hit_lambda_angle_dist->Write();
	hit_lambda_momentum_decay_position->Write();
	hit_lambda_momentum_angle->Write();
	hit_lambda_decay_position_angle->Write();
	hit_neutron_momentum_dist->Write();
	hit_pion_momentum_dist->Write();
	hit_lambda_neutron_momentum_dist->Write();
	hit_lambda_pion_momentum_dist->Write();
	hit_lambda_decay_pion_momentum->Write();
	hit_lambda_decay_neutron_momentum->Write();
	hit_neutron_gamma_closest_distance_dist->Write();
	hit_lambda_momentum_closest_distance->Write();
	hit_lambda_decay_dist_closest_distance->Write();
	hit_lambda_angle_neutron_angle->Write();
	
	efficiency_plot->Write();
	efficiency_lambda_decay_distance->Write();
	efficiency_lambda_momentum->Write();
	efficiency_mom_angle_plot->Write();

	two_gamma_angle_hist->Write();
	neutron_angle_hist->Write();
	two_gamma_vs_neutron_angle->Write();
	momentum_vs_two_gamma_angle->Write();
	momentum_vs_neutron_angle->Write();

	hit_two_gamma_angle_hist->Write();
	hit_neutron_angle_hist->Write();
	hit_two_gamma_vs_neutron_angle->Write();
	hit_momentum_vs_two_gamma_angle->Write();
	hit_momentum_vs_neutron_angle->Write();

	neutron_hit_map->Write();
	gamma_hit_map->Write();

	output->Close();

	return 1;
}
