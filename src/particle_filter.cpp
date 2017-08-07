/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (unsigned int i = 0; i < num_particles; ++i) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		// Print your samples to the terminal.
		particle.weight = 1;
		particles.push_back(particle);
		weights.push_back(particle.weight);

	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	double x, y, theta;
	for (unsigned int i=0; i < particles.size(); i++) {
		Particle p = particles[i];
		if (fabs(yaw_rate) < 0.001) {
			x = p.x + velocity*delta_t*(cos(p.theta));
			y = p.y + velocity*delta_t*(sin(p.theta));
			theta = p.theta;

		} else {
			x = p.x + velocity/yaw_rate*(sin(p.theta + yaw_rate*delta_t)-sin(p.theta));
			y = p.y + velocity/yaw_rate*(cos(p.theta)-cos(p.theta + yaw_rate*delta_t));
			theta = p.theta + yaw_rate*delta_t;
		}
		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_theta(theta, std_pos[2]);
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); i++){

		double minimum = 1e100;
		int index = -1;
		for (unsigned int j = 0; j < predicted.size(); j++){
			double dx = (observations[i].x - predicted[j].x);
			double dy = (observations[i].y - predicted[j].y);
			double distance = sqrt(dx*dx+dy*dy);
			if (distance<minimum){
				minimum = distance;
				index = j;
			}
		}

		observations[i].id = index;

	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	float sensor_range_2 = sensor_range*sensor_range;

	for (unsigned int i = 0; i< particles.size();i++){
		Particle p = particles[i];
		std::vector<LandmarkObs> predicted;
		std::vector<LandmarkObs> map_coord_obs;

		for (unsigned int j = 0; j< map_landmarks.landmark_list.size();j++){
			float l_x = map_landmarks.landmark_list[j].x_f;
			float l_y = map_landmarks.landmark_list[j].y_f;
			if (((l_x-p.x)*(l_x-p.x)+(l_y-p.y)*(l_y-p.y))<sensor_range_2){
				LandmarkObs lm_filtered;
				lm_filtered.id = map_landmarks.landmark_list[j].id_i;
				lm_filtered.x =  l_x;
				lm_filtered.y =  l_y;
				predicted.push_back(lm_filtered);
			}
		}
		for (unsigned int j = 0; j< observations.size();j++){
			LandmarkObs new_lmo;
			new_lmo.id = observations[j].id;
			new_lmo.x = p.x + cos(p.theta)*observations[j].x -sin(p.theta)*observations[j].y;
			new_lmo.y = p.y + cos(p.theta)*observations[j].y +sin(p.theta)*observations[j].x;
			map_coord_obs.push_back(new_lmo);
		}
		dataAssociation(predicted, map_coord_obs);
		double ac_prob = 1.0;
		for (unsigned int j = 0; j< map_coord_obs.size();j++){
			int mco_id = map_coord_obs[j].id;
			double x = map_coord_obs[j].x;
			double u_x = predicted[mco_id].x;
			double y = map_coord_obs[j].y;
			double u_y =  predicted[mco_id].y;
			ac_prob *= multivariate_gaussian(x, y, u_x, u_y, std_landmark[0],std_landmark[1]);

		}
		p.weight = ac_prob;
		weights[i] = ac_prob;
	}

}

double ParticleFilter::multivariate_gaussian(double x,double y,double u_x,double u_y,double s_x,double s_y){
	double dx2 = (x-u_x)*(x-u_x)/(2*s_x*s_x);
	double dy2 = (y-u_y)*(y-u_y)/(2*s_y*s_y);
 	double nor = 1.0/(2.0*M_PI*s_x*s_y);
	return nor*exp(-(dx2+dy2));
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());
	vector<Particle> new_particles;

	for (unsigned int i=0; i < num_particles; i++)
	{
		new_particles.push_back(particles[distribution(gen)]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
