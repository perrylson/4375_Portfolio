/*
Program reads a csv file; extracts data as numeric vectors; and calculates and displays various statistics
*/

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

//Function vec_sum finds the sum of a numeric vector
double vec_sum(vector<double> vec){
	double sum_elems = 0;

	for(vector<double>::iterator itr = vec.begin(); itr != vec.end(); ++itr)
		sum_elems += *itr;
	
	return sum_elems;
}

//Function vec_mean finds the mean of a numeric vector
double vec_mean(vector<double> vec){
	return vec_sum(vec)/vec.size();
}

//Function vec_mean finds the median of a numeric vector
double vec_median(vector<double> vec){
	sort(vec.begin(), vec.end());
	
	return (vec.size()%2==0) ? (vec.at((vec.size()/2)-1) + vec.at(vec.size()/2))/2 : vec.at(floor(vec.size()/2));
}

//Function vec_range finds the range of a numeric vector
double vec_range(vector<double> vec){
	sort(vec.begin(), vec.end());

	return vec.back()-vec.front();
}

//Function print_stats calls the earlier stat functions and prints various statistics 
void print_stats(vector<double> vec) {
	cout << "Sum: " << vec_sum(vec) << endl;
	cout << "Mean: " << vec_mean(vec) << endl;
	cout << "Median: " << vec_median(vec) << endl;
	cout << "Range: " << vec_range(vec) << endl;
}

//Function covar calculates the covariance between two numeric vectors
double covar(vector<double> vecX, vector<double> vecY){
	double vecXMean = vec_mean(vecX), vecYMean = vec_mean(vecY), summation = 0;

	for(vector<double>::iterator itrX = vecX.begin(), itrY = vecY.begin(); itrX != vecX.end() && itrY != vecY.end(); (++itrX, ++itrY))
		summation += (*itrX-vecXMean)*(*itrY-vecYMean);

	return summation / (vecX.size()-1);
}

//Function cor calculates the correlation between two numeric vectors
double cor(vector<double> vecX, vector<double> vecY){
	return covar(vecX, vecY)/(sqrt(covar(vecX, vecX))*sqrt(covar(vecY, vecY)));
}

int main(int argc, char** argv) {

	ifstream inFS;
	string line;
	string rm_in, medv_in;
	const int MAX_LEN = 1000;
	vector<double> rm(MAX_LEN);
	vector<double> medv(MAX_LEN);

	//Opening file
	cout << "Opening file Boston.csv." << endl;

	inFS.open("Boston.csv");
	if (!inFS.is_open()) {
		cout << "Could not open Boston.csv." << endl;
		return 1;
	}

	cout << "Reading line 1" << endl;
	getline(inFS, line);

	cout << "heading: " << line << endl;

	int numObservations = 0;
	while (inFS.good()) {
		getline(inFS, rm_in, ',');
		getline(inFS, medv_in, '\n');

		rm.at(numObservations) = stof(rm_in);
		medv.at(numObservations) = stof(medv_in);

		numObservations++;
	}

	rm.resize(numObservations);
	medv.resize(numObservations);

	cout << "New length " << rm.size() << endl;

	cout << "Closing file Boston.csv." << endl;
	inFS.close();

	cout << "Number of records: " << numObservations << endl;

	//Display statistics
	cout << "\nStats for rm" << endl;
	print_stats(rm);

	cout << "\nStats for medv" << endl;
	print_stats(medv);

	cout << "\nCovariance = " << covar(rm, medv) << endl;

	cout << "\nCorrelation = " << cor(rm, medv) << endl;

	cout << "\nProgram terminated" << endl;
}
