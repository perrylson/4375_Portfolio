/*
Program implements Naive Bayes in C++; reads Titanic data; and predict survival based on passenger class, gender, and age.
It displays apriori, likelihood for discrete and continuous predictors, and various metrics
*/

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>    

using namespace std;

const double PI = 3.14159265358979323846;


//Likelihood of continuous predictor - age 
double calc_age_lh(double age, double mean_age, double var_age){
    return 1 / (var_age*sqrt(2*PI)) * exp(-(pow(age-mean_age, 2))/(2*pow(var_age, 2)));
}

//Calculate Baye's Theorem
int calculateBTheorem(int pclass, int sx, double age, vector<double> apriori, vector<vector<double>> lh_pclass,
vector<vector<double>> lh_sx, vector<double> age_mean, vector<double> age_var){
    double calc_s = lh_pclass[1][pclass] * lh_sx[1][sx] * apriori[1] * calc_age_lh(age, age_mean[1], age_var[1]);
    double calc_d = lh_pclass[0][pclass] * lh_sx[0][sx] * apriori[0] * calc_age_lh(age, age_mean[0], age_var[0]);
    double denom = lh_pclass[1][pclass] * lh_sx[1][sx] * calc_age_lh(age, age_mean[1], age_var[1]) * apriori[1] + 
    lh_pclass[0][pclass] * lh_sx[0][sx] * calc_age_lh(age, age_mean[0], age_var[0]) * apriori[0];

    double prob_surviv = calc_s / denom;
    double prob_dead = calc_d / denom;

    return prob_surviv >= prob_dead ? 1 : 0;
}

//Perform inference on test dataset
void performInference(vector<double> survived_vec, vector<double> pclass_vec, vector<double> sx_vec, vector<double> age_vec, vector<double> apriori, vector<vector<double>> lh_pclass,
vector<vector<double>> lh_sx, vector<double> age_mean, vector<double> age_var){

    double accuracy = 0, sensitivity = 0, specificity = 0;
    int TP = 0;
    int TN = 0;
    int FP = 0;
    int FN = 0;

    for(int i =0; i < survived_vec.size(); i++){

        int res = calculateBTheorem(pclass_vec[i], sx_vec[i], age_vec[i], apriori, lh_pclass, lh_sx, age_mean, age_var);
        
        if(survived_vec[i] == res && survived_vec[i] == 0){
            TN += 1;
        }
        else if (survived_vec[i] == res && survived_vec[i] == 1){
            TP += 1;
        }
        else if (survived_vec[i] != res && survived_vec[i] == 0){
            FP += 1;
        }            
        else if (survived_vec[i] != res && survived_vec[i] == 1){
            FN += 1;
        }    
        
    }

    accuracy = double(TP + TN)/(TP+TN+FP+FN);
    sensitivity = double(TP)/(TP+FN);
    specificity = double(TN)/(TN+FP);

    cout << "Accuracy: " << accuracy << endl;
    cout << "Sensitivity: " << sensitivity << endl;
    cout << "Specificity: " << specificity << endl;


}



int main(int argc, char** argv) {
	ifstream inFS;
	string line;
	string dummy_in, pclass_in, survived_in, sx_in, age_in;
	const int MAX_LEN = 1500;
	vector<double> pclass(MAX_LEN);
	vector<double> survived(MAX_LEN);
	vector<double> sx(MAX_LEN);
	vector<double> age(MAX_LEN);

	inFS.open("titanic_project.csv");
	if (!inFS.is_open()) {
		cout << "Could not open titanic_project.csv." << endl;
		return 1;
	}

	getline(inFS, line);


	int numObservations = 0;
	while (inFS.good()) {
		getline(inFS, dummy_in, ',');
		getline(inFS, pclass_in, ',');
		getline(inFS, survived_in, ',');
		getline(inFS, sx_in, ',');
		getline(inFS, age_in, '\n');

		pclass.at(numObservations) = stof(pclass_in);
		survived.at(numObservations) = stof(survived_in);
		sx.at(numObservations) = stof(sx_in);
		age.at(numObservations) = stof(age_in);

		numObservations++;
	}

	inFS.close();


	pclass.resize(numObservations);
	survived.resize(numObservations);
	sx.resize(numObservations);
	age.resize(numObservations);

    //Use first 800 observations as training set; the remaining data is used for test inference
    vector <double> pclass_train(pclass.begin(), pclass.begin()+800);
    vector <double> survived_train(survived.begin(), survived.begin()+800);
    vector <double> sx_train(sx.begin(), sx.begin()+800);
    vector <double> age_train(age.begin(), age.begin()+800);

    vector <double> pclass_test(pclass.begin()+800, pclass.end());
    vector <double> survived_test(survived.begin()+800, survived.end());
    vector <double> sx_test(sx.begin()+800, sx.end());
    vector <double> age_test(age.begin()+800, age.end());

    vector<vector<double>> trainingSet
    {
        survived_train,
        sx_train,
        pclass_train,
        age_train
    };

    vector<vector<double>> testSet
    {
        survived_test,
        sx_test,
        pclass_test,
        age_test
    };

    chrono::time_point<chrono::system_clock> start, end;
 
    start = chrono::system_clock::now();

    //Calculate aprioris
    double numDead = count(survived_train.begin(), survived_train.end(), 0), numSurvived = count(survived_train.begin(), survived_train.end(), 1);
    double priorDead = numDead/survived_train.size(), priorSurvived = numSurvived/survived_train.size();

    vector<double> countSurvived = {numDead, numSurvived};
    vector<double> apriori = {priorDead, priorSurvived};

    vector<vector<double>> lh_pclass(2, vector<double>(3));
    vector<vector<double>> lh_sx(2, vector<double>(2));
    vector<double> age_mean = {0,0};
    vector<double> age_var = {0,0};

    //Calculate likelihood for pclass
    for(int i = 0; i < 2; i++){
        for(int j = 1; j < 4; j++){
            int count = 0;
            for (int x = 0; x < survived_train.size(); x++){
                if(survived_train[x] == i && pclass_train[x] == j){
                    count += 1;
                }
            }
            lh_pclass[i][j-1] = count / countSurvived[i];
        }
    } 

    //Calculate likelihood for gender   
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            int count = 0;
            for (int x = 0; x < survived_train.size(); x++){
                if(survived_train[x] == i && sx_train[x] == j){
                    count += 1;
                }
            }
            lh_sx[i][j] = count / countSurvived[i];
        }
    } 

    //Find mean and variance so program can calculate likelihood for age
    for(int i =0; i < 2; i++){
        double sum_mean = 0, sum_var = 0;
        int count = 0;
        for(int j = 0; j < survived_train.size(); j++){
            if(survived_train[j] == i){
                sum_mean += age_train[j];
                count += 1;
            }
        }
        double mean = sum_mean / count;

        for(int j = 0; j < survived_train.size(); j++){
            if(survived_train[j] == i){
                sum_var += pow(age_train[j] - mean, 2);
            }
        } 
        age_mean[i] = mean;
        age_var[i] = sum_var / (count-1);
    }



    end = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = end - start; 

    cout << "A-priori probabilities:" << endl;
    cout << 0 << "         " << 1 << endl;
    cout << priorDead << "   " << priorSurvived << endl;
    cout << endl;

    cout << "Conditional probabilities:" << endl;
    
    cout << "Pclass" << endl;
    cout << "  " << 1 << "        " << 2 << "       " << 3 << endl;
    for (int i = 0; i < lh_pclass.size(); i++)
    {
        cout << i << " ";
        for (int j = 0; j < lh_pclass[i].size(); j++){
            cout << lh_pclass[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "Sex" << endl;
    cout << "  " << 0 << "        " << 1 << endl;
    for (int i = 0; i < lh_sx.size(); i++)
    {   cout << i << " ";
        for (int j = 0; j < lh_sx[i].size(); j++){
            cout << lh_sx[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    cout << "Age" << endl;
    cout << "  " << "Mean" << "        " << "Variance" << endl;
    cout << 0 << " " << age_mean[0] << "     " << age_var[0] <<endl;
    cout << 1 << " " << age_mean[1] << "     " << age_var[1] <<endl;

    cout << endl;
    cout << "Training time: " << elapsed_seconds.count() << endl;
    performInference(survived_test, pclass_test, sx_test, age_test, apriori, lh_pclass, lh_sx, age_mean, age_var);


}




