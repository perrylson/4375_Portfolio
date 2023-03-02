/*
Program implements Logistic Regression in C++; reads Titanic data; and predict survival based on gender.
It displays weight's coefficients and various metrics.
*/

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>    

using namespace std;


//Implement sigmoid function
double sigmoid(double num){
    return 1.0 / (1+exp(-1*num));
}

//Map prob to either 0 or 1; threshhold is set to 0.5
double transformProb(double prob){

    return prob >= 0.5 ? 1 : 0;

}


//Perform matrix addition or subtraction (matX +/- matY) with given flag variable
vector<vector<double>> matrixAddSub(vector<vector<double>> matX, vector<vector<double>> matY, bool flag){

    vector<vector<double>> vector_res(matX.size(), vector<double>(matY[0].size()));

    for(int i = 0; i < vector_res.size(); i++){
        for(int j = 0; j < vector_res[i].size(); j++){
            vector_res[i][j] = flag ? matX[i][j] + matY[i][j] : matX[i][j] - matY[i][j];
        }
    }
    return vector_res;
}

//Perform matrix multiplication (matX * matY)
vector<vector<double>> matrixMult(vector<vector<double>> matX, vector<vector<double>> matY){

    vector<vector<double>> vector_prod(matX.size(), vector<double>(matY[0].size()));

    for(int i = 0; i < vector_prod.size(); i++){
        for(int j = 0; j < vector_prod[i].size(); j++){
            for (int k = 0; k < matX[0].size(); k++){
                vector_prod[i][j] += matX[i][k] * matY[k][j];
            }
        }
    }

    return vector_prod;
}

//Perform scalar multiplication (num * matX)
 vector<vector<double>> scalarMult(vector<vector<double>> matX, double num){

    for(int i =0; i < matX.size(); i++){
        for(int j =0; j < matX[i].size(); j++){
            matX[i][j] *= num;
        }
    }
    return matX;
}

//Apply function argument to each entry in matX
 vector<vector<double>> matrixMap(vector<vector<double>> matX, double (*func)(double)){

    for(int i =0; i < matX.size(); i++){
        for(int j =0; j < matX[i].size(); j++){
            matX[i][j] = func(matX[i][j]);
        }
    }

    return matX;
}

//Transpose matrix
vector<vector<double>> matrixTranspose(vector<vector<double>> matX){
    
    vector<vector<double>> vector_transpose(matX[0].size(), vector<double>(matX.size()));
    for(int i =0; i < matX.size(); i++){
        for(int j =0; j < matX[0].size(); j++){
            vector_transpose[j][i] = matX[i][j];
        }
    }

    return vector_transpose;

}


//Perform gradient descent on the dataset; can set number of iterations and learning rate value
vector<vector<double>> performGradDescent(vector<vector<double>> dataset, int iter_amt = 500000, float learning_rate = 0.001){
    
    //Initialize starting weights for intercept and gender column
    vector<vector<double>> weights = {{1.0}, {1.0}};

    //Initialize labels and training matrices
    vector<vector<double>> labels = {dataset[0].size(), vector<double>(1)};
    vector<vector<double>> data_matrix(dataset[1].size(), vector<double>(dataset.size()));

    //Fill in matrices
    for(int i =0; i < labels.size(); i++){
        labels[i][0] = dataset[0][i];
    }

    for(int i =0; i < dataset[0].size(); i++){
        data_matrix[i][0] = 1;
        data_matrix[i][1] = dataset[1][i];
    }

    //Perform gradient descent over a given number of iterations
    for(int i =0; i < iter_amt; i++ ){

        //Perform matrix multiplication between data matrix and weights
        vector<vector<double>> vector_prod = matrixMult(data_matrix, weights);

        //Apply sigmoid function to get survival probabilities
        vector_prod = matrixMap(vector_prod, &sigmoid);

        //Calculate error matrix with matrix subtraction; labels - vector_prod
        vector<vector<double>> error = matrixAddSub(labels, vector_prod, false);

        //Transpose data matrix
        vector<vector<double>> intermediateOp1 = matrixTranspose(data_matrix);

        //Perform matrix multiplication between tranposed data and error matrices
        vector<vector<double>> intermediateOp2 = matrixMult(intermediateOp1, error);

        //Apply scalar multiplication with learning rate; learning_rate * intermediateOp2
        vector<vector<double>> intermediateOp3 = scalarMult(intermediateOp2, learning_rate);

        //Calculate new weights with matrix addition; weights + intermediateOp3
        weights = matrixAddSub(weights, intermediateOp3, true);

    }

    return weights;
}


//Perform inference on a given dataset; uses the weight parameter for Logistic Regression
void performInference(vector<vector<double>> dataset, vector<vector<double>> weights){

    //Create a label matrix
    vector<vector<double>> labels = {dataset[0].size(), vector<double>(1)};

    double accuracy = 0, sensitivity = 0, specificity = 0;
    int TP = 0, TN = 0, FP = 0, FN = 0;

    //Initialize a matrix that will hold the intercept and gender column
    vector<vector<double>> data_matrix(dataset[1].size(), vector<double>(dataset.size()));

    //Fill in matrices
    
    for(int i =0; i < labels.size(); i++){
        labels[i][0] = dataset[0][i];
    }

    for(int i =0; i < dataset[0].size(); i++){
        data_matrix[i][0] = 1;
        data_matrix[i][1] = dataset[1][i];
    }

    //Perform Logistic Regression

    //Perform matrix multiplication between data matrix and weights
    vector<vector<double>> vector_prod = matrixMult(data_matrix, weights);

    //Apply sigmoid function to get survival probabilities 
    vector_prod = matrixMap(vector_prod, &sigmoid);

    //Map probabilities to either 0 or 1 
    vector_prod = matrixMap(vector_prod, &transformProb);

    //Iterate through vector and record true positive, true negative, false positive, and false negative instances
    for (int i = 0; i < vector_prod.size(); i++)
    {
        for (int j = 0; j < vector_prod[i].size(); j++){
            
            if(vector_prod[i][j] == labels[i][0] && labels[i][0] == 0){
                TN += 1;
            }
            else if (vector_prod[i][j] == labels[i][0] && labels[i][0] == 1){
                TP += 1;
            }
            else if (vector_prod[i][j] == 1 && labels[i][0] == 0){
                FP += 1;
            }            
            else if (vector_prod[i][j] == 0 && labels[i][0] == 1){
                FN += 1;
            }            

        }
    }

    //Calculate metrics
    accuracy = double(TP + TN)/(TP+TN+FP+FN);
    sensitivity = double(TP)/(TP+FN);
    specificity = double(TN)/(TN+FP);

    //Output metrics
    cout << "Accuracy: " << accuracy << endl;
    cout << "Sensitivity: " << sensitivity << endl;
    cout << "Specificity: " << specificity << endl;

}

int main(int argc, char** argv) {
	ifstream inFS;
	string line;
	string dummy_in, pclass_in, survived_in, sx_in, age_in;
	const int MAX_LEN = 1500;
	vector<double> survived(MAX_LEN);
	vector<double> sx(MAX_LEN);


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

		survived.at(numObservations) = stof(survived_in);
		sx.at(numObservations) = stof(sx_in);

		numObservations++;
	}

	inFS.close();

	survived.resize(numObservations);
	sx.resize(numObservations);


    //Use first 800 observations as training set; the remaining data is used for test inference
    vector <double> survived_train(survived.begin(), survived.begin()+800);
    vector <double> sx_train(sx.begin(), sx.begin()+800);

    vector <double> survived_test(survived.begin()+800, survived.end());
    vector <double> sx_test(sx.begin()+800, sx.end());

    vector<vector<double>> trainingSet
    {
        survived_train,
        sx_train
    };

    vector<vector<double>> testSet
    {
        survived_test,
        sx_test
    };

    //Record training time for Logistic Regression
    chrono::time_point<chrono::system_clock> start, end;
 
    start = chrono::system_clock::now();

    //Perform gradient descent with 500 iterations
    vector<vector<double>> weights = performGradDescent(trainingSet, 500);
    
    end = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = end - start; 

    //Output training time
    cout << "Training time: " << elapsed_seconds.count() << endl;

    cout << endl;
    //Output final weights
    cout << "Coefficients [intercepts, sex]: " << "[" << weights[0][0] << ", " << weights[1][0] << "]" << endl;
    cout << endl;

    //Perform inference on test sets with trained weights
    performInference(testSet, weights);
}