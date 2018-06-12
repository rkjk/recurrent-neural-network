#include "rnn_headers.h"

/*
Libraries Used:
  Eigen - Matrix Algebra library
*/


int main()
{
  cout <<"-------------------------------------------------"<< endl
     << "-------------------------------------------------"<< endl
     << "-------------------------------------------------"<< endl
     << "A Recurrent Neural Network to perform binary addition of 2 numbers" <<endl
     << "Library used for fast matrix computations : Eigen " << endl
     << "-------------------------------------------------" << endl
     << "-------------------------------------------------" << endl
     << "-------------------------------------------------" << endl;


  srand(235);

  unsigned numIter;
  bool printProgress;

  vector <double> learnRate = {0.005,0.01,0.05,0.1};
  vector <future<int>> fut(learnRate.size());
  vector <int> iterations(learnRate.size());

  numIter = 60000;

  cout << "Maximum Training Sample Size : " << numIter << endl;
  cout << "Learning Rates being tested : " ;
  for(auto i : learnRate) {
    cout << i << ", " ;
  }
  cout << endl;

  cout << "--------------------------------" << endl
       << "--------------------------------" << endl
       << "Estimating Learning rate of RNN " << endl << endl
       << "Train RNN for a maximum of " << numIter << " training samples " << endl
       << "RNN training will stop when : " << endl
       << "	1) The Training Error falls below a threshold , in this case 0.03 " << endl
       << "	2) The Training runs for the maximum possible steps and then exits " << endl << endl
       << "We will choose the learning rate that returns in the least possible iterations " << endl
       << "--------------------------------" << endl
       << "--------------------------------" << endl << endl << endl;

  for (int i = 0; i < learnRate.size(); ++i) {
    fut[i] = async(&rnnStep, learnRate[i], numIter, false);
  }

  cout << "Launched " << learnRate.size() << " threads " << endl << endl;

  for (int i = 0; i < learnRate.size(); ++i) {
    try {
      iterations[i] = fut[i].get();
    }
    catch(exception &e) {
      cout << e.what() << endl;
    }
  }


  cout << "The Learning Rate and their corresponding training iterations "<< endl;
  int min = numIter;
  int argmin;
    cout <<"--------------------------" << endl
	 <<"Learning Rate   Iterations" << endl;
  for(int i = 0; i < learnRate.size(); ++i) {
    cout << learnRate[i] << "                 " <<  iterations[i] << endl;
    if(min > iterations[i]) {
      min = iterations[i];
      argmin = i;
    }
  }
  

  cout << "--------------------------------" << endl;

  cout << "Learning Rate with fastest convergence is " << learnRate[argmin] << endl;

  cout << "Training RNN with different training examples " << endl;

  int temp = rnnStep(learnRate[argmin], numIter, true);

}
