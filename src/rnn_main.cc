#include "rnn_headers.h"


int rnnStep(double alpha, unsigned numIter, bool printProgress)
{
  /*
   * Function to train the Recurrent Neural Network
   *
   * arguments:
   *  alpha : Learning rate of the RNN
   *  numIter : Maximum Number of Training Samples to train RNN
   *  printProgress : Print the Error every 1000 samples
  */




  /*
   *  input_dim : Number of neurons in Input layer
   *  hidden_dim : Number of neurons in Hidden layer
   *  output_dim : Number of neurons in output layer 
   */



  int input_dim = 2;
  int hidden_dim = 16;
  int output_dim = 1;

  //Largest number that can be added without overflowing 
  int largest_number = pow(2,binary_dim);



  /*
   *Eigen MatrixXd containers to hold the network matrices
   */
  MatrixXd synapse_0 = MatrixXd::Random(input_dim, hidden_dim);  //2x16 matrix for input->hidden
  MatrixXd synapse_1 = MatrixXd::Random(hidden_dim, output_dim);  //16x1 matrix for hidden->output
  MatrixXd synapse_h = MatrixXd::Random(hidden_dim, hidden_dim);  //16x16 matrix for hidden->hidden

  /*
   *Matrices to store weight updates.
   *These matrices will be updated during the backpropagation phase of the training algorithm
   */
  MatrixXd synapse_0_update = MatrixXd::Zero(input_dim,hidden_dim);
  MatrixXd synapse_1_update = MatrixXd::Zero(hidden_dim, output_dim);  
  MatrixXd synapse_h_update = MatrixXd::Zero(hidden_dim, hidden_dim);  

  

  /*
   * Training for a batch of samples
   *
   *
   * For each training sample, the 2 phases are:
   *  Forward Propagation : The input is forward propgated through the network and through time
   *  BackPropagation : Error is backpropagated through network and through time
   */


  for(int j=0; j < numIter; ++j)
  {

    /*
     * Sample the 2 inputs a_int and b_int using C++ stdlib.
     * Calculate True Output c_int = a_int + b_int.
     * Convert from integer to binary using int2bin.
     */

    int a_int = rand() % (largest_number/2);
    vector <int> a = int2bin(a_int);

    int b_int = rand() % (largest_number/2);
    vector <int> b = int2bin(b_int);

    vector <int> c = int2bin(a_int + b_int);


    //Placeholder Vector to store the actual output obtained from RNN
    MatrixXd d = MatrixXd::Zero(binary_dim,1);
    
    //Track the error
    double overallError = 0;


    /*
     * Vector containers holding:
     *	layer_2_deltas : Output error = c - d
     *	layer_1_values : Hidden Layer at each timestep
     */

    vector <MatrixXd> layer_2_deltas;
    vector <MatrixXd> layer_1_values;
    layer_1_values.push_back(MatrixXd::Zero(1,hidden_dim));



  /* 
   * Forward Propagation
   *
   * We iterate through each bit of the input - 
   * So for an 8-bit input, we will do forward propagation 8 times, each time we will feed as input, 1 bit from a_int 
   * and corresponding bit from b_int. Output (a single neuron to approximate a single bit) is the sum.
   *
   * So for this application, number of timesteps = 8.
   */

    for (int position = 0; position < binary_dim; position++)
    {
      //Current Input
      MatrixXd X(1,2);
      X << a[binary_dim - position -1], b[binary_dim - position -1];

      double y = c[binary_dim - position - 1];

      //hidden layer (input ~* prev_hidden)
      MatrixXd layer_1 = X*synapse_0 + layer_1_values.back()*synapse_h;
      layer_1 = layer_1.unaryExpr(&sigmoid);

      //output layer which is a scalar but in MatrixXd format
      MatrixXd layer_2 = layer_1*synapse_1;
      layer_2 = layer_2.unaryExpr(&sigmoid);

      //Measure error between expected and actual 
      double layer_2_error = y - layer_2(0,0);

      //Store Error * output_gradient --> will be used in Backprop
      layer_2_deltas.push_back( layer_2_error * layer_2.unaryExpr(&sigmoidOutputToGradient));

      //Update the Error
      overallError += abs(layer_2_error);

      //Calculate the actual output predicted by RNN
      d(binary_dim-position-1,0) = round(layer_2(0,0));

      //Store Hidden layer to be used in the next timestep
      layer_1_values.push_back(layer_1);
    }



    // Set Hidden layer for 1 future step --> required to initialize the BackProp procedure
    MatrixXd future_layer_1_delta = MatrixXd::Zero(1,hidden_dim);



    /*
     * BackPropagation of gradients through the RNN.
     * Since number of timesteps is 8, the TBPP through time is for 8 timesteps
     *
     * At each step, the input is the corresponding bits from a_int and b_int
     */


    for (int position=0; position < binary_dim; ++position)
    {
      MatrixXd X(1,2);
      X << a[position], b[position];

      //hidden layer that we stored during the feedforward step
      //NOTE: Index changed from reference to match the NumPy notation given in blog
      MatrixXd layer_1 = layer_1_values[binary_dim - position ];

      //previous hidden layer which is just the previous vector element 
      //NOTE: Index changed from reference 
      MatrixXd prev_layer_1 = layer_1_values[binary_dim - position - 1];

      //Error at output layer
      MatrixXd layer_2_delta = layer_2_deltas[binary_dim-position-1];

      //Propagate error to hidden layer from present output and future hidden layer
      MatrixXd layer_1_delta = future_layer_1_delta*synapse_h.transpose() + (layer_2_delta*synapse_1.transpose()).cwiseProduct
			      (layer_1.unaryExpr(&sigmoidOutputToGradient));


      //Calculate the update matrices
      synapse_1_update += layer_1.transpose() * layer_2_delta;
      synapse_h_update += prev_layer_1.transpose() * layer_1_delta;
      synapse_0_update += X.transpose() * layer_1_delta;
    }


    /*
     * Update the Network weights.
     */

    synapse_0 += synapse_0_update * alpha;
    synapse_1 += synapse_1_update * alpha;
    synapse_h += synapse_h_update * alpha;



    /*
     * Set the update matrices to 0 to be used for next training sample
     */

    synapse_0_update *= 0;
    synapse_1_update *= 0;
    synapse_h_update *= 0;


    /*
     * Track Training accuracy every 1000 samples
     */
    if(overallError <= 0.04) {
      if(printProgress) {
	vector <int> dvec(d.data(), d.data() + d.size());
	cout << "Iteration :" << j << endl;
	cout << "Error : " << overallError << endl;
	cout << "Pred  : " << d.transpose() << endl;
	cout << "True  : " << c << endl;
	cout << a_int <<"+"<< b_int <<"="<< bin2int(dvec) << endl;
	cout << "---------------"<< endl;
      }


      return j-1;
    }

    
    if(j % 1000 == 0 && printProgress == true)
    {
      vector <int> dvec(d.data(), d.data() + d.size());
      cout << "Iteration :" << j << endl;
      cout << "Error : " << overallError << endl;
      cout << "Pred  : " << d.transpose() << endl;
      cout << "True  : " << c << endl;
      cout << a_int <<"+"<< b_int <<"="<< bin2int(dvec) << endl;

      cout << "---------------"<< endl;
    }
    
  }

  return numIter;
}
