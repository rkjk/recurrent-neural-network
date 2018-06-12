#include "rnn_headers.h"

double sigmoid(double x)
{
  /*
   * Return Sigmoid of given double where Sigmoid function is given as:
   * Sigmoid(x) = 1/(1 + exp(-x))
   */


  return 1.0/(1.0 + exp(-1*x));

}

double sigmoidOutputToGradient(double output)
{
  
  /*
   * Return Derivative of the Sigmoid Function evaluated at "output"
   *
   */


  return output * (1 - output);

}


vector <int> int2bin(unsigned int num)
{
  /* 
   * Convert unsigned integer to binary
   *
   * Arguments:
   *  num - number to be converted
   *
   * Return:
   *  vector of size(binary_dim) with zeroth element being MSB.
   */


  vector <int> temp(binary_dim);
  fill(temp.begin(), temp.end(), 0);

  for(int i=0;i < binary_dim; i++)
  {
    temp[i] = (num%2==0 ? 0: 1);
    num /= 2;
    if(num==0) break;
  }
  reverse(temp.begin(), temp.end());
  return temp;
}

int bin2int( vector<int> num)
{
  /* 
   * Convert Binary number to unsigned integer
   */

  int size = num.size()-1;
  reverse(num.begin(), num.end());
  int sum=0;
  for (int i=size;i>=0; --i)
  {
    sum += num[i] * pow(2,i);
  }

  return sum;
}
