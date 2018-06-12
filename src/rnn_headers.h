#ifndef HEADER_H
#define HEADER_H

#include <../Eigen/Core>
#include <../Eigen/Dense>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
#include <thread>
#include <future>

using namespace Eigen;
using namespace std;

//Number of Timesteps in RNN 
const int binary_dim = 8;

int rnnStep(double alpha, unsigned numIter, bool printProgress);
double sigmoid(double x);
double sigmoidOutputToGradient(double output);
int bin2int( vector<int> num);
vector <int> int2bin(unsigned int num);



template<class T>
ostream& operator<<(ostream& stream, const std::vector<T>& values)
{
    /*
     * Overload the ostream operator to make printing vectors easy
     */

    stream << "[";
    for (int i = 0; i < values.size(); i++)
    {
      if (i > 0)
	stream << ',';
      stream << values[i];
    }
    stream << ']';
    return stream;
}

template<typename T1, typename T2>
vector <T2> operator-(vector <T1>& m1, vector <T2>& m2)
{
  /*
   * Overload the '-' operator for easy vector addition
   *
   * Arguments:
   *  m1 - vector of type T1
   *  m2 - vector of type T2
   */



  const unsigned long vsize = m1.size();
  vector <T2> diff (vsize);

  for (unsigned i = 0; i != vsize; ++i)
  {
    diff[i] = m1[i] - m2[i];
  }
  return diff;
}

template<typename T1, typename T2>
vector <T2> operator+(vector <T1>& m1, vector <T2>& m2)
{
  /*
   * Overload the '+' operator for easy vector addition
   *
   * Arguments:
   *  m1 - vector of type T1
   *  m2 - vector of type T2
   */



  const unsigned long vsize = m1.size();
  vector <T2> sum (vsize);

  for (unsigned i = 0; i != vsize; ++i)
  {
    sum[i] = m1[i] + m2[i];
  }
  return sum;
}

#endif
