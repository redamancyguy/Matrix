//#include <iostream>
//#include <vector>
//#include <algorithm>
//#include "include/TimerClock.hpp"
//#include "include/ThreadPool.hpp"
//#include <NumCpp.hpp>
//#include <MemoryPool.hpp>
//
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

//using namespace std;
//using namespace Eigen;
//namespace Eigen{
//    Eigen::Matrix2d cross(Eigen::Matrix2d a,Eigen::Matrix2d b){
//        Eigen::Matrix2d result(a);
//        auto aa = b.begin();
//        return result;
//    }
//}
#include "Matrix.hpp"
#include "LineaerRegression.hpp"

int main() {
    MachineLearning::LinearRegression<double> lng(true);
//    MachineLearning::LinearRegression<double> lng(false);
    MachineLearning::Matrix<double> x(10, 1, MachineLearning::Matrix<double>::InitType::linear_space, 0, 99);
    MachineLearning::Matrix<double> y = x * x + x * 300 + 100;
    std::cout << x.h_stack(y) << std::endl;
    MachineLearning::Matrix<double> poly_x = x.h_stack(x * x);
    lng.fit(poly_x, y);
    std::cout << lng.coefficient << std::endl;
    std::cout << lng.intercept << std::endl;

    MachineLearning::Matrix<double> aa(2, 1);
    aa(0, 0) = 1;
    aa(1, 0) = 2;

    std::cout << lng.predict(poly_x).h_stack(y) << std::endl;

//    std::cout<<x.broadcast(aa)<<std::endl;
//    std::cout<<y<<std::endl;


}
