#include <iostream>
#include <vector>
#include <algorithm>
#include <TimerClock.hpp>
#include "resources/ThreadPool.hpp"
#include "resources/MemoryPool.hpp"
#include "compute/Matrix.hpp"
#include "compute/LineaerRegression.hpp"
#include "DEFINES.h"

void ML_TEST() {
    ML::LinearRegression<double> lng(true);
//    ML::LinearRegression<double> lng(false);
    ML::Matrix<double> x(10, 1, ML::Matrix<double>::InitType::linear_space, 0, 99);
    ML::Matrix<double> y = x * x + x * 300 + 100;
    std::cout << x.h_stack(y) << std::endl;
    ML::Matrix<double> poly_x = x.h_stack(x * x);
    lng.fit(poly_x, y);
    std::cout << lng.coefficient << std::endl;
    std::cout << lng.intercept << std::endl;

    ML::Matrix<double> aa(2, 1);
    aa(0, 0) = 1;
    aa(1, 0) = 2;

    std::cout << lng.predict(poly_x).h_stack(y) << std::endl;
}


#include <SegmentVectorWise.hpp>
#include <DynamicElementWiseSegment.h>
#include "dataset.hpp"

int main() {
//    std::default_random_engine::seed<int>(123);
    auto data_array = dataset_Fixed::get_data<double>("longitudes-200M.bin.data", dynamic_segment_size);
    long loss = 0;
    std::default_random_engine e;
    TimerClock tc;
   for(int l = 0;l<1;l++){
       std::vector<double> source_data(dynamic_segment_size);
       for (int i = 0; i < dynamic_segment_size; i++) {
           source_data[i] = data_array[i];
       }
       std::sort(source_data.begin(), source_data.end());
       DynamicElementWiseSegment<double, double> sg;
       sg.setSize(dynamic_segment_size);
       for (int i = 0; i < dynamic_segment_size; i++) {
           sg.getArray()[i].first = source_data[i];
           sg.getArray()[i].second = i;
       }
//       sg.fit();
       sg.fit_sample(10);
       for (int j = 0; j < 10000000; j++) {
           for (int i = 0; i < 2; i++) {
               int ind = sg.exponential_search(source_data[i], sg.predict(source_data[i]));
               loss += std::abs(ind - sg.predict(source_data[i]));
               sg.SGD(source_data[i],ind);

               if (sg.getArray()[ind].first != source_data[i]) {
                   exit(123);
               }
           }
       }
   }
    std::cout << loss << " : " << tc.get_timer_microSec() << std::endl;
}

//time     10000000 : 33249
//time_SGD 10000000 : 343609

//vector1 15015970 loss 11298330000
//vector2 16501266 loss 9241350000
//vector3 18881072 loss 9293300000
//vector6 26440201 loss 7489160000
//element 2387799 loss 11297900000


//loss 11297900000
//loss 13556310000
//loss 83865600000