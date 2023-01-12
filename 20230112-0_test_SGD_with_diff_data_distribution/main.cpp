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
    long count = 0;
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
       for (int j = 0; j < 10000*2; j++) {
           for (int i = 0; i < dynamic_segment_size/2; i++) {
//           for (int i = 0; i < dynamic_segment_size; i++) {
               int ind = sg.exponential_search(source_data[i], sg.predict(source_data[i]));
               loss += std::abs(ind - sg.predict(source_data[i]));
               sg.SGD(source_data[i],ind);
//               sg.SGD_(source_data[i],ind);

               if (sg.getArray()[ind].first != source_data[i]) {
                   exit(123);
               }
           }
       }
       count += sg.getSgdCount();
   }
    std::cout << loss << " : " << tc.get_timer_microSec() <<"  : "<<count<< std::endl;
}

//all data
//time NO sgd         159708280000 : 2437190  : -824730
//time_SGD  1/1900    157463421722 : 5620739  : 8193899
//time_SGD  1/1900    157463421722 : 5620739  : 8193899


// 1/2 data
//time NO sgd         41126300000 : 1038360  : 2627466
//time_SGD  1/1900    26046024620 : 2468077  : 4097800


//SGD
//20 data

//time rate = 1/100   1587986924 : 1350738  : 4000100
//time rate = 1/250    971997574 : 1167357  : 4000250
//time rate = 1/350    758028460 : 1212576  : 4000350
//time rate = 1/450    609996890 : 1147144  : 4000450
//time rate = 1/550    501989658 : 1101732  : 4000550
//time rate = 1/950    254000891 : 1041875  : 4000950
//time rate = 1/1550   66004656 : 791703  : 4001550
//time rate = 1/1650   48000653 : 784675  : 4001650
//time rate = 1/1750   34006692 : 757566  : 4001750
//time rate = 1/1850   26007523 : 777029  : 4001850
//time rate = 1/1900   24007975 : 792604  : 4001900 //Optimal
//time no SGD         954000000 : 291617  : 137174
//time rate = 1/1950   26008177 : 787894  : 4001950
//time rate = 1/2050   30008720 : 742432  : 4002050
