#include <iostream>
#include <vector>
#include <algorithm>
#include <TimerClock.hpp>
#include "resources/ThreadPool.hpp"
#include "resources/MemoryPool.hpp"
#include "compute/Matrix.hpp"
#include "compute/LineaerRegression.hpp"
#include "DEFINES.h"
#include <iomanip>

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
//    for(double  i = 3;i<100;i+=1){
//        auto qs = DynamicElementWiseSegment<double, double>::sqrt(i);
//        if(qs * qs != i){
//            std::cout <<qs<<" fasle:::  "<<i<<std::endl;
//        }else{
//            std::cout <<qs<<":::  "<<i<<std::endl;
//        }
//    }
//    return 0;
    int data_times = 5;
    auto data_array = dataset_Fixed::get_data<double>("longitudes-200M.bin.data",
                                                      data_times * dynamic_segment_size);
    long loss = 0;
    TimerClock tc;
    long count = 0;
    for (int l = 0; l < 1; l++) {
        std::vector<double> source_data(data_times * dynamic_segment_size);
        for (int i = 0; i < dynamic_segment_size; i++) {
            source_data[i] = data_array[i];
//           source_data[i] = i + 10;
        }
        std::sort(source_data.begin(), source_data.end());
        //SGD
//        DynamicElementWiseSegment<double, double> sg(  0.01);
//        DynamicElementWiseSegment<double, double> sg(0.02);
        //Adam
//        DynamicElementWiseSegment<double, double> sg( 0.1);
//        DynamicElementWiseSegment<double, double> sg(0.0005);
        DynamicElementWiseSegment<double, double> sg(0.00003);
        //loss:879188552 time:2333181  count:0
        //loss:876485628 time:3670126  count:0
        sg.setSize(dynamic_segment_size);
        for (int i = 0; i < dynamic_segment_size; i++) {
            sg.getArray()[i].first = source_data[i];
            sg.getArray()[i].second = i;
        }
//       sg.fit();
//1000
//loss:3809724808 time:3218166  count:0
//loss:3809724808 time:3518125  count:0
//64
//loss:21899222 time:155510  count:0
//loss:21899222 time:128538  count:0
//loss:22200000 time:49092  count:0
//loss:22200000 time:42344  count:0
        sg.fit_std(10);
        for (int j = 0; j < 100000 * 1; j++) {
            for (int i = 0; i < dynamic_segment_size; i++) {
                int pred = sg.predict_std(source_data[i]);
                int ind = sg.linear_search(source_data[i], pred);
//                int ind = sg.exponential_search(source_data[i],pred);
                loss += std::abs(ind - sg.predict_std(source_data[i]));
//                loss += std::abs(ind - 0);
//                sg.SGD_std(source_data[i], ind);
//                sg.SGD_std_adam(source_data[i], ind);
//                sg.SGD_std_v(source_data[i], ind);
//               std::cout<< std::abs(ind - sg.predict(source_data[i])) <<std::endl;
                if (i % 1000 == 0) {
//                   std::cout<<"rescale" <<std::endl;
//                   sg.re_scale(100);
                }
//               usleep(10000);
                if (sg.getArray()[ind].first != source_data[i]) {
                    exit(123);
                }
            }
        }
        count += sg.getSgdCount();
    }
    std::cout << "loss:" << loss << " time:" << tc.get_timer_microSec() << "  count:" << count << std::endl;
}
//std loss:876485628 time:3607241  count:0
//mys loss:876485628 time:10658501  count:0
//
//none sgd  loss:2570000000 time:1911301  count:0
//with sgd  loss: 246190445 time:1752734  count:0
//with sgd  loss: 694440606 time:2037720  count:0
//with adam loss:1062024865 time:3469984  count:0
//with adam loss:1392013219 time:3766918  count:0 //
//with adam loss:1280015033 time:3735408  count:0
//with adam loss: 954162473 time:3597207  count:0
//with adam loss: 916343617 time:3575915  count:0  0.001



