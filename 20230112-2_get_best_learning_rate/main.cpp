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
    auto data_array = dataset_Fixed::get_data<double>("longitudes-200M.bin.data", dynamic_segment_size);
    for (double init_learning_factor = 100; init_learning_factor < 1000; init_learning_factor += 30) {
        for (double decrease = 1; decrease > 1e-10; decrease /= 3) {
            long loss = 0;
            TimerClock tc;
            long count = 0;
            for (int l = 0; l < 1; l++) {
                std::vector<double> source_data(dynamic_segment_size);
                for (int i = 0; i < dynamic_segment_size; i++) {
                    source_data[i] = data_array[i];
//           source_data[i] = i + 10;
                }
                std::sort(source_data.begin(), source_data.end());
                DynamicElementWiseSegment<double, double> sg(init_learning_factor, decrease);
                sg.setSize(dynamic_segment_size);
                for (int i = 0; i < dynamic_segment_size; i++) {
                    sg.getArray()[i].first = source_data[i];
                    sg.getArray()[i].second = i;
                }
//       sg.fit();
                sg.fit_std(10);
                for (int j = 0; j < 10000 * 20; j++) {
                    for (int i = 0; i < dynamic_segment_size / 20; i++) {
//           for (int i = 0; i < dynamic_segment_size; i++) {
//               int ind = sg.exponential_search(source_data[i], sg.predict(source_data[i]));
//               loss += std::abs(ind - sg.predict(source_data[i]));
                        int ind = sg.exponential_search(source_data[i], sg.predict_std(source_data[i]));
                        loss += std::abs(ind - sg.predict_std(source_data[i]));
                        sg.SGD_std(source_data[i], ind);
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
            std::cout << "init_learning_factor : " << init_learning_factor <<
                      " decrease:" <<
                      std::setw(23) << decrease <<
                      std::setw(15) << loss << " : " <<
                      std::setw(10) <<tc.get_timer_microSec() << "  : " <<
                      std::setw(10) << count
                      << std::endl;
        }
    }
}

