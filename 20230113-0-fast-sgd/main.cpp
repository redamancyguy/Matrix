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


void print_bit(void *source) {
    for (int i = 0; i < 8; i++) {
        char *p = (char *) source;
        p += (7 - i);
        unsigned char pp = 128;
        for (int j = 7; j > -1; j--) {
            ::printf("%d", (pp & *p) != 0);
            pp = pp >> 1;
        }
    }
    puts("");
}

int main() {
    int data_times = 5;
    auto data_array = dataset_Fixed::get_data<double>("longitudes-200M.bin.data",
                                                      data_times * dynamic_segment_size);
    double loss = 0;
    TimerClock tc;
    long count = 0;
    std::vector<double> source_data(data_times * dynamic_segment_size);
    for (int i = 0; i < data_times * dynamic_segment_size; i++) {
        source_data[i] = data_array[i];
//            source_data[i] = i + 10;
    }
    std::sort(source_data.begin(), source_data.end());
    //SGD
//        DynamicElementWiseSegment<double, double> sg(  0.01);
//        DynamicElementWiseSegment<double, double> sg(0.02);
    //Adam
//        DynamicElementWiseSegment<double, double> sg( 0.1);
//        DynamicElementWiseSegment<double, double> sg(0.0005);
    using key_type = double;
    using value_type = double;
    DynamicElementWiseSegment<key_type, value_type> sg(0.001);
    //loss:879188552 time:2333181  count:0
    //loss:876485628 time:3670126  count:0
    sg.setSize(dynamic_segment_size);
    for (int i = 0; i < dynamic_segment_size; i++) {
        sg.getArray()[i].first = source_data[i];
        sg.getArray()[i].second = i;
    }

    sg.fit_std(10);//loss:0 time:328230  count:0
//        sg.fit(10);                //loss:0 time:111167  count:0
    double cycles = 1000000000;
    int query_selective = 50;
    for (int j = 0; j < cycles; j++) {
        for (int i = 0; i < dynamic_segment_size / query_selective; i++) {
            int pred = sg.predict_std(source_data[i]);
//            int ind = sg.get_position(source_data[i], DynamicElementWiseSegment<key_type, value_type>::SGD_TYPE::SGD);
            int ind = sg.get_position(source_data[i], DynamicElementWiseSegment<double, double>::SGD_TYPE::NONE);
            loss += std::abs(pred - ind);
        }
    }
    std::cout << " time:"
              << (double) tc.get_timer_microSec() * 1e3 / ((cycles * dynamic_segment_size / query_selective))
              << "     loss:" << loss / (cycles * dynamic_segment_size) << std::endl;
}
//loss:0 time:252361  count:0
//loss:0 time:261279  count:0



