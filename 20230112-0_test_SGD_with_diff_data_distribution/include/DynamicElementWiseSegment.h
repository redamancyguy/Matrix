//
// Created by 孙文礼 on 2023/1/11.
//
#include <DEFINES.h>
#include <cstdlib>
#include <iostream>
#include <random>

#ifndef SIMPLE_TEST_SEGMENTELEMENTWISE_H
#define SIMPLE_TEST_SEGMENTELEMENTWISE_H

template<class key_type, class value_type>
class DynamicElementWiseSegment {
private:
public:
    double slope;
    double intercept;
    double x_mean;
    double y_mean;
    std::int32_t _size;
    std::pair<key_type, value_type> array[dynamic_segment_size];

    //insert_raw
    void Add(const std::int32_t position, const std::pair<key_type, const value_type> &pair) {
        for (std::int32_t i = _size; i > position; i--) {
            array[i] = array[i - 1];
        }
        array[position] = pair;
        _size++;
    }

    //delete_raw
    void Erase(const std::int32_t position) {
        for (std::int32_t i = position + 1; i < _size; i--) {
            array[i - 1] = array[i];
        }
        _size--;
    }

    //update_raw
    void Set(const std::int32_t position, const std::pair<key_type, const value_type> &pair) {
        array[position] = pair;
    }

    //select_raw
    std::pair<key_type, value_type> Get(const std::int32_t position) {
        return array[position];
    }

public:
//    double SGD_COUNT = 100;
//    double SGD_COUNT = 250;
//    double SGD_COUNT = 350;
//    double SGD_COUNT = 450;
    double SGD_COUNT = 1800;
//    double SGD_COUNT = 1000;
    void SGD(const key_type key, const std::int32_t position) {
//        1/size() learning rate
        double residual_error = key * slope + intercept - position;
        double dl_ds = key * residual_error *1/SGD_COUNT;
        double dl_di = residual_error* 1/SGD_COUNT;
        slope -= (1 / (double)size()) * dl_ds;
        intercept -= (1 / (double)size()) * dl_di;
        SGD_COUNT += 0.1;
    }

    void SGD_(const key_type key, const std::int32_t position) {
//        1/size() learning rate
        double residual_error = key * slope + intercept - position;
        double dl_ds = key * residual_error *1/SGD_COUNT;
        double dl_di = residual_error* 1/SGD_COUNT;
//        std::cout<<slope<<" : "<<intercept;
//        slope -= (1 / (double)size()) * dl_ds;
//        intercept -= (1 / (double)size()) * dl_di;
//        std::cout<<"======="<<slope<<" : "<<intercept<<std::endl;
//        usleep(1000);
        SGD_COUNT += dl_di;
        SGD_COUNT += dl_ds;
    }

    double getSgdCount() const {
        return SGD_COUNT;
    }

    std::pair<key_type, value_type> *getArray() {
        return array;
    }

    void setSize(int32_t size) {
        _size = size;
    }

    DynamicElementWiseSegment() : _size(0) {}

    [[nodiscard]] std::int32_t size() const {
        return _size;
    }

    void fit() {
        y_mean = size() / 2;
        x_mean = 0;
        double lxx = 0;
        double lxy = 0;
        for (int i = 0; i < size(); i++) {
            x_mean += array[i].first;
        }
        x_mean /= _size;
        for (int i = 0; i < size(); i++) {
            lxx += (x_mean - array[i].first) * (x_mean - array[i].first);
        }
        for (int i = 0; i < size(); i++) {
            lxy += (x_mean - array[i].first) * (y_mean - i);
        }
        slope = lxy / lxx;
        intercept = y_mean - x_mean * slope;
    }

    void fit_sample(int sample_number = 10) {
        y_mean = size() / 2;
        x_mean = 0;
        double lxx = 0;
        double lxy = 0;
        std::default_random_engine e;
        for (int i = 0; i < sample_number; i++) {
            x_mean += array[e() & _size].first;
        }
        x_mean /= sample_number;
        for (int i = 0; i < sample_number; i++) {
            int random_index = e() & _size;
            lxx += (x_mean - array[random_index].first) * (x_mean - array[random_index].first);
            lxy += (x_mean - array[random_index].first) * (y_mean - random_index);
        }
        slope = lxy / lxx;
        intercept = y_mean - x_mean * slope;
    }

    std::int32_t predict(const key_type key) {
        std::int32_t result;
        result = slope * key + intercept;
        if (result <= 0) {
            return 0;
        }
        if (result >= size() - 1) {
            return size() - 1;
        }
        return result;
    }

    std::int32_t exponential_search(const key_type key, const std::int32_t position_rough) {
        std::int32_t left = 1;
        std::int32_t right = 1;
        if (array[position_rough].first == key) {
            return position_rough;
        }
        if (array[position_rough].first < key) {
            do {
                left = right;
                right = right << 1;
                if (position_rough + right >= _size - 1) {
                    right = _size - position_rough - 1;
                    break;
                }
            } while (array[position_rough + right].first < key);
            left = position_rough + left;
            right = position_rough + right;
        } else {
            do {
                right = left;
                left = left << 1;
                if (position_rough - left <= 0) {
                    left = position_rough;
                    break;
                }
            } while (array[position_rough - left].first > key);
            left = position_rough - left;
            right = position_rough - right;
        }
        std::int32_t middle = (left + right) / 2;
        while (array[middle].first != key && left < right) {
            array[middle].first < key ? left = middle + 1 : right = middle - 1;
            middle = (left + right) / 2;
        }
        return middle;
    }

    std::int32_t linear_search(const key_type key, std::int32_t position_rough) {
        if (array[position_rough].first == key) {
            return position_rough;
        }
        if (array[position_rough].first < key && position_rough < _size) {
            while (array[position_rough].first < key) { position_rough++; }
            return position_rough;
        } else {
            while (array[position_rough].first > key && position_rough >= 0) { position_rough--; }
            return position_rough;
        }
    }

    inline bool Is_Full() {
        return _size == dynamic_segment_size;
    }

    friend std::ostream &operator<<(std::ostream &out, DynamicElementWiseSegment<key_type, value_type> &matrix) {
        out << "[";
        for (size_t i = 0; i < matrix.Size(); i++) {
            out << "{" << matrix.array[i].first << "," << matrix.array[i].second << "}, ";
        }
        out << "]" << std::endl;
        return out;
    }
};

#endif //SIMPLE_TEST_SEGMENTELEMENTWISE_H
