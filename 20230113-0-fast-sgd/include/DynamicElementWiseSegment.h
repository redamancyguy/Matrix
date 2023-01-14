//
// Created by 孙文礼 on 2023/1/11.
//
#include <DEFINES.h>
#include <cstdlib>
#include <iostream>
#include <random>
#include <resources/MemoryPool.hpp>

#ifndef SIMPLE_TEST_SEGMENTELEMENTWISE_H
#define SIMPLE_TEST_SEGMENTELEMENTWISE_H


template<class key_type, class value_type>
class DynamicElementWiseSegment {
public:
    enum SGD_TYPE {
        NONE = 0,
        SGD = 1,
        SGD_MOMENTUM = 2,
        SGD_ADAM = 3,
    };
private:
    static std::default_random_engine random_engine;
//    static MemoryPool<>;
public:
    double slope{};
    double intercept{};
    double x_mean{};
    double y_mean{};
    double x_std_dev{};
    double y_std_dev{};
//    double learning_rate = 250;
    double learning_rate = 1e-3;
    double m_slope = 0;
    double m_intercept = 0;;
    double v_slope = 0;
    double v_intercept = 0;
    double beta_m = 0.9;
    double beta_v = 0.999;
//    double m_intercept = 0;
    double dads = 0;
    std::int32_t _size;
    std::int32_t _capacity;
    std::pair<key_type, value_type> *array;

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
    std::int_fast32_t get_position(const key_type key, SGD_TYPE how_to_SGD = NONE) {
//    std::int_fast32_t get_position(const key_type key, SGD_TYPE how_to_SGD = NONE) {
        double key_ = key;
        double prediction;
        key_ = (key_ - x_mean) / x_std_dev;
        prediction = slope * key_ + intercept;
        int position = (int) (prediction * y_std_dev + y_mean + 0.5);
        if (position <= 0) {
            position = 0;
        } else if (position >= size() - 1) {
            position = size() - 1;
        }

        position = exponential_search(key, position);
//        position = linear_search(key, position);
        switch (how_to_SGD) {
            case NONE: {
                break;
            }
            case SGD: {
                double residual_error = prediction - (((double) position) - y_mean) / y_std_dev;
                double dl_ds = key * residual_error * learning_rate;
                double dl_di = residual_error * learning_rate;
                slope -= dl_ds;
                intercept -= dl_di;
                break;
            }
            case SGD_MOMENTUM: {
                break;
            }
            case SGD_ADAM: {
                break;
            }
        }
        return position;
    }


    void SGD_std(key_type key, std::int32_t position) {
        double std_x = ((double) key - x_mean) / x_std_dev;
        double std_y = ((double) position - y_mean) / y_std_dev;
        double residual_error = std_x * slope + intercept - std_y;
        double dl_ds = key * residual_error * learning_rate;
        double dl_di = residual_error * learning_rate;
        slope -= dl_ds;
        intercept -= dl_di;
    }

    void SGD_std_v(key_type key, std::int32_t position) {
        double std_x = ((double) key - x_mean) / x_std_dev;
        double std_y = ((double) position - y_mean) / y_std_dev;
        double residual_error = std_x * slope + intercept - std_y;
        double dl_ds = key * residual_error;
        double dl_di = residual_error;
        m_slope = beta_m * m_slope + (1 - beta_m) * dl_ds;
        m_intercept = beta_m * m_intercept + (1 - beta_m) * dl_di;
        slope -= learning_rate * m_slope;
        intercept -= learning_rate * m_intercept;
    }

    void SGD_std_adam(key_type key, std::int32_t position) {
        double std_x = ((double) key - x_mean) / x_std_dev;
        double std_y = ((double) position - y_mean) / y_std_dev;
        double residual_error = std_x * slope + intercept - std_y;
        double dl_ds = key * residual_error;
        double dl_di = residual_error;
        m_slope = beta_m * m_slope + (1 - beta_m) * dl_ds;
        m_intercept = beta_m * m_intercept + (1 - beta_m) * dl_di;
        v_slope = beta_v * v_slope + (1 - beta_v) * dl_ds * dl_ds;
        v_intercept = beta_v * v_intercept + (1 - beta_v) * dl_di * dl_di;
        slope -= learning_rate * ((m_slope / (1 - beta_m)) / (1e-8 + std::sqrt(v_slope / (1 - beta_v))));
        intercept -= learning_rate * ((m_intercept / (1 - beta_m)) / (1e-8 + std::sqrt(v_intercept / (1 - beta_v))));
    }

    double getSgdCount() const {
        return learning_rate;
    }

    std::pair<key_type, value_type> *getArray() {
        return array;
    }

    void setSize(int32_t size) {
        _size = size;
        array = (std::pair<key_type, value_type> *) std::realloc(array, sizeof(std::pair<key_type, value_type>) * size);
    }

    ~DynamicElementWiseSegment() {
        std::free(array);
    }

    DynamicElementWiseSegment(double init_learning_rate = 1e-3) :
            learning_rate(init_learning_rate),
            _size(0), _capacity(default_segment_size) {
        array = (std::pair<key_type, value_type> *) std::malloc(sizeof(std::pair<key_type, value_type>) * _capacity);
    }

    [[nodiscard]] std::int32_t size() const {
        return _size;
    }

    void fit(int sample_number = 10) {
        y_mean = 0;
        x_mean = 0;
        x_std_dev = 0;
        y_std_dev = 0;
        double lxx = 0;
        double lxy = 0;
        double lyy = 0;
        double tempe_x_array[sample_number];
        double tempe_y_array[sample_number];
        for (int i = 0; i < sample_number; i++) {
            int random_index = random_engine() % _size;
            tempe_x_array[i] = array[random_index].first;
            tempe_y_array[i] = random_index;
            x_mean += array[random_index].first;
            y_mean += random_index;
            x_std_dev += array[random_index].first * array[random_index].first;
            y_std_dev += random_index * random_index;
        }
        x_mean /= sample_number;
        y_mean /= sample_number;
        x_std_dev /= sample_number;
        y_std_dev /= sample_number;
        x_std_dev -= x_mean * x_mean;
        y_std_dev -= y_mean * y_mean;
        x_std_dev = std::sqrt(x_std_dev);
        y_std_dev = std::sqrt(y_std_dev);
        for (int i = 0; i < sample_number; i++) {
            lxx += (x_mean - tempe_x_array[i]) * (x_mean - tempe_x_array[i]);
            lxy += (x_mean - tempe_x_array[i]) * (y_mean - tempe_y_array[i]);
            lyy += (y_mean - tempe_y_array[i]) * (y_mean - tempe_y_array[i]);
        }
        slope = lxy / lxx;
        intercept = y_mean - x_mean * slope;
    }
//    void fit() {
//        y_mean = size() / 2;
//        x_mean = 0;
//        double lxx = 0;
//        double lxy = 0;
//        for (int i = 0; i < size(); i++) {
//            x_mean += array[i].first;
//        }
//        x_mean /= _size;
//        for (int i = 0; i < size(); i++) {
//            lxx += (x_mean - array[i].first) * (x_mean - array[i].first);
//        }
//        for (int i = 0; i < size(); i++) {
//            lxy += (x_mean - array[i].first) * (y_mean - i);
//        }
//        slope = lxy / lxx;
//        intercept = y_mean - x_mean * slope;
//        x_std_dev = lxx / size();
//        y_std_dev = (double) (size() * size()) / ((double) 12);
//        x_std_dev = std::sqrt(x_std_dev);
//        y_std_dev = std::sqrt(y_std_dev);
//    }

    void re_scale(int sample_number = 10) {
        learning_rate = 1000;
        intercept = intercept * y_std_dev + y_mean - slope * x_mean * y_std_dev / x_std_dev;
        slope = slope / x_std_dev * y_std_dev;
        //
        y_mean = 0;
        x_mean = 0;
        x_std_dev = 0;
        y_std_dev = 0;
        double tempe_x_array[sample_number];
        double tempe_y_array[sample_number];
        for (int i = 0; i < sample_number; i++) {
            int random_index = random_engine() % _size;
            tempe_x_array[i] = array[random_index].first;
            tempe_y_array[i] = random_index;
            x_mean += array[random_index].first;
            y_mean += random_index;
            x_std_dev += array[random_index].first * array[random_index].first;
            y_std_dev += random_index * random_index;
        }
        x_mean /= sample_number;
        y_mean /= sample_number;
        x_std_dev /= sample_number;
        y_std_dev /= sample_number;
        x_std_dev -= x_mean * x_mean;
        y_std_dev -= y_mean * y_mean;
        x_std_dev = std::sqrt(x_std_dev);
        y_std_dev = std::sqrt(y_std_dev);
        //
        intercept = (intercept + x_mean * slope - y_mean) / y_std_dev;
        slope = slope * x_std_dev / y_std_dev;
    }

    void fit_std(int sample_number = 10) {
        y_mean = 0;
        x_mean = 0;
        x_std_dev = 0;
        y_std_dev = 0;
        double lxx = 0;
        double lxy = 0;
        double lyy = 0;
        double tempe_x_array[sample_number];
        double tempe_y_array[sample_number];
        for (int i = 0; i < sample_number; i++) {
            int random_index = random_engine() % _size;
            tempe_x_array[i] = array[random_index].first;
            tempe_y_array[i] = random_index;
            x_mean += array[random_index].first;
            y_mean += random_index;
            x_std_dev += array[random_index].first * array[random_index].first;
            y_std_dev += random_index * random_index;
        }
        x_mean /= sample_number;
        y_mean /= sample_number;
        x_std_dev /= sample_number;
        y_std_dev /= sample_number;
        x_std_dev -= x_mean * x_mean;
        y_std_dev -= y_mean * y_mean;
        x_std_dev = std::sqrt(x_std_dev);
        y_std_dev = std::sqrt(y_std_dev);
        for (int i = 0; i < sample_number; i++) {
            lxx += (x_mean - tempe_x_array[i]) * (x_mean - tempe_x_array[i]);
            lxy += (x_mean - tempe_x_array[i]) * (y_mean - tempe_y_array[i]);
            lyy += (y_mean - tempe_y_array[i]) * (y_mean - tempe_y_array[i]);
        }
        slope = lxy / lxx;
        intercept = y_mean - x_mean * slope;
        intercept = (intercept + x_mean * slope - y_mean) / y_std_dev;
        slope = slope * x_std_dev / y_std_dev;
    }

    std::int32_t predict_std(const key_type key) {
        double key_ = key;
        double result;
        key_ = (key_ - x_mean) / x_std_dev;
        result = slope * key_ + intercept;
        result = result * y_std_dev + y_mean + 0.5;
        if (result <= 0) {
            return 0;
        }
        if (result >= size() - 1) {
            return size() - 1;
        }
        return (std::int32_t) result;
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
        if (array[position_rough].first == key) {
            return position_rough;
        }
        std::int32_t left = 1;
        std::int32_t right = 1;
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

template<typename key_type, typename value_type>
std::default_random_engine DynamicElementWiseSegment<key_type, value_type>::random_engine =
        std::default_random_engine(123);
#endif //SIMPLE_TEST_SEGMENTELEMENTWISE_H
