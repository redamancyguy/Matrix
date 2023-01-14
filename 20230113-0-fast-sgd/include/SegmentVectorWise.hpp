//
// Created by 孙文礼 on 2023/1/11.
//
#include <DEFINES.h>
#include <cstdlib>
#include <iostream>
#include <compute/Matrix.hpp>
#include <compute/LineaerRegression.hpp>
#ifndef SIMPLE_TEST_SEGMENTVECTORWISE_H
#define SIMPLE_TEST_SEGMENTVECTORWISE_H
template<class key_type, class value_type>
class SegmentVectorWise {
private:
    int degree{};
    ML::LinearRegression<key_type> lng;
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
    std::pair<key_type, value_type> Get(const std::int32_t position) {
        return array[position];
    }

public:

    std::pair<key_type, value_type> *getArray() {
        return array;
    }

    void setSize(int32_t size) {
        _size = size;
    }

    SegmentVectorWise() : _size(0) { lng = ML::LinearRegression<key_type>(true); _capacity = dynamic_segment_size;  }

    [[nodiscard]] std::int32_t size() const {
        return _size;
    }

    void fit(int degree_input = 1) {
        ML::Matrix<key_type> x(size(), 1);
        for (int j = 0; j < size(); j++) {
            x(j, 0) = (double) array[j].first;
        }
        this->degree = degree_input;
        auto x_poly = x.UnivariatePolynomial( degree_input);
        auto y = ML::linear_space<double>(0, size() - 1, size());
        lng.fit(x_poly, y);
    }

    std::int32_t predict(const key_type key) {
        std::int32_t result;
        ML::Matrix<key_type> x(1,1);
        x(0,0) = (key_type)key;
        result = lng.predict(x.UnivariatePolynomial(degree))(0,0);
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

    friend std::ostream &operator<<(std::ostream &out, SegmentVectorWise<key_type, value_type> &matrix) {
        out << "[";
        for (size_t i = 0; i < matrix.Size(); i++) {
            out << "{" << matrix.array[i].first << "," << matrix.array[i].second << "}, ";
        }
        out << "]" << std::endl;
        return out;
    }
};
#endif //SIMPLE_TEST_SEGMENTVECTORWISE_H
