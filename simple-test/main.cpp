#include <iostream>
#include <vector>
#include <algorithm>
#include <TimerClock.hpp>
#include <ThreadPool.hpp>
#include <MemoryPool.hpp>
#include <Matrix.hpp>
#include <LineaerRegression.hpp>


void ML_TEST(){
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
}

template<class T>
class Vector {
public:
    T *array;
    std::size_t size;

    Vector() {
        size = 0;
        array = (T*)std::malloc(sizeof(T) * 64);
    }

    ~Vector() {
        std::free(array);
    }

    void Insert(std::size_t position, const T &element) {
        for (size_t i = size; i > position; i--) {
            array[i] = array[i-1];
        }
        array[position] = element;
        size++;
    }

    friend std::ostream &operator<<(std::ostream &out, const Vector<T> &matrix) {
        out << "[" ;
//        for(size_t i = 0;i<size;i++){
//
//        }
        out << "]" <<std::endl;
        return out;
    }
};

int main() {
    Vector<double> dv;

    for(int i = 0;i<60;i++){
        dv.Insert(0,std::rand());
    }


}
