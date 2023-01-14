#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include "Matrix.hpp"
#ifndef LinearRegression_HPP
#define LinearRegression_HPP
namespace ML {
    template<typename T>
    class LinearRegression {
    public:
        bool fit_intercept;
        Matrix<T> coefficient;
        Matrix<T> intercept;

        explicit LinearRegression(bool fit_intercept = true) : fit_intercept(fit_intercept) {
        }

        void fit(const Matrix<T> &x, const Matrix<T> &y) {

            if (fit_intercept) {
                auto x_ = Matrix<T>(x.getRows(), 1, 1).h_stack(x);
                coefficient = x_.transpose().dot(x_).inverse().dot(x_.transpose().dot(y));
                intercept = coefficient.v_split(0, y.getCols());
                coefficient = coefficient.v_split(y.getCols(), 0);
            } else {
                coefficient = x.transpose().dot(x).inverse().dot(x.transpose().dot(y));
            }

        }

        Matrix<T> predict(const Matrix<T> &x) const {
            if (fit_intercept) {
                return x.dot(coefficient) + intercept;
            } else {
                return x.dot(coefficient);
            }
        }
    };

}
#endif //LinearRegression_HPP