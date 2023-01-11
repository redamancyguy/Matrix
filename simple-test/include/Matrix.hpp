//
// Created by 孙文礼 on 2023/1/10.
//
#include <cstdlib>
#include <iostream>
#include "MyException.h"
#include "Eigen/Eigen"
#include "Eigen/Core"

#ifndef SIMPLE_TEST_MATRIX_HPP
#define SIMPLE_TEST_MATRIX_HPP

namespace MachineLearning {
    template<typename T, class Allocator = std::allocator<T>>
    class Matrix {
    public:
        enum Axis {
            all,
            row,
            col,
        };
        enum InitType {
            linear_space,
            random,
        };
    private:
        std::uint_fast32_t rows{};
        std::uint_fast32_t cols{};
        T *array{};
        Allocator allocator{};
    public:
        [[nodiscard]] uint_fast32_t getRows() const {
            return rows;
        }

        [[nodiscard]] uint_fast32_t getCols() const {
            return cols;
        }

        Matrix() : rows(0), cols(0) {}

        Matrix(std::uint_fast32_t rows, std::uint_fast32_t cols, T init_value) : rows(rows), cols(cols) {
            array = (T *) std::malloc(sizeof(T) * rows * cols);
            std::fill(array, array + (rows * cols), init_value);
        }

        Matrix(std::uint_fast32_t rows, std::uint_fast32_t cols, InitType init_type = InitType::random,
               T min = 0, T max = 1) : rows(rows), cols(cols) {
            array = (T *) std::malloc(sizeof(T) * rows * cols);
            switch (init_type) {
                case InitType::random: {
                    std::generate(array, array + (rows * cols), std::rand);
                    for (int _ = 0, size = (int) (rows * cols); _ < size; _++) {
                        *(array + _) += min;
                    }
                    for (int _ = 0, size = (int) (rows * cols); _ < size; _++) {
                        *(array + _) *= (max - min);
                    }
                    break;
                }

                case InitType::linear_space:
                    T step = (max - min) / (size() - 1);
                    T value = min - step;
                    for (T *start = array, *end = array + size(); start < end; start++) {
                        *start = (value += step);
                    }
                    break;
            }

        }

        Matrix dot(Matrix<T> matrix) const {
            if (cols != matrix.rows) {
                throw MyException("The matrix dimension does not meet the requirements");
            }
            T (*array1)[cols];
            *((void **) &array1) = ((void *) array);
            T (*array2)[matrix.cols];
            *((void **) &array2) = ((void *) matrix.array);
            Matrix<T> result(rows, matrix.cols, 0);
            T (*array3)[result.cols];
            *((void **) &array3) = ((void *) result.array);
            for (std::uint_fast32_t _0 = 0; _0 < rows; _0++) {
                for (std::uint_fast32_t _1 = 0; _1 < matrix.cols; _1++) {
                    for (std::uint_fast32_t _2 = 0; _2 < cols; _2++) {
                        array3[_0][_1] += array1[_0][_2] * array2[_2][_1];
                    }
                }
            }
            return result;
        }

        friend std::ostream &operator<<(std::ostream &out, const Matrix<T> &matrix) {
            out << "[" << matrix.rows << "," << matrix.cols << "]" << "\n{ ";
            T (*array1)[matrix.cols];
            *((void **) &array1) = ((void *) matrix.array);
            for (int _0 = 0; _0 < matrix.rows; _0++) {
                out << "{ ";
                for (int _1 = 0; _1 < matrix.cols; _1++) {
                    out << array1[_0][_1] << ", ";
                }
                out << "},\n ";
            }
            out << "}";
            return out;
        }

        T &operator()(std::int32_t inRowIndex, std::int32_t inColIndex) noexcept {
            if (inRowIndex < 0) {
                inRowIndex += rows;
            }

            if (inColIndex < 0) {
                inColIndex += cols;
            }
            return array[inRowIndex * cols + inColIndex];
        }

        Matrix(const Matrix<T> &matrix) {
//            std::cout<<"copy "<<std::endl;
            rows = matrix.rows;
            cols = matrix.cols;
            if (array) {
                free(array);
            }
            array = (T *) std::malloc(sizeof(T) * cols * rows);
            if (array == nullptr) {
                std::cout << "" << std::endl;
            }
            std::copy(matrix.array, matrix.array + size(), array);
        }

        Matrix(Matrix<T> &&matrix) noexcept {
//            std::cout<<"move "<<std::endl;
            this->rows = matrix.rows;
            this->cols = matrix.cols;
            if (array) {
                std::free(array);
            }
            this->array = matrix.array;
            matrix.array = nullptr;
        }

        ~Matrix() {
            if (array) {
                std::free(array);
            }
        }

        Matrix sum(Axis axis = Axis::all) {
            T(*array1)[cols];
            *((void **) &array1) = ((void *) array);
            if (axis == Axis::row) {
                Matrix<T> result(1, cols, 0);
                T *array3 = result.array;
                for (int _0 = 0; _0 < rows; _0++) {
                    for (int _1 = 0; _1 < cols; _1++) {
                        array3[_1] += array1[_0][_1];
                    }
                }
                return result;
            } else if (axis == Axis::col) {
                Matrix<T> result(rows, 1, 0);
                T *array3 = result.array;
                for (int _0 = 0; _0 < rows; _0++) {
                    for (int _1 = 0; _1 < cols; _1++) {
                        array3[_0] += array1[_0][_1];
                    }
                }
                return result;
            } else {
                Matrix<T> result(1, 1, 0);
                T *array3 = result.array;
                for (int _0 = 0; _0 < rows; _0++) {
                    for (int _1 = 0; _1 < cols; _1++) {
                        *array3 += array1[_0][_1];
                    }
                }
                return result;
            }
        }

        Matrix operator+(const Matrix<T> &matrix_args) {
            auto matrix = broadcast(matrix_args);
            Matrix<T> result(*this);
            T *array3 = result.array;
            T *array2 = matrix.array;
            for (int _ = 0, size = rows * cols; _ < size; _++) {
                *(array3++) += *(array2++);
            }
            return result;
        }

        Matrix broadcast(const Matrix<T> &matrix) const {
            int row_ = rows / matrix.rows;
            int col_ = cols / matrix.cols;
            if (!(rows == row_ * matrix.rows && cols == col_ * matrix.cols)) {
                throw MyException("The matrix dimension does not meet the requirements");
            }
            Matrix<T> result(rows, cols);
            T(*array1)[cols];
            *((void **) &array1) = ((void *) array);
            T(*array2)[col_];
            *((void **) &array2) = ((void *) matrix.array);
            T(*array3)[cols];
            *((void **) &array3) = ((void *) result.array);
            for (int i = 0; i < matrix.rows; i++) {
                for (int j = 0; j < matrix.cols; j++) {
                    for (int k = 0; k < col_; k++) {
                        array3[i][k * matrix.cols + j] = array2[i][j];
                    }
                }
            }
            for (int i = 1; i < row_; i++) {
                std::copy(result.array, result.array + (matrix.size() * col_),
                          result.array + (i * matrix.size() * col_));
            }
            return result;
        }

        Matrix operator-(const Matrix<T> &matrix_args) {
            auto matrix = broadcast(matrix_args);
            Matrix<T> result(*this);
            T *array3 = result.array;
            T *array2 = matrix.array;
            for (int _ = 0, size = rows * cols; _ < size; _++) {
                *(array3++) -= *(array2++);
            }
            return result;
        }


        Matrix operator*(const Matrix<T> &matrix_args) {
            auto matrix = broadcast(matrix_args);
            Matrix<T> result(*this);
            T *array3 = result.array;
            T *array2 = matrix.array;
            for (int _ = 0, size = rows * cols; _ < size; _++) {
                *(array3++) *= *(array2++);
            }
            return result;
        }

        Matrix operator/(const Matrix<T> &matrix_args) {
            auto matrix = broadcast(matrix_args);
            Matrix<T> result(*this);
            T *array3 = result.array;
            T *array2 = matrix.array;
            for (int _ = 0, size = rows * cols; _ < size; _++) {
                *(array3++) /= *(array2++);
            }
            return result;
        }

        Matrix operator/(const T value) {
            Matrix<T> result(*this);
            T *array3 = result.array;
            for (int _ = 0, size = rows * cols; _ < size; _++) {
                *(array3++) /= value;
            }
            return result;
        }


        Matrix operator*(const T value) {
            Matrix<T> result(*this);
            T *array3 = result.array;
            for (int _ = 0, size = rows * cols; _ < size; _++) {
                *(array3++) *= value;
            }
            return result;
        }


        Matrix operator+(const T value) {
            Matrix<T> result(*this);
            T *array3 = result.array;
            for (int _ = 0, size = rows * cols; _ < size; _++) {
                *(array3++) += value;
            }
            return result;
        }

        Matrix operator-(const T value) {
            Matrix<T> result(*this);
            T *array3 = result.array;
            for (int _ = 0, size = rows * cols; _ < size; _++) {
                *(array3++) -= value;
            }
            return result;
        }

        Matrix h_stack(const Matrix<T> &matrix)const  {
            if (matrix.rows != rows) {
                throw MyException("The matrix dimension does not meet the requirements");
            }
            Matrix<T> result(rows, (cols + matrix.cols));
            T(*array1)[cols];
            *((void **) &array1) = ((void *) array);
            T(*array2)[matrix.cols];
            *((void **) &array2) = ((void *) matrix.array);
            T(*array3)[result.cols];
            *((void **) &array3) = ((void *) result.array);
            for (int _0 = 0; _0 < rows; _0++) {
                for (int _1 = 0; _1 < cols; _1++) {
                    array3[_0][_1] = array1[_0][_1];
                }
            }
            for (int _0 = 0; _0 < rows; _0++) {
                for (int _1 = 0; _1 < matrix.cols; _1++) {
                    array3[_0][_1 + cols] = array2[_0][_1];
                }
            }
            return result;
        }

        [[nodiscard]] std::uint_fast32_t size() const {
            return rows * cols;
        }

        Matrix v_stack(const Matrix<T> &matrix) {
            if (matrix.cols != cols) {
                throw MyException("The matrix dimension does not meet the requirements");
            }
            Matrix<T> result(rows + matrix.rows, cols);
            std::copy(array, array + this->size(), result.array);
            std::copy(matrix.array, matrix.array + matrix.cols * matrix.rows, result.array + size());
            return result;
        }

        Matrix h_split(int col1 = 0, int col2 = 0) {
            if (col1 < 0) {
                col1 += cols;
            }
            if (col2 <= 0) {
                col2 += cols;
            }
            if (col2 < col1) {
                throw MyException("The matrix dimension does not meet the requirements");
            }
            Matrix<T> result(rows, col2 - col1);
            T(*array1)[cols];
            *((void **) &array1) = ((void *) array);
            T(*array3)[result.cols];
            *((void **) &array3) = ((void *) result.array);
            for (int _0 = 0; _0 < rows; _0++) {
                for (int _1 = col1; _1 < col2; _1++) {
                    array3[_0][_1 - col1] = array1[_0][_1];
                }
            }
            return result;
        }

        Matrix v_split(int row1 = 0, int row2 = 0) {
            if (row1 < 0) {
                row1 += rows;
            }
            if (row2 <= 0) {
                row2 += rows;
            }
            if (row2 < row1) {
                throw MyException("The matrix dimension does not meet the requirements");
            }
            Matrix<T> result(row2 - row1, cols);
            std::copy(array + row1 * cols, array + row2 * cols, result.array);
            return result;
        }

        Matrix transpose() const {
            Matrix<T> result(cols, rows);
            T(*array1)[cols];
            *((void **) &array1) = ((void *) array);
            T *array3 = result.array;
            for (int _0 = 0; _0 < cols; _0++) {
                for (int _1 = 0; _1 < rows; _1++) {
                    *(array3++) = array1[_1][_0];
                }
            }
            return result;
        }

        Matrix mean(Axis axis = Axis::all) const {
            T(*array1)[cols];
            *((void **) &array1) = ((void *) array);
            if (axis == Axis::row) {
                Matrix<T> result(1, cols, 0);
                T *array3 = result.array;
                for (int _0 = 0; _0 < rows; _0++) {
                    for (int _1 = 0; _1 < cols; _1++) {
                        array3[_1] += array1[_0][_1];
                    }
                }
                for (auto start = result.array, end = result.array + result.size(); start < end; start++) {
                    *start /= (T) rows;
                }
                return result;
            } else if (axis == Axis::col) {
                Matrix<T> result(rows, 1, 0);
                T *array3 = result.array;
                for (int _0 = 0; _0 < rows; _0++) {
                    for (int _1 = 0; _1 < cols; _1++) {
                        array3[_0] += array1[_0][_1];
                    }
                }
                for (auto start = result.array, end = result.array + result.size(); start < end; start++) {
                    *start /= (T) cols;
                }
                return result;
            } else {
                Matrix<T> result(1, 1, 0);
                T *array3 = result.array;
                for (int _0 = 0; _0 < rows; _0++) {
                    for (int _1 = 0; _1 < cols; _1++) {
                        *array3 += array1[_0][_1];
                    }
                }
                *array3 /= (T) size();
                return result;
            }
        }

        Matrix inverse() {
            if (cols != rows) {
                throw MyException("The matrix dimension does not meet the requirements");
            }
            Matrix<T> result(rows, cols, 0);
            for (int _ = 0; _ < rows; _++) {
                result(_, _) = 1;
            }
            Matrix<T> temp(*this);
            T(*array1)[cols];
            *((void **) &array1) = ((void *) temp.array);
            T(*array3)[result.cols];
            *((void **) &array3) = ((void *) result.array);
            for (int _0 = 0; _0 < cols; _0++) {
                int max_index = _0;
                T max_value = std::numeric_limits<T>::min();
                for (int i = _0; i < rows; i++) {
                    if (max_value < array1[_0][i]) {
                        max_index = i;
                        max_value = array1[_0][i];
                    }
                }
                // find the max one (main element)
                if (max_index != _0) {
                    //exchange tow rows
                    for (int i = 0; i < cols; i++) {
                        auto t_ = array1[_0][i];
                        array1[_0][i] = array1[max_index][i];
                        array1[max_index][i] = t_;
                    }
                    for (int i = 0; i < cols; i++) {
                        auto t_ = array3[_0][i];
                        array3[_0][i] = array3[max_index][i];
                        array3[max_index][i] = t_;
                    }
                }
                T multiple = array1[_0][_0];
                for (int i = 0; i < cols; i++) {
                    array1[_0][i] /= multiple;
                }
                for (int i = 0; i < cols; i++) {
                    array3[_0][i] /= multiple;
                }
                for (int _1 = 0; _1 < rows; _1++) {
                    if (_1 == _0)continue;
                    //倍数
                    multiple = array1[_1][_0] / array1[_0][_0];
                    for (int i = 0; i < cols; i++) {
                        array1[_1][i] -= multiple * array1[_0][i];
                    }
                    for (int i = 0; i < cols; i++) {
                        array3[_1][i] -= multiple * array3[_0][i];
                    }
                }
            }
            return result;
        }

        Matrix inverse_assistant() {
            std::cout<<"not good function"<<std::endl;
            Matrix<T> result_(rows, cols);
            T(*array1)[cols];
            *((void **) &array1) = ((void *) array);
            Eigen::MatrixXd result(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result(i,j) = (decltype(result(i,j)))((*this)(i,j));
                }
            }
            result = result.inverse();
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result_(i,j) =result(i,j);
                }
            }
            return result_;
        }

        Matrix &operator=(const Matrix<T> &matrix) {
            this->rows = matrix.rows;
            this->cols = matrix.cols;
            if (array) {
                std::free(array);
            }
            array = (T *) std::malloc(sizeof(T) * cols * rows);
            std::copy(matrix.array, matrix.array + size(), array);
            return *this;
        }
    };
}
#endif //SIMPLE_TEST_MATRIX_HPP
