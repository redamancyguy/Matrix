//
// Created by 孙文礼 on 2023/1/10.
//
#include <iostream>
#ifndef SIMPLE_TEST_MYEXCEPTION_H
#define SIMPLE_TEST_MYEXCEPTION_H
class MyException : public std::exception
{
public:
    MyException() : message("Error."){}
    explicit MyException(const std::string& str) : message("Error : " + str) {}
    ~MyException() noexcept override = default;

    [[nodiscard]] const char* what() const noexcept override {
        return message.c_str();
    }

private:
    std::string message;
};
#endif //SIMPLE_TEST_MYEXCEPTION_H
