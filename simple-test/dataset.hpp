//
// Created by 孙文礼 on 2023/1/10.
//

#ifndef SIMPLE_TEST_DATASET_HPP
#define SIMPLE_TEST_DATASET_HPP
#define  _CRT_SECURE_NO_WARNINGS


#include <iostream>

using namespace std;

static const char *father_path = "/Users/sunwenli/Desktop/simple-database/dataset/";
class dataset_Fixed {
public:
    static void get_data(const std::string& filename,std::size_t count){
        FILE *in_file = fopen((std::string(father_path)+filename).c_str(), "rb");

    }

private:
    std::string filename;

};

int main() {
    // printf("%lld\n", sizeof(long double));
    // Sleep(100000000000);
    char in_file_name[1024] = "/Users/sunwenli/Desktop/simple-database/dataset/longitudes-200M.bin.data";
    char out_file_name[1024] = "/Users/sunwenli/Desktop/simple-database/dataset/longitudes-200M.str.data1";
    FILE *in_file = fopen(in_file_name, "rb"), *out_file = fopen(out_file_name, "w");

    count = fread(buffer + all_count, sizeof(data_type), buffer_length, in_file)
    fclose(in_file);
}

#endif //SIMPLE_TEST_DATASET_HPP
