#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <cassert>
#include <chrono>

#define BUFFER_SIZE 100000

using namespace std;
using namespace std::chrono;
using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

/*
 * This file assumes that the tensor is 1-indexed. We read in the tensor
 * 
 * It is intended to read a Konect temporal network
 * and turn it into a tensor with the timestamp replaced by year, month, day. 
 * 
 */
template<typename IDX_T, typename VAL_T>
void convertFromKonect(string in_file, uint64_t num_lines) {
    cout << "Starting file conversion!" << endl; 

    string converted_filename = in_file + "_converted.tns";

    std::string line, token;
    std::ifstream iffstream;
    iffstream.open(in_file);

    ofstream offstream;
    offstream.open(converted_filename, ios::out);

    // Skip the first ine
    getline(iffstream, line);

    uint64_t dim = 5;
    vector<IDX_T> idx_buffer(dim, 0); 

    cout << "Dimension: " << dim << endl; 
    cout << "NNZ: " << num_lines << endl; 

    vector<uint32_t> quantities(3, 0);

    for(uint64_t i = 0; i < num_lines; i++) {
        if(i % 100000 == 0) {
            cout << "Processed up to line " << i << endl;
        }
        for(int j = 0; j < 4; j++) {
            if(j < 3) {
                IDX_T idx;
                iffstream >> idx;
                if(j < 2) {
                    idx_buffer[j] = idx;
                }
            }
            else {
                uint64_t timestamp;
                iffstream >> timestamp;
                std::chrono::seconds seconds{timestamp};
                TimePoint time_point(seconds);
                const std::chrono::year_month_day ymd{std::chrono::floor<std::chrono::days>(time_point)};

                quantities[0] = ((uint32_t) static_cast<int>(ymd.year())) - 2000;
                quantities[1] = (static_cast<unsigned>(ymd.month()));
                quantities[2] = (static_cast<unsigned>(ymd.day()));

                for(uint32_t k = 2; k < 5; k++) {
                    uint32_t quantity = quantities[k-2];
                    idx_buffer[k] = quantity;
                }
            }
        }
        VAL_T val = 1.0;
        offstream << idx_buffer[0] 
                << " " << idx_buffer[1] 
                << " " << idx_buffer[2]
                << " " << idx_buffer[3] 
                << " " << idx_buffer[4]
                << " " << val 
                << "\n";
    }
    iffstream.close();
    offstream.close();
}

int main(int argc, char** argv) {
    string name(argv[1]);
    uint64_t num_lines = atol(argv[2]);
 
    convertFromKonect<uint32_t, double>(name, num_lines);
}