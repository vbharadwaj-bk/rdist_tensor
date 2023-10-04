#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <cassert>
#include <chrono>

using namespace std;

/*
 * This file assumes that the tensor is 1-indexed. We read in the tensor
 * 
 * It is intended to read a Konect temporal network
 * and turn it into a tensor with the timestamp replaced by year, month, day. 
 * 
 */
template<typename IDX_T, typename VAL_T>
void conv(string in_file, uint64_t num_lines) {
    cout << "Starting file conversion!" << endl; 

    string converted_filename = in_file + "_1f.tns";

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

    for(uint64_t i = 0; i < num_lines; i++) {
        if(i % 100000 == 0) {
            cout << "Processed up to line " << i << endl;
        }
        for(int j = 0; j < 5; j++) {
            IDX_T idx;
            iffstream >> idx;  
            idx_buffer[j] = idx;  
        }
        VAL_T val;
        iffstream >> val; 
        val = 1.0;
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
 
    conv<uint32_t, double>(name, num_lines);
}