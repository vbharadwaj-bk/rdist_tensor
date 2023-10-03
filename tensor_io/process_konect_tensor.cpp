#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <cassert>
#include <chrono>
#include <hdf5.h>

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
void convertFromKonect(string in_file, hsize_t num_lines) {
    cout << "Starting file conversion!" << endl; 

    string converted_filename = in_file + "_converted.hdf5";
    hid_t file = H5Fcreate(converted_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    std::string line, token;
    std::ifstream iffstream;
    iffstream.open(in_file);

    // Skip the first ine
    getline(iffstream, line);

    hsize_t dim = 5;

    hid_t idx_datatype;
    hid_t val_datatype; 
    if(std::is_same<IDX_T, uint64_t>::value) {
        idx_datatype = H5Tcopy(H5T_NATIVE_ULLONG);
    }
    else if(std::is_same<IDX_T, uint32_t>::value) {
        idx_datatype = H5Tcopy(H5T_NATIVE_UINT);
    }
    else {
        assert(false);
    }

    if(std::is_same<VAL_T, double>::value) {
        val_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
    }
    else if(std::is_same<IDX_T, float>::value) {
        val_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
    }
    else {
        assert(false);
    }

    hid_t file_dataspace = H5Screate_simple(1, &num_lines, NULL); 
    hid_t mode_size_dataspace = H5Screate_simple(1, &dim, NULL);

    string max_mode_set = "MAX_MODE_SET";
    string min_mode_set = "MIN_MODE_SET";

    hid_t max_mode_dataset = H5Dcreate(file, max_mode_set.c_str(), idx_datatype, mode_size_dataspace,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 

    hid_t min_mode_dataset = H5Dcreate(file, min_mode_set.c_str(), idx_datatype, mode_size_dataspace,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 

    vector<unique_ptr<IDX_T>> idx_buffers;
    vector<IDX_T> mode_maxes;
    vector<IDX_T> mode_mins;

    vector<hid_t> datasets;

    unique_ptr<VAL_T> val_buffer(new VAL_T[BUFFER_SIZE]); 

    cout << "Dimension: " << dim << endl; 
    cout << "NNZ: " << num_lines << endl; 

    for(int i = 0; i < dim; i++) {
        idx_buffers.emplace_back(new IDX_T[BUFFER_SIZE]);

        string datasetname = "MODE_" + std::to_string(i);

        hid_t idx_dataset = H5Dcreate(file, datasetname.c_str(), idx_datatype, file_dataspace,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 

        datasets.push_back(idx_dataset);

        mode_maxes.push_back(0);
        mode_mins.push_back(0);
    }

    string value_set_name = "VALUES";
    hid_t val_dataset = H5Dcreate(file, value_set_name.c_str(), val_datatype, 
                file_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
    datasets.push_back(val_dataset);

    hsize_t buffer_size = BUFFER_SIZE;
    hid_t memory_dataspace = H5Screate_simple(1, &buffer_size, NULL); 

    hsize_t pos_in_buffer = 0;
    vector<uint32_t> quantities(3, 0);

    for(hsize_t i = 0; i < num_lines; i++) {
        if(i % 100000 == 0) {
            cout << "Processed up to line " << i << endl;
        }
        for(int j = 0; j < 4; j++) {
            if(j < 3) {
                IDX_T idx;
                iffstream >> idx;
                if(j < 2) {
                    idx_buffers[j].get()[pos_in_buffer] = idx;

                    if (mode_maxes[j] == 0 || mode_maxes[j] < idx) {
                        mode_maxes[j] = idx;
                    }

                    if (mode_mins[j] == 0 || mode_mins[j] > idx) {
                        mode_mins[j] = idx;
                    }
                }
            }
            else {
                uint64_t timestamp;
                iffstream >> timestamp;
                std::chrono::seconds seconds{timestamp};
                TimePoint time_point(seconds);
                const std::chrono::year_month_day ymd{std::chrono::floor<std::chrono::days>(time_point)};

                quantities[0] = ((uint32_t) static_cast<int>(ymd.year()));
                quantities[1] = (static_cast<unsigned>(ymd.month()));
                quantities[2] = (static_cast<unsigned>(ymd.day()));

                for(uint32_t k = 2; k < 5; k++) {
                    uint32_t quantity = quantities[k-2];
                    idx_buffers[k].get()[pos_in_buffer] = quantity;

                    if (mode_maxes[k] == 0 || mode_maxes[k] < quantity) {
                        mode_maxes[k] = quantity;
                    }

                    if (mode_mins[k] == 0 || mode_mins[k] > quantity) {
                        mode_mins[k] = quantity;
                    }
                }
            }
        }
        VAL_T val = 1.0;
        val_buffer.get()[pos_in_buffer] = val; 

        pos_in_buffer++;

        if(pos_in_buffer == buffer_size) {
            hsize_t offset = i + 1 - pos_in_buffer;

            int status = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, &offset, NULL, 
                    &buffer_size, NULL);

            for(int j = 0; j < dim; j++) {
                status = H5Dwrite(datasets[j], idx_datatype, memory_dataspace, file_dataspace,
                        H5P_DEFAULT, idx_buffers[j].get());
            }

            status = H5Dwrite(datasets[dim], val_datatype, memory_dataspace, file_dataspace,
                    H5P_DEFAULT, val_buffer.get());

            pos_in_buffer = 0;
        }
    }

    if(pos_in_buffer > 0) {
        hsize_t file_offset = num_lines - pos_in_buffer;
        hsize_t memory_offset = 0;

        int status = H5Sselect_hyperslab(memory_dataspace, H5S_SELECT_SET, &memory_offset, NULL, 
                &pos_in_buffer, NULL);
        status = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, &file_offset, NULL, 
                &pos_in_buffer, NULL);

        for(int j = 0; j < dim; j++) {
            status = H5Dwrite(datasets[j], idx_datatype, memory_dataspace, file_dataspace,
                    H5P_DEFAULT, idx_buffers[j].get());
        }

        status = H5Dwrite(datasets[dim], val_datatype, memory_dataspace, file_dataspace,
                H5P_DEFAULT, val_buffer.get());

        pos_in_buffer = 0;
    }

    int status = H5Dwrite(max_mode_dataset, idx_datatype, H5P_DEFAULT, H5P_DEFAULT,
            H5P_DEFAULT, mode_maxes.data());

    status = H5Dwrite(min_mode_dataset, idx_datatype, H5P_DEFAULT, H5P_DEFAULT,
            H5P_DEFAULT, mode_mins.data());

    iffstream.close();

    for(int i = 0; i < datasets.size(); i++) {
        H5Dclose(datasets[i]);
    }


    H5Dclose(max_mode_dataset);
    H5Dclose(min_mode_dataset);

    H5Tclose(idx_datatype);
    H5Tclose(val_datatype);
    H5Sclose(mode_size_dataspace); 
    H5Sclose(file_dataspace); 
    H5Sclose(memory_dataspace); 

    status = H5Fclose(file);
}

int main(int argc, char** argv) {
    string name(argv[1]);
    hsize_t num_lines = atol(argv[2]);
 
    convertFromKonect<uint32_t, double>(name, num_lines);
}