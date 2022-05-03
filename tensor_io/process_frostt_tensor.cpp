#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <cassert>
#include <hdf5.h>

#define BUFFER_SIZE 10000

using namespace std;

/*
 * This file assumes that the tensor is 1-indexed. 
 */

void convertFromFROSTT(string in_file, unsigned long long num_lines) {
    cout << "Starting file conversion!" << endl; 

    string converted_filename = in_file + "_converted.hdf5";
    hid_t file = H5Fcreate(converted_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    std::string line, token;
    std::ifstream firstline_stream(in_file, std::ifstream::in);
    std::ifstream iffstream(in_file, std::ifstream::in);

    // Read the first line and count the number of tokens 
    std::getline(firstline_stream, line); 
    std::istringstream is( line );

    
    int count = 0;
    while ( std::getline( is, token, ' ')) {
        ++count;       
    }
    

    firstline_stream.close();
    hsize_t dim = count - 1;

    hid_t idx_datatype = H5Tcopy(H5T_NATIVE_ULLONG);
    hid_t val_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);

    hid_t file_dataspace = H5Screate_simple(1, &num_lines, NULL); 


    hid_t mode_size_dataspace = H5Screate_simple(1, &dim, NULL);


    string max_mode_set = "MAX_MODE_SET";
    string min_mode_set = "MIN_MODE_SET";

    hid_t max_mode_dataset = H5Dcreate(file, max_mode_set.c_str(), idx_datatype, mode_size_dataspace,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 

    hid_t min_mode_dataset = H5Dcreate(file, min_mode_set.c_str(), idx_datatype, mode_size_dataspace,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 

    vector<unique_ptr<unsigned long long>> idx_buffers;
    vector<unsigned long long> mode_maxes;
    vector<unsigned long long> mode_mins;

    vector<hid_t> datasets;

    unique_ptr<double> val_buffer(new double[BUFFER_SIZE]); 

    cout << "Dimension: " << dim << endl; 
    cout << "NNZ: " << num_lines << endl; 


    for(int i = 0; i < dim; i++) {
        idx_buffers.emplace_back(new unsigned long long[BUFFER_SIZE]);

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

    unsigned long long pos_in_buffer = 0;

    for(hsize_t i = 0; i < num_lines; i++) {
        for(int j = 0; j < dim; j++) {
            unsigned long long idx;
            iffstream >> idx;
            idx_buffers[j].get()[pos_in_buffer] = idx;

            if (mode_maxes[j] == 0 || mode_maxes[j] < idx) {
                mode_maxes[j] = idx;
            }

            if (mode_mins[j] == 0 || mode_mins[j] > idx) {
                mode_mins[j] = idx;
            }
        }
        double val;
        iffstream >> val;
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

int main(int* argc, char** argv) {
    string name(argv[1]);
    unsigned long long num_lines = atol(argv[2]);
 
    convertFromFROSTT(name, num_lines);
}