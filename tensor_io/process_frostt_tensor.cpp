#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <hdf5.h>

#define BUFFER_SIZE 10

using namespace std;

void convertFromFROSTT(string in_file, int num_lines) {
    cout << "Starting file conversion!" << endl; 

    string converted_filename = in_file + "_converted.tns"
    hid_t file = H5Fcreate(converted_filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    std::string line, token;
    std::ifstream firstline_stream(in_file, std::ifstream::in);
    std::ifstream iffstream(in_file);

    // Read the first line and count the number of tokens 
    std::getline(firstline_stream, line); 
    std::istringstream is( line );

    int count = 0;
    while ( std::getline( is, token, ' ')) {
        ++count;       
    }

    firstline_stream.close();
    int dim = count - 1;
    int buffer_pos = 0;

    hid_t idx_datatype = H5Tcopy(H5T_NATIVE_INT);
    hid_t val_datatype = H5Tcopy(H5T_NATIVE_ULLONG);
    hid_t file_dataspace = H5Screate_simple(1, &num_lines, NULL); 

    vector<unique_ptr<unsigned long long>> idx_buffers;
    vector<hid_t> datasets;

    unique_ptr<double> val_buffer(new double[BUFFER_SIZE]); 

    for(int i = 0; i < dim; i++) {
        idx_buffers.emplace_back(new unsigned long long[BUFFER_SIZE]);

        string datasetname = "MODE_" + std::to_string(i);

        hid_t idx_dataset = H5Dcreate(file, datasetname.c_str(), idx_datatype, file_dataspace,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 

        datasets.push_back(idx_dataset);
    }

    string value_set_name = "VALUES";
    hid_t val_dataset = H5Dcreate(file, value_set_name.c_str(), val_datatype, 
                file_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
    datasets.push_back(val_dataset);

    cout << dim << endl; 

    for(int i = 0; i < num_lines; i++) {
        for(int j = 0; j < dim; j++) {
            unsigned long long idx;
            iffstream >> idx;
            idx_buffers[j].get()[i] = idx; 
        }
        double val;
        iffstream >> val;
        val_buffer.get()[i] = val; 
        buffer_pos++;
    }
 
    iffstream.close();

    H5Tclose(idx_datatype);
    H5Tclose(val_datatype);
    H5Sclose(file_dataspace); 

    for(int i = 0; i < datasets.size(); i++) {
        H5Dclose(datasets[i]);
    }

    status = H5Fclose(file);

    /*
    hid_t       file;                     
    file = H5Fcreate(in_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t    dataset, datatype, dataspace;  
    
    hsize_t dimsf[1];
    dimsf[0] = 20;
    
    dataspace = H5Screate_simple(1, dimsf, NULL); 
    datatype = H5Tcopy(H5T_NATIVE_INT);
    int status = H5Tset_order(datatype, H5T_ORDER_LE);
    char* DATASETNAME = "MY_TEST_DATASET";

    dataset = H5Dcreate(file, DATASETNAME, datatype, dataspace,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);


    hsize_t buffer_size = 5;
    hid_t local_space = H5Screate_simple(1, &buffer_size, NULL); 

    int data[5];
    for(int i = 0; i < 5; i++) {
        data[i] = i;
    }

    hsize_t offset = 5;
    hsize_t count = 5;

    hid_t dataset_space = H5Dget_space(dataset); 
    status = H5Sselect_hyperslab(dataset_space, H5S_SELECT_SET, &offset, NULL, 
             &count, NULL);

    status = H5Dwrite(dataset, H5T_NATIVE_INT, local_space, dataset_space,
              H5P_DEFAULT, data);

    H5Tclose(datatype);
    H5Dclose(dataset);
    H5Sclose(dataspace); 

    status = H5Fclose(file);
    */
}

int main(int* argc, char** argv) {
    convertFromFROSTT("../../tensors/test.tns", 4);
}