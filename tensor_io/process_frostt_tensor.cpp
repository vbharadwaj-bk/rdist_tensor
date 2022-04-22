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

    std::string line, token;
    std::ifstream firstline_stream(in_file);
    std::ifstream iffstream(in_file);

    // Read the first line and count the number of tokens 
    std::getline(firstline_stream, line); 
    std::istringstream is( line );

    int count;
    while ( std::getline( is, line, ' ' )) {
        ++count;       
    }

    firstline_stream.close();
    int dim = count - 1;
    int buffer_pos = 0;

    vector<unique_ptr<unsigned long long>> idx_buffers;
    unique_ptr<double> val_buffer(new double[BUFFER_SIZE]); 

    for(int i = 0; i < dim; i++) {
        idx_buffers.emplace_back(new unsigned long long[BUFFER_SIZE]); 
    }

    for(int i = 0; i < num_lines * count; i++) {
        for(int j = 0; j < dim; j++) {
            idx_buffers[j][i] << iffstream;
            cout << idx_buffers[j][i] << endl;
        }
        val_buffer[i] << iffstream;
        cout << val_buffer[i] << endl;
        buffer_pos++;
    }
 
    firstline_stream.close();

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
    convertFromFROSTT("../tensors/test.tns", 5);
}