#include <iostream>
#include <fstream>
#include <string>
#include <hdf5.h>

using namespace std;

void convertFromFROSTT(string in_file, int num_lines) {

    // Read the first line of a 

    hid_t       file;                 /* declare file identifier */
    /*
    * Create a new file using H5ACC_TRUNC 
    * to truncate and overwrite any file of the same name,
    * default file creation properties, and 
    * default file access properties.
    * Then close the file.
    */
    file = H5Fcreate(in_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t    dataset, datatype, dataspace;  /* declare identifiers */
    
    /* 
     * Create a dataspace: Describe the size of the array and 
     * create the dataspace for a fixed-size dataset. 
     */

    hsize_t dimsf[1];
    dimsf[0] = 20;
    
    dataspace = H5Screate_simple(1, dimsf, NULL); 
    /*
     * Define a datatype for the data in the dataset.
     * We will store little endian integers.
     */
    datatype = H5Tcopy(H5T_NATIVE_INT);
    int status = H5Tset_order(datatype, H5T_ORDER_LE);
    /*
     * Create a new dataset within the file using the defined 
     * dataspace and datatype and default dataset creation
     * properties.
     * NOTE: H5T_NATIVE_INT can be used as the datatype if 
     * conversion to little endian is not needed.
     */

    char* DATASETNAME = "MY_TEST_DATASET";

    dataset = H5Dcreate(file, DATASETNAME, datatype, dataspace,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);


    hsize_t buffer_size = 5;
    hid_t local_space = H5Screate_simple(1, &buffer_size, NULL); 
    //status = H5Sselect_hyperslab(dataset_space, H5S_SELECT_SET, &offset, NULL, 
    //         &count, NULL);

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
}

int main(int* argc, char** argv) {
    convertFromFROSTT("test.hdf5", 5);
}