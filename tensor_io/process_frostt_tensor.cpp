#include <iostream>
#include <fstream>
#include <string>
#include <h5cpp/hdf5.hpp>

#include

using namespace std;

void convertFromFROSTT(string in_file, int num_lines) {

    hid_t       file;                 /* declare file identifier */
    /*
    * Create a new file using H5ACC_TRUNC 
    * to truncate and overwrite any file of the same name,
    * default file creation properties, and 
    * default file access properties.
    * Then close the file.
    */
    file = H5Fcreate(FILE, H5ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t    dataset, datatype, dataspace;  /* declare identifiers */
    
    /* 
     * Create a dataspace: Describe the size of the array and 
     * create the dataspace for a fixed-size dataset. 
     */

    hsize_t dimsf[1];
    dimsf[0] = 20;
    
    dataspace = H5Screate_simple(RANK, dimsf, NULL); 
    /*
     * Define a datatype for the data in the dataset.
     * We will store little endian integers.
     */
    datatype = H5Tcopy(H5T_NATIVE_INT);
    status = H5Tset_order(datatype, H5T_ORDER_LE);
    /*
     * Create a new dataset within the file using the defined 
     * dataspace and datatype and default dataset creation
     * properties.
     * NOTE: H5T_NATIVE_INT can be used as the datatype if 
     * conversion to little endian is not needed.
     */

    char* DATSETNAME = "MY_TEST_DATASET"

    dataset = H5Dcreate(file, DATASETNAME, datatype, dataspace,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5Tclose(datatype);
    H5Dclose(dataset);
    H5Sclose(dataspace); 

    status = H5Fclose(file); 
}

int main(int argc*, char** argv) {
    convertFromFROSTT()
}