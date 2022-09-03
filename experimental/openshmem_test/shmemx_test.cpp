#include <shmem.h> 
#include <shmemx.h> 
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
    shmem_init();

    int my_rank = _my_pe();

    if(my_rank == 0) {
        cout << "SHEM Initialized!" << endl;
    }

    shmem_finalize();

    if(my_rank == 0) {
        cout << "SHMEM Finalized!" << endl;
    } 
}

