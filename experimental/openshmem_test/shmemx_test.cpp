#include <shmem.h> 
#include <shmemx.h> 
#include <iostream>

using namesapce std;

int main(int argc, char** argv) {
    shmem_init();

    cout << "SHEM Initialized!" << endl;

    shmem_finalize();

    cout << "SHMEM Finalized!" << endl;
}

