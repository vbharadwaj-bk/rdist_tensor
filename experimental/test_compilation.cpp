#include "cblas.h"
#include "lapacke.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
    // Test out the dot product function from cblas
    double x[] = {1, 2, 3, 4, 5, 6};
    double y[] = {1, 2, 3, 4, 5, 6};
    double result = cblas_ddot(6, x, 1, y, 1);
    cout << "Result: " << result << endl;

    // Test out the matrix multiplication function from cblas
    double a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double b[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double c[9];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1, a, 3, b, 3, 0, c, 3);

    cout << "Result: " << endl;
    for (int i = 0; i < 9; i++) {
        cout << c[i] << " ";
    }
    cout << endl;
    return 0; 
}