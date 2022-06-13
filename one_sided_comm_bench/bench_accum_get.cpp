#include <mpi.h>
#include <vector>
#include <memory>
#include <iostream>

using namespace std;

/*
 * The benchmark function multiplies elementwise 
 * two tall-skinny matrices and sends each row
 * to a distinct processor for accumulation. This is
 * very similar to the communication pattern for a
 * tree-mode Khatri-Rao product. 
 */

void one_sided_mpi(
		vector<double> &a,
		vector<double> &b,
		vector<double> &c,
		vector<int> destinations,
		int rows,
		int cols
		) {
	
	double* a_ptr = a.data();
	double* b_ptr = b.data();
	double* c_ptr = c.data();

	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < r; j++) {
			c_ptr[i * r + j] = a_ptr[i * r + j] * b_ptr[i * r + j];
		}
	}
}


int main(int argc, char** argv) {
	int n = atoi(argv[1]);
	string method(argv[2]);
	int proc_count = atoi(argv[3]);

	std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis_real(-1.0, 1.0);
    std::uniform_real_distribution<> dis_real(-1.0, 1.0);
	std::uniform_int_distribution<> dis_int(0, proc_count - 1);

	vector<double> a(n, 0.0);
	vector<double> b(n, 0.0);
	vector<double> c(n, 0.0);
	vector<int> destinations(n, 0);

    for (int i = 0; i < n; i++) {
		a[i] = dis_real(gen);
		b[i] = dis_real(gen);
		c[i] = dis_real(gen);

		destinations[i] = dis_int(gen);
    }

	if(method == "one_sided_mpi") {
		cout << "Benchmarking one-sided MPI!" << endl;
	}
	else {
		cout << "Error, invalid method!" << endl;
		exit(1);
	}
}
