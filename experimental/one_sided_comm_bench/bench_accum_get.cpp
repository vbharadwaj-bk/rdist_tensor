#include <mpi.h>
#include <vector>
#include <memory>
#include <iostream>
#include <random>
#include <chrono>

using namespace std;

typedef chrono::time_point<std::chrono::steady_clock> my_timer_t; 

my_timer_t start_clock() {
    return std::chrono::steady_clock::now();
}

double stop_clock_get_elapsed(my_timer_t &start) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

/*
 * The benchmark function multiplies elementwise 
 * two tall-skinny matrices and sends each row
 * to a distinct processor for accumulation. This is
 * very similar to the communication pattern for a
 * tree-mode Khatri-Rao product. 
 */

void one_sided_mpi(
		double* a_ptr,
		double* b_ptr,
		double* c_ptr,
		vector<int> &destinations,
		vector<int> &destination_rows,
		int rows,
		int cols,
		int argc,
		char** argv
		) {

	MPI_Init(&argc, &argv);


	double * accum_window; MPI_Win win;

	MPI_Win_allocate(rows * cols * sizeof(double), 
		sizeof(double), 
		MPI_INFO_NULL,
		MPI_COMM_WORLD, 
		&accum_window, 
		&win); 

	MPI_Win_fence(MPI_MODE_NOPRECEDE, win);


	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			for(double k = 0; k < 35; k += 1.0) {
				c_ptr[i * cols + j] = a_ptr[i * cols + j] * b_ptr[i * cols + j] * k;
			}
		}

		MPI_Accumulate(
			c_ptr + (i * cols), 
			cols,
			MPI_DOUBLE, 
			destinations[i],
			destination_rows[i] * cols, 
			cols,
			MPI_DOUBLE, 
			MPI_SUM, 
			win);
	}

	for(int i = 0; i < rows; i++) {	
		MPI_Request req;
		MPI_Rput(
			c_ptr + (i * cols), 
			cols,
			MPI_DOUBLE, 
			destinations[i],
			destination_rows[i] * cols, 
			cols,
			MPI_DOUBLE,
			win,
			&req);
	}
	
	auto start = start_clock(); 
	MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
	double elapsed = stop_clock_get_elapsed(start);
	
	cout << "Fence time: " << elapsed << endl;

	MPI_Win_free(&win);
	MPI_Finalize();	
}


int main(int argc, char** argv) {
	int rows = atoi(argv[1]);
	int cols = atoi(argv[2]);
	string method(argv[3]);
	int proc_count = atoi(argv[4]);

	std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis_real(-1.0, 1.0);
	std::uniform_int_distribution<> dis_int(0, proc_count - 1);
	std::uniform_int_distribution<> dis_int_rows(0, rows-1);

	vector<double> a(rows * cols, 0.0);
	vector<double> b(rows * cols, 0.0);
	vector<double> c(rows * cols, 0.0);

	double* a_ptr = a.data();
	double* b_ptr = b.data();
	double* c_ptr = c.data();

	vector<int> destinations(rows, 0);
	vector<int> destination_rows(rows, 0);

    for (int i = 0; i < rows * cols; i++) {
		a[i] = dis_real(gen);
		b[i] = dis_real(gen);
		c[i] = dis_real(gen);
    }

    for (int i = 0; i < rows; i++) {
		destinations[i] = dis_int(gen);
		destination_rows[i] = dis_int_rows(gen);
	}

	if(method == "one_sided_mpi") {
		cout << "Benchmarking one-sided MPI!" << endl;
		auto start = start_clock();
		one_sided_mpi(a_ptr, 
				b_ptr, 
				c_ptr, 
				destinations, 
				destination_rows, 
				rows, 
				cols, 
				argc, 
				argv);
		double elapsed = stop_clock_get_elapsed(start);
		cout << "Time elapsed: " << elapsed << endl;
	}
	else {
		cout << "Error, invalid method!" << endl;
		exit(1);
	}
}
