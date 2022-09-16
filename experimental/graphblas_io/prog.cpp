extern "C" {
  #include <GraphBLAS.h>
}

#include <algorithm>
#include <atomic>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <bitset>
#include <cassert>
#include <filesystem>
#include <queue>
#include <climits>
#include <cstdio>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include "progressbar.hpp"
#include <omp.h>
#include <hdf5.h>

using namespace std;
namespace fs = std::filesystem;

#define BLOCKSIZE 512
#define BUFFERSIZE 200000000

struct posix_header
{                              /* byte offset */
  char name[100];               /*   0 */
  char mode[8];                 /* 100 */
  char uid[8];                  /* 108 */
  char gid[8];                  /* 116 */
  char size[12];                /* 124 */
  char mtime[12];               /* 136 */
  char chksum[8];               /* 148 */
  char typeflag;                /* 156 */
  char linkname[100];           /* 157 */
  char magic[6];                /* 257 */
  char version[2];              /* 263 */
  char uname[32];               /* 265 */
  char gname[32];               /* 297 */
  char devmajor[8];             /* 329 */
  char devminor[8];             /* 337 */
  char prefix[155];             /* 345 */
                                /* 500 */
};

int octal_string_to_int(char *current_char, unsigned int size){
    unsigned int output = 0;
    while(size > 0){
        output = output * 8 + *current_char - '0';
        current_char++;
        size--;
    }
    return output;
}

int divide_and_roundup(int x, int y) {
  return 1 + ((x - 1) / y);
}

/*
 * Not multithreaded, since HDF5 is currently not multithreaded.
 * This  
 */
class HDF5_Writer {
public:
  hsize_t dim;
  atomic_uint32_t threads_writing;

  vector<vector<uint32_t>> idx_buffers; 
  vector<double> val_buffer;
  hsize_t buffer_size, buffer_position; 

  hid_t file;
  hsize_t file_offset;

  hid_t memory_dataspace, file_dataspace;

  hid_t idx_datatype, val_datatype;
  vector<hid_t> datasets;

  hid_t mode_size_dataspace; 
  hid_t max_mode_dataset, min_mode_dataset;

  HDF5_Writer(hsize_t array_size,
                        hsize_t buffer_size, 
                        string filename
  ) {
    dim = 3;
    file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    file_dataspace = H5Screate_simple(1, &array_size, NULL); 
    memory_dataspace = H5Screate_simple(1, &buffer_size, NULL); 
    mode_size_dataspace = H5Screate_simple(1, &dim, NULL);

    this->buffer_size = buffer_size;

    for(int i = 0; i < 3; i++) {
      idx_buffers.emplace_back(buffer_size, 0);
    }
    val_buffer.resize(buffer_size);

    buffer_position = 0;
    file_offset = 0;
    threads_writing = 0;

    idx_datatype = H5Tcopy(H5T_NATIVE_UINT);
    val_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
    for(int i = 0; i < 3; i++) {
      string datasetname = "MODE_" + std::to_string(i);

      hid_t idx_dataset = H5Dcreate(file, datasetname.c_str(), idx_datatype, file_dataspace,
                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
      datasets.push_back(idx_dataset);
    }
    
    string value_set_name = "VALUES";

    hid_t val_dataset = H5Dcreate(file, value_set_name.c_str(), val_datatype, 
                file_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
    datasets.push_back(val_dataset);

    string max_mode_set = "MAX_MODE_SET";
    string min_mode_set = "MIN_MODE_SET";

    max_mode_dataset = H5Dcreate(file, max_mode_set.c_str(), idx_datatype, mode_size_dataspace,
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 

    min_mode_dataset = H5Dcreate(file, min_mode_set.c_str(), idx_datatype, mode_size_dataspace,
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 

  }

  void write_dataset_bounds(uint32_t row_bound, 
      uint32_t col_bound, 
      uint32_t time_bound) {
    // Assumes that the mins are all zero

    vector<uint32_t> mode_maxes = {row_bound, col_bound, time_bound};
    vector<uint32_t> mode_mins(3, 0); 

    int status = H5Dwrite(max_mode_dataset, idx_datatype, H5P_DEFAULT, H5P_DEFAULT,
            H5P_DEFAULT, mode_maxes.data());

    status = H5Dwrite(min_mode_dataset, idx_datatype, H5P_DEFAULT, H5P_DEFAULT,
            H5P_DEFAULT, mode_mins.data());
  }

  void flush_buffer() {
      cout << "Flushing memory buffer to disk at position " << file_offset << endl;
      hsize_t memory_offset = 0;
      int status = H5Sselect_hyperslab(memory_dataspace, H5S_SELECT_SET, &memory_offset, NULL, 
              &buffer_position, NULL);
      status = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, &file_offset, NULL, 
              &buffer_position, NULL);

      for(int j = 0; j < dim; j++) {
          status = H5Dwrite(datasets[j], idx_datatype, memory_dataspace, file_dataspace,
                  H5P_DEFAULT, idx_buffers[j].data());
      }
      status = H5Dwrite(datasets[dim], val_datatype, memory_dataspace, file_dataspace,
              H5P_DEFAULT, val_buffer.data());

      file_offset += buffer_position; 
      buffer_position = 0;
  }

  ~HDF5_Writer() {
    for(int i = 0; i < datasets.size(); i++) {
        H5Dclose(datasets[i]);
    }

    H5Dclose(max_mode_dataset);
    H5Dclose(min_mode_dataset);

    H5Tclose(idx_datatype);
    H5Tclose(val_datatype); 

    H5Sclose(file_dataspace); 
    H5Sclose(memory_dataspace); 
    H5Sclose(mode_size_dataspace); 

    H5Fclose(file);
  }

  /*
   * Writes to the buffer; at intervals, flushes the buffer
   * to disk and resets the position marker. 
   */
  uint32_t reserve_for_write(uint32_t space_req) {
    hsize_t position_capture;
    #pragma omp critical
    {
      if(space_req > buffer_size) {
        assert(false);
      }
      if(buffer_position + space_req > buffer_size) {
        
        uint32_t thread_capture;

        do {
          thread_capture = threads_writing; 
        }
        while(thread_capture != 0);

        flush_buffer();
      }

      threads_writing++;
      position_capture = buffer_position;
      
      buffer_position += space_req;
    }
    return position_capture;
  }

  void complete_write() {
    threads_writing--;
  }
};

class CAIDA_Reader {
  uint64_t total_nnz;
  double total_packet_count;
  int pagesize;

  vector<double> row_nnz_counts, col_nnz_counts;
  vector<uint32_t> row_compress, col_compress;
  vector<string> tar_list;

  uint32_t max_row, max_col;

  hsize_t nrows, ncols;

  HDF5_Writer* writer;

public:
  CAIDA_Reader(vector<string> &data_folders) {
    GrB_init (GrB_NONBLOCKING);
    total_nnz = 0;
    total_packet_count = 0;
    pagesize = getpagesize();

    nrows = UINT32_MAX;
    ncols = UINT32_MAX;

    // Let's begin by computing the rows and columns of the
    // sparse tensor with the greatest number of nonzeros 
    row_nnz_counts.resize(nrows);
    col_nnz_counts.resize(ncols);

    row_compress.resize(nrows);
    col_compress.resize(ncols);

    writer = nullptr;

    get_tarfile_list(data_folders);
    process_caida_data(data_folders);
    cout << "\nInitialized CAIDA Dataset!" << endl; 
    cout << "Total nonzeros: " << total_nnz << endl;
    cout << "Total packet count: " << total_packet_count << endl;

    string output_filename("/pscratch/sd/v/vbharadw/tensors/caida_data.hdf5");
    writer = new HDF5_Writer(total_nnz, BUFFERSIZE, output_filename);

    compress_zero_elements();

    writer->write_dataset_bounds(max_row, max_col, 64 * tar_list.size());

    string max_mode_set = "MAX_MODE_SET";
    string min_mode_set = "MIN_MODE_SET";


    // The second time around, we write the processed nonzeros to an HDF5 
    // file
    process_caida_data(data_folders);
    writer->flush_buffer();

    //write_stats_to_file();
  }

  ~CAIDA_Reader() {
    delete writer;
    GrB_finalize ();
  }

  void compress_zero_elements() {
    // Sadly, this is not multithreaded 

    uint32_t current = 0; 
    for(int i = 0; i < row_nnz_counts.size(); i++) {
      uint32_t capture = current;
      if(row_nnz_counts[i] > 0.0) {
        current++;
      }
      row_compress[i] = capture; 
    }

    max_row = current;
    cout << "Row Current: " << max_row << endl;

    current = 0; 
    for(int i = 0; i < col_nnz_counts.size(); i++) {
      uint32_t capture = current;
      if(col_nnz_counts[i] > 0.0) {
        current++;
      }
      col_compress[i] = capture;  
    }

    max_col = current;
    cout << "Col Current: " << max_col << endl;
  }

  void write_stats_to_file() {
    string filename("/pscratch/sd/v/vbharadw/tensors/caida_stats.hdf5");

    string rowset_name("ROW_COUNTS");
    string colset_name("COL_COUNTS");

    hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t count_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);

    hid_t rowct_dataspace = H5Screate_simple(1, &nrows, NULL); 
    hid_t colct_dataspace = H5Screate_simple(1, &ncols, NULL); 

    hid_t rowct_dataset = H5Dcreate(file, 
                rowset_name.c_str(), count_datatype, rowct_dataspace,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hid_t colct_dataset = H5Dcreate(file, 
                colset_name.c_str(), count_datatype, colct_dataspace,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    int status = H5Dwrite(rowct_dataset, count_datatype, H5P_DEFAULT, H5P_DEFAULT,
            H5P_DEFAULT, row_nnz_counts.data());

    status = H5Dwrite(colct_dataset, count_datatype, H5P_DEFAULT, H5P_DEFAULT,
            H5P_DEFAULT, col_nnz_counts.data());

    H5Dclose(rowct_dataset);
    H5Dclose(colct_dataset);
    H5Sclose(rowct_dataspace);
    H5Sclose(colct_dataspace);
    H5Tclose(count_datatype);

    status = H5Fclose(file);
  }

  int read_graphblas_file(char* buf, int tarfile_idx, int idx_in_tarfile) {
        struct posix_header* header =
          (struct posix_header*) buf; 
          // In case we need the filename for something

          bitset<8> size_highbits(*(header->size)); 

          assert(size_highbits[0] == 0);
          int size_of_file = octal_string_to_int(header->size, 11);

          if(size_of_file == 0)
            return 0;

          char* contents = buf + BLOCKSIZE;

          //std::string filename(header->name);

          GrB_Matrix C;
          GrB_Matrix_deserialize(
              &C, 
              NULL, 
              (void*) contents, 
              size_of_file
              ); 

          GrB_Index nrows;
          GrB_Index ncols;
          GrB_Index nvals;
          GrB_Matrix_nrows(&nrows, C);
          GrB_Matrix_nrows(&ncols, C);
          GrB_Matrix_nvals(&nvals, C);

          vector<GrB_Index> I(nvals, 0);
          vector<GrB_Index> J(nvals, 0);
          vector<uint32_t> V(nvals, 0);

          #pragma omp atomic
          total_nnz += nvals;

          GrB_Matrix_extractTuples_UINT32(
            I.data(),
            J.data(),
            V.data(),
            &nvals,
            C
          );

          if(writer == nullptr) {
            for(int i = 0; i < nvals; i++) {
              #pragma omp atomic
              row_nnz_counts[I[i]]++;

              #pragma omp atomic
              col_nnz_counts[J[i]]++;

              #pragma omp atomic 
              total_packet_count += V[i];
            }
          }
          else {
            hsize_t position = writer->reserve_for_write(nvals);

            uint32_t time_component = tarfile_idx * 64 + idx_in_tarfile; 

            for(int i = 0; i < nvals; i++) {
              writer->idx_buffers[0][position + i] = row_compress[I[i]];
              writer->idx_buffers[1][position + i] = col_compress[J[i]];
              writer->idx_buffers[2][position + i] = time_component; 
              writer->val_buffer[position + i] = V[i];
            }
            writer->complete_write();
          }

          GrB_Matrix_free(&C);

          // Must keep this as a multiple of BLOCKSIZE 
          int bytes_parsed = BLOCKSIZE 
              + divide_and_roundup(size_of_file, BLOCKSIZE) * BLOCKSIZE;

          return bytes_parsed;
  }

  void read_tarfile(string path, int tarfile_idx) {
      struct stat sb;
      stat(path.c_str(), &sb);
      int size = sb.st_size;
      int rounded_size = divide_and_roundup(size, pagesize) * pagesize;

      FILE* in_file = fopen(path.c_str(), "rb"); 
      vector<char> read_buffer(size, 0);

      //cout << "Starting file read!" << endl;
      fread(read_buffer.data(), size, 1, in_file);
      //cout << "Ended file read!" << endl;

      //fclose(in_file);

      //int fd = open(path.c_str(), O_RDONLY);
      //char* data = (char*) mmap((caddr_t) 0, rounded_size, PROT_READ, MAP_SHARED, fd, 0);

      int position = 0;

      for(int i = 0; i < 64; i++) {
        int bytes_parsed = read_graphblas_file(read_buffer.data() + position, tarfile_idx, i);
        position += bytes_parsed;
      }
      fclose(in_file);
      //munmap((caddr_t) data, rounded_size); 
  }

  void get_tarfile_list(vector<string> &data_folders) {
    queue<fs::path> remaining_folders;

    for (int i = 0; i < data_folders.size(); i++) {
      fs::path p(data_folders[i]);
      remaining_folders.push(p);
    }

    while(! remaining_folders.empty()) {
      fs::path latest = remaining_folders.front();
      remaining_folders.pop();
      for (const auto & entry : fs::directory_iterator(latest)) {
        fs::path entry_path = entry.path();
        if(entry.is_directory()) {
          remaining_folders.push(entry_path);
        }
        else {
          tar_list.push_back(entry_path.string());
        }
      }
    }

    sort(tar_list.begin(), tar_list.end());
    auto it = unique(tar_list.begin(), tar_list.end());
    tar_list.resize(distance(tar_list.begin(), it));
  } 

  void process_caida_data(vector<string> &data_folders) {
    int bar_shown = 0;
    int files_processed = 0;
    int num_files = tar_list.size();

    progressbar bar(num_files);

    #pragma omp parallel for schedule(dynamic, 1)
    for(int i = 0; i < num_files; i++) {
      read_tarfile(tar_list[i], i);

      #pragma omp atomic
      files_processed++;

      int threadnum = omp_get_thread_num();
      if(threadnum == 0) {
        int fp_capture;
        #pragma omp atomic read
        fp_capture = files_processed;
        while(bar_shown < fp_capture) {
          bar_shown++;
          bar.update();
        } 
      }
    } 
  }
};

int main (int argc, char** argv)
{
    vector<string> data_folders;
    for(int i = 1; i < argc; i++) {
      data_folders.emplace_back(argv[i]);
    }

    CAIDA_Reader reader(data_folders);
}
