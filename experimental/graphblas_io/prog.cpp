extern "C" {
  #include <GraphBLAS.h>
}

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <bitset>
#include <cassert>
#include <filesystem>
#include <queue>
#include <omp.h>
#include "progressbar.hpp"

using namespace std;
namespace fs = std::filesystem;

#define BLOCKSIZE 512

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

/*class CAIDA_Reader {
    uint64_t total_nnz;

    CAIDA_Reader() {

    }
};*/

int read_graphblas_file(char* buf) {
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
            ); // desc

        GrB_Index nrows;
        GrB_Index ncols;
        GrB_Index nvals;
        GrB_Matrix_nrows(&nrows, C);
        GrB_Matrix_nrows(&ncols, C);
        GrB_Matrix_nvals(&nvals, C);

        vector<GrB_Index> I(nvals, 0);
        vector<GrB_Index> J(nvals, 0);
        vector<uint32_t> V(nvals, 0);

        GrB_Matrix_extractTuples_UINT32(
          I.data(),
          J.data(),
          V.data(),
          &nvals,
          C
        );

        GrB_Matrix_free(&C);

        // We will keep this as a multiple of 512 
        int bytes_parsed = BLOCKSIZE 
            + divide_and_roundup(size_of_file, BLOCKSIZE) * BLOCKSIZE;

        return bytes_parsed;
}

void read_tarfile(string path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);

    if (file.read(buffer.data(), size))
    {
        bool continue_parsing = true;
        int position = 0;

        for(int i = 0; i < 64; i++) {
          int bytes_parsed = read_graphblas_file(buffer.data() + position);
          position += bytes_parsed;
        }
    }
}

vector<string> get_tarfile_list(string &base_folder) {
  queue<fs::path> remaining_folders;
  fs::path base_path(base_folder);
  remaining_folders.push(base_path);

  vector<string> tarfile_list;

  while(! remaining_folders.empty()) {
    fs::path latest = remaining_folders.front();
    remaining_folders.pop();
    for (const auto & entry : fs::directory_iterator(latest)) {
      fs::path entry_path = entry.path();
      if(entry.is_directory()) {
        remaining_folders.push(entry_path);
      }
      else {
        tarfile_list.push_back(entry_path.string());
      }
    }
  }

  return tarfile_list;
} 

int main (int argc, char** argv)
{
    GrB_init (GrB_NONBLOCKING);
    std::string foldername(argv[1]);
    vector<string> tar_list = get_tarfile_list(foldername);

    int bar_shown = 0;
    int files_processed = 0;
    int num_files = tar_list.size();

    progressbar bar(num_files);

    #pragma omp parallel for
    for(int i = 0; i < num_files; i++) {
      read_tarfile(tar_list[i]);

      #pragma omp atomic 
      files_processed++;

      int threadnum = omp_get_thread_num();
      if(threadnum == 0) {
        #pragma omp atomic read
        int fp_capture = files_processed;
        while(bar_shown < fp_capture) {
          bar_shown++;
          bar.update();
        } 
      }
    } 

    GrB_finalize ();
}
