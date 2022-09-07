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

using namespace std;
namespace fs = std::filesystem;


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

int main (int argc, char** argv)
{

    std::string path = "/global/project/projectdirs/m1982/vbharadw/rdist_tensor/experimental/graphblas_io/build";
    for (const auto & entry : fs::directory_iterator(path)) {
        std::cout << entry.path() << std::endl;
    }

    GrB_init (GrB_NONBLOCKING);

    std::string filename(argv[1]);

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size))
    {
      struct posix_header* header =
        (struct posix_header*) buffer.data();
        std::string filename(header->name);
        // In case we need the filename for something

        bitset<8> size_highbits(*(header->size)); 

        assert(size_highbits[0] == 0);
        int size_of_file = octal_string_to_int(header->size, 11); 
        char* contents = buffer.data() + 512;

        //GrB_Descriptor desc;

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

    }

    GrB_finalize ( ) ;
}
