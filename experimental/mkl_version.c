#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
 
int main(void)
  {
    MKLVersion Version;
 
    mkl_get_version(&Version);
 
 
    printf("Major version:           %d\n",Version.MajorVersion);
    printf("Minor version:           %d\n",Version.MinorVersion);
    printf("Update version:          %d\n",Version.UpdateVersion);
    printf("Product status:          %s\n",Version.ProductStatus);
    printf("Build:                   %s\n",Version.Build);
    printf("Platform:                %s\n",Version.Platform);
    printf("Processor optimization:  %s\n",Version.Processor);
    printf("================================================================\n");
    printf("\n");
 
    return 0;
  }
