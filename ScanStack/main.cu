/*

nvcc --extended-lambda -lineinfo -res-usage -arch=sm_86 -std=c++17 main.cu && ./a.out > main_Output.txt

*/

#include "Push_Pop_Seperately.h"
#include "Read_Files.h"
#include "One_push_Pop_Random.h"
#include "X_Push_Y_Pop.h"

int main()
{
    // push_Pop_Seperately();

    // Read_Files();

    One_push_Pop_Random();

    // X_Push_Y_Pop();

    return 0;
}