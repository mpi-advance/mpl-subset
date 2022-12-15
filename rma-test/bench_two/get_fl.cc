#include <mpi.h>
#include <iostream>
#include <vector>

using namespace std;

#ifndef NITERS 
#define NITERS 1000
#endif

#ifndef SKIP
#define SKIP 10
#endif

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Win win; 

    vector<int> vec(1000, 0);

    MPI_Win_create(vec.data(), vec.size() * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);    

    if (rank == 0)
    {
        for (int i = 0; i < 1000; i++)
        {
            vec[i] = i;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0)
    {
        MPI_Get(vec.data(), vec.size(), MPI_INT, 1, 0, 1000, MPI_INT, win); 
        
    }  
    MPI_Win_flush(1, win);                    

    cout << vec[50];

    MPI_Finalize();
}
