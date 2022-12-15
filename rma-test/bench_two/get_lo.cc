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

    // timing 
    double start;
    double end;
    double time;
    double total;
    double avg;
    double msg_size;
    double win_size;
    double bw;
    double combinedavg;
    double combinedbw;

    MPI_Win_create(vec.data(), vec.size() * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);    

    if (rank == 1)
    {
        for (int i = 0; i < 1000; i++)
        {
            vec[i] = i;
        }
    }

    for (int i = 0; i < NITERS; i++)
    {

        if (i < SKIP)
        {
            start = MPI_Wtime();
        }
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 1, 0, win);  

        if (rank == 0)
        {               
            MPI_Get(vec.data(), vec.size(), MPI_INT, 1, 0, 1000, MPI_INT, win);          
        }

        MPI_Win_unlock(1, win);

        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    end = MPI_Wtime();

    total = end - start;
	avg = total / (NITERS - SKIP);

	msg_size = sizeof(int);
	win_size = vec.size();
	double tmp = msg_size / 1e6 * win_size;
	bw = tmp / avg; 

	MPI_Reduce(&avg, &combinedavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bw, &combinedbw, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0)
    {
        cout <<"Average time taken by one process (ms): " << combinedavg * 1000 << endl;
        cout << "Average bandwidth: " << (combinedbw / 2) << endl;
    } 

    #ifdef DEBUG
    {
        cout << "vec at rank " << rank << ":"; 
            for (int i = 0; i < 20; i++)
            {
                cout << vec[i] << " ";
            }
        cout << endl;
    }
    #endif

    MPI_Win_free(&win);
    MPI_Finalize();
}
