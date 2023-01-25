#include <mpi.h>
#include <iostream>
#include <vector>

using namespace std;

#ifndef NITERS 
#define NITERS 1000 // default number of iterations 
#endif

#ifndef SKIP
#define SKIP 10 // default number of iterations to skip when timing
#endif

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Win win; 

    vector<int> vec(1000, 1);

    // timing vars
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

    MPI_Win_create(vec.data(), vec.size() * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);    // create window

    // fill vector with data
    if (rank == 0)
    {
        for (int i = 0; i < 1000; i++)
        {
            vec[i] = i;
        }
    }

    for (int i = 0; i < NITERS; i++) // for set # of iterations
    {
        if (i < SKIP) // skip first 10 iterations (default)
        {
            start = MPI_Wtime(); // start time
        }

        MPI_Win_fence(MPI_MODE_NOPRECEDE, win);

        if(rank == 1)
        {
            MPI_Accumulate(vec.data(), vec.size(), MPI_INT, 0, 0, 1000, MPI_INT, MPI_SUM, win);  // do rma 
        }
       
        MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

    }

    end = MPI_Wtime(); // end time

    // calculate latency/bw
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

    // print values for correctness check if desired
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
