#include <iostream>
#include <mpi.h>
#include <vector>
#include <bits/stdc++.h>

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

    vector<int> vec1(1000, 1);
    vector<int> vec2(1000, 2);
    vector<int> vec3(1000);    // result

    //timing 
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

    if(rank == 0)
    {      
        MPI_Win_create(vec1.data(), vec1.size() * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }
    else if (rank == 1)
    {
        MPI_Win_create(vec2.data(), vec2.size() * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }

    #ifdef DEBUG
    if(rank == 0)
    {      
        cout << "Vector 1: [ ";
        for (int i = 0; i < 2; i++)
        {
            cout << vec1[i] << " ";
        }
        cout << "]" << endl;
    }
    else if (rank == 1)
    {
        cout << "Vector 2: [ ";
        for (int i = 0; i < 2; i++)
        {
            cout << vec2[i] << " ";
        }
        cout << "]" << endl;
    }
    #endif

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i=0; i < NITERS; i++)
    {
        if(i< SKIP)
        {
            start = MPI_Wtime();
        }

        MPI_Win_fence(MPI_MODE_NOPRECEDE, win);

        if(rank == 0)
        {
            MPI_Fetch_and_op(vec1.data(), vec3.data(), MPI_INT, 1, i, MPI_SUM, win);
        }

        MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
        MPI_Win_sync(win);        
    }    
    end = MPI_Wtime();

    #ifdef DEBUG
    if(rank == 1)
    {
        cout << "Vector 1 after fetch and op: [ ";
        for (int i = 0; i < 5; i++)
            {
                cout << vec1[i] << " ";
            }
            cout << "]" << endl;
    
        cout << "Vector 2 after fetch and op: [ ";
        for (int i = 0; i < 5; i++)
            {
                cout << vec2[i] << " ";
            }
            cout << "]" << endl;
    }
    #endif

    MPI_Barrier(MPI_COMM_WORLD);
        
	total = end - start;
	avg = total / (NITERS - SKIP);

	msg_size = sizeof(int);
	win_size = vec1.size();
	double tmp = msg_size / 1e6 * win_size;
	bw = tmp / avg; 

	MPI_Reduce(&avg, &combinedavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bw, &combinedbw, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout <<"Average time taken by one process (ms): " << combinedavg * 1000 << endl;
        cout << "Average bandwidth: " << (combinedbw / 2) << endl;
    } 

    MPI_Win_free(&win);
    MPI_Finalize();
}
