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
    MPI_Group grp;
    MPI_Comm_group(MPI_COMM_WORLD, &grp);

    //window addresses
    vector<int> vec1(1000, 0);
    vector<int> vec2(1000, 1);   

    //timing 
    vector<double> times;
    double start;
    double end;
    double time;
    double total;
    double avg;
    double min;
    double msg_size;
    double win_size;
    double bw1;

    if(rank == 0)
    { 
        cout << "Vector 1: [ ";
        for (int i = 0; i < 2; i++)
        {
            cout << vec1[i] << " ";
        }
        cout << "]" << endl;
        MPI_Win_create(vec1.data(), vec1.size() * 4, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }
    else if (rank == 1)
    {
        cout << "Vector 2: [ ";
        for (int i = 0; i < 2; i++)
        {
            cout << vec2[i] << " ";
        }
        cout << "]" << endl;
        MPI_Win_create(NULL, vec2.size() * 4, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i=0; i < 1000; i++)
    {
        if(rank == 0)
        {
            start = MPI_Wtime();
        }

        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);

        if(rank == 1)
        {
            MPI_Put(vec2.data(), 1, MPI_INT, 0, i, 1, MPI_INT, win);
        }

        MPI_Win_unlock(0, win);

        if(rank == 0)
        {
            end = MPI_Wtime();
            time = end - start;
            times.push_back(time);
        }
        
    }

    /* if (rank == 0)
    {
        cout << "Vector 1 after put: [ ";
        for (int i = 0; i < vec1.size(); i++)
            {
                cout << vec1[i] << " ";
            }
            cout << "]" << endl;
    } */

    if(rank == 0)
    {
        total = 0; 
        for(int i = 0; i < times.size(); i++)
        {
            total += times[i];
        }
        avg = total / times.size();
        cout << "Avg of 1000 runs: " << avg * 1000 << "s" << endl;
        min = *min_element(times.begin(), times.end());
        cout << "Min of 1000 runs: " << min * 1000 << "s" << endl;

        msg_size = sizeof(int);
        win_size = vec1.size();
        bw1 = msg_size / 1e6 * win_size;
        cout << "Bandwidth: " << bw1/avg << endl;
    }

    MPI_Win_free(&win);
    MPI_Finalize();
}
