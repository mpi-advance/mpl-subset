#include <iostream>
#include <mpi.h> // included only until window creation is sorted out. should be fully encompassed in MPL at that point
#include <vector>
#include <mpl/mpl.hpp> // mpl header file (need to connect rma to this)
#include <cstdlib>

using namespace std;

#ifndef SIZE
#define SIZE 100
#endif

int main()
{
    const mpl::communicator &comm_world(mpl::environment::comm_world());

    vector<int> v1(SIZE, 1);
    vector<int> v2(SIZE, 2);
    int sum = 0;

    if(comm_world.size() == 2)
    {
        if(comm_world.rank() == 0)
        {
            for (int i = 0; i<SIZE; i++)
            {
                sum += v1[i];
            }
        }
        else if (comm_world.rank() == 1)
        {
            for (int i = 0; i<SIZE; i++)
            {
                sum += v2[i];
            }
        }

        
    }
}

