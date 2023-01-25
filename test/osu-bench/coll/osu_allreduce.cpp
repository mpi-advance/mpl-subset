#define BENCHMARK "OSU MPI%s Allreduce Latency Test"
/*
 * Copyright (C) 2002-2016 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include "osu_coll.h"

#if defined(USE_MPL_CXX)
#include <mpl/mpl.hpp>
#endif

int main(int argc, char *argv[])
{
    int i, numprocs, rank, size;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double timer=0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0;
    float *sendbuf, *recvbuf;
    int po_ret;
    size_t bufsize;

    set_header(HEADER);
    set_benchmark_name("osu_allreduce");
    enable_accel_support();
    po_ret = process_options(argc, argv);

    if (po_okay == po_ret && none != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }
#if defined(USE_MPL_CXX)
    const mpl::communicator &comm_world(mpl::environment::comm_world());
    numprocs = comm_world.size();
    rank = comm_world.rank();
#else
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
#endif

    switch (po_ret) {
        case po_bad_usage:
            print_bad_usage_message(rank);
#if defined(USE_MPL_CXX)
            return EXIT_FAILURE;
#else
            MPI_Finalize();
            exit(EXIT_FAILURE);
#endif
        case po_help_message:
            print_help_message(rank);
#if defined(USE_MPL_CXX)
            return EXIT_SUCCESS;
#else
            MPI_Finalize();
            exit(EXIT_SUCCESS);
#endif
        case po_version_message:
            print_version_message(rank);
#if defined(USE_MPL_CXX)
            return EXIT_SUCCESS;
#else
            MPI_Finalize();
            exit(EXIT_SUCCESS);
#endif
        case po_okay:
            break;
    }

    if(numprocs < 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires at least two processes\n");
        }
#if defined(USE_MPL_CXX)
        return EXIT_SUCCESS;
#else
        MPI_Finalize();
        exit(EXIT_FAILURE);
#endif
    }

    if (options.max_message_size > options.max_mem_limit) {
        options.max_message_size = options.max_mem_limit;
    }

    options.min_message_size /= sizeof(float);
    if (options.min_message_size < DEFAULT_MIN_MESSAGE_SIZE) {
        options.min_message_size = DEFAULT_MIN_MESSAGE_SIZE;
    }

    bufsize = sizeof(float)*(options.max_message_size/sizeof(float));
    if (allocate_buffer((void**)&sendbuf, bufsize, options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    set_buffer(sendbuf, options.accel, 1, bufsize);

    bufsize = sizeof(float)*(options.max_message_size/sizeof(float));
    if (allocate_buffer((void**)&recvbuf, bufsize, options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    set_buffer(recvbuf, options.accel, 0, bufsize);

    print_preamble(rank);

    for(size=options.min_message_size; size*sizeof(float) <= options.max_message_size; size *= 2) {
#if defined(USE_MPL_CXX)
        mpl::contiguous_layout<float> l(size); 
#endif 
        if(size > LARGE_MESSAGE_SIZE) {
            options.skip = options.skip_large;
            options.iterations = options.iterations_large;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        timer=0.0;
        for(i=0; i < options.iterations + options.skip ; i++) {
            t_start = MPI_Wtime();
#if defined(USE_MPL_CXX)
            comm_world.allreduce(mpl::plus<float>(), sendbuf, recvbuf, l);
#else
            MPI_Allreduce(sendbuf, recvbuf, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD );
#endif 
            t_stop=MPI_Wtime();
            if(i>=options.skip){

            timer+=t_stop-t_start;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        latency = (double)(timer * 1e6) / options.iterations;

        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0,
                MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
        avg_time = avg_time/numprocs;

        print_stats(rank, size * sizeof(float), avg_time, min_time, max_time);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free_buffer(sendbuf, options.accel);
    free_buffer(recvbuf, options.accel);
#if defined(USE_MPL_CXX)
#else
    MPI_Finalize();
#endif

    if (none != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}