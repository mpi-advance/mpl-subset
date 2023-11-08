#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <kokkos.hpp>
#include <request.hpp>
#include <operator.hpp>

#define DYNAMIC 1
#define STATIC 0
#define EMPTY -1
#define SHARED 1
#define UNSHARED 0

namespace mpl
{
    // window
    template<class T, int mode = STATIC, int share_flag = UNSHARED>
    class Window
    {
        public: 
            // window
            int win(int size, MPI_Info info, MPI_Comm comm, void* baseptr, T* data);     //passed comm must be correct (same as used to create communicator)   // selector
            void attach(Window<T> win, const T &data, int size);
            void detach(Window<T> win, const T &data);            
            
            //communication
            void put(const T &data, int dest, const T &win);
            void get(const T &data, int dest, const T &win);
            template<typename T, typename F> void accumulate(const T &data, int dest, F op, const T &win);
            template<typename T, typename F> void get_accumulate(const T &data, const T &result, int dest, F op, const T &win);
            template<typename T, typename F> void fetch_and_op(const T &data, const T &result, int dest, F op, const T &win);
            void compare_and_swap(const T &data, const T &compare, const T &result, int dest, const T &win);
            impl::irequest rput(const T &data, int dest, const T &win);
            impl::irequest rget(const T &data, int dest, const T &win);
            template<typename T, typename F> impl::irequest racc(const T &data, int dest, F op, const T &win);
            template<typename T, typename F> impl::irequest rget_acc(const T &data, const T &result, int dest, F op, const T &win);

            //synchronization
            void flush(int dest, const T &win);
            void flush_all(const T &win);
            void flush_local(int dest, const T &win);
            void flush_local_all(const T &win);
            void fence(int assert, const T &win);
            void lock(int lock_type, int dest, int assert, const T &win);
            void lock_all(int assert, const T &win);
            void unlock(int dest, const T &win);
            void unlock_all(const T &win);
        private:
            int win_static(const T &data, int size, MPI_Info info, MPI_Comm comm);                                                  // overloaded! win create
            int win_static(int size, MPI_Info info, MPI_Comm comm, void* baseptr, int share_flag);                                  // overloaded! win allocate & allocate shared                                           
            int win_dynamic(MPI_Comm comm, MPI_Info info);                                                                          // win_create_dynamic
            int flag = mode;
            MPI_Win win_address;
    };

    // window creation

    template<class T, int mode, int share_flag>
    int Window<T, mode, share_flag>::win(int size, MPI_Info info = MPI_INFO_NULL, MPI_Comm comm = MPI_COMM_WORLD, void* baseptr = EMPTY, T* data = EMPTY)                          //window creation method selection
    {
        if(flag == DYNAMIC)
        {
            win_dynamic(info, comm);
        }
        else
        {
            if(baseptr != EMPTY)
            {
                win_static(data, size);
            }
            else
            {
                win_static(size, info, comm, baseptr, share_flag);
            }
        }
    }

    

    template<class T, int mode, int share_flag>
    int Window<T, mode, share_flag>::win_dynamic(MPI_Comm comm, MPI_Info info)
    {
        MPI_Win_create_dynamic(info, comm, &win_address);
    }

    template<class T, int mode, int share_flag>
    int Window<T, mode, share_flag>::win_static(const T &data, int size, MPI_Info info, MPI_Comm comm)
    {
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Win_create(&data, size, disp, info, comm, &win_address)
    }

    template<class T, int mode, int share_flag>
    int Window<T, mode, share_flag>::win_static(int size, MPI_Info info, MPI_Comm comm, void* baseptr, int share_flag)
    {
        int disp_unit = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        if(share_flag == SHARED)
        {
            MPI_Win_allocate_shared(size, disp_unit, info, comm, baseptr, &win_address);
        }
        else
        {
            MPI_Win_allocate(size, disp_unit, info, comm, baseptr, &win_address);
        }
    }

    template<class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::attach(const T &win, const T &data, int size)
    {
        MPI_Win_attach(&win, &data, size);
    }

    template<class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::detach(const T &win, const T &data)
    {
        MPI_Win_detach(&win, &data);
    }

    // communication

    template<class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::put(const T &data, int dest, const T &win)
    {
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Put(&data, 1, T, dest, disp, 1, T, &win);
    }

    template <class T, int mode, int share_flag>
    impl::irequest Window<T, mode, share_flag>::rput(const T &data, int dest, const T &win)
    {
        MPI_Request req;
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Rput(&data, 1, T, dest, disp, 1, T, &win, &req);
        return impl::irequest(req);
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::get(const T &data, int dest, const T &win)
    {
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Get(&data, 1, T, dest, disp, 1, T, &win);
    }

    template <class T, int mode, int share_flag>
    impl::irequest Window<T, mode, share_flag>::rget(const T &data, int dest, const T &win)
    {
        MPI_Request req;
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Rget(&data, 1, T, dest, disp, 1, T, &win, &req);
        return impl::irequest(req);
    }

    template <typename T, typename F>
    void accumulate(const T &data, int dest, F op, const T &win)
    {
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Accumulate(&data, 1, T, dest, disp, 1, T, detail::get_op<T, F>(f).mpi_op, &win);
    }

    template <typename T, typename F>
    impl::irequest racc(const T &data, int dest, F op, const T &win)
    {
        MPI_Request req;
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Raccumulate(&data, 1, T, dest, disp, 1, T, detail::get_op<T, F>(f).mpi_op, &win, &req);
        return impl::irequest(req);
    }

    template <typename T, typename F>
    void get_accumulate(const T &data, const T &result, int dest, F op, const T &win)
    {
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Get_accumulate(&data, 1, T, &result, 1, T, dest, disp, 1, T, detail::get_op<T, F>(f).mpi_op, &win);
    }

    template <typename T, typename F>
    impl::irequest rget_acc(const T &data, const T &result, int dest, F op, const T &win)
    {
        MPI_Request req;
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Rget_accumulate(&data, 1, T, &result, 1, T, dest, disp, 1, T, detail::get_op<T, F>(f).mpi_op, &win, &req);
        return impl::irequest(req);
    }

    template <typename T, typename F>
    void fetch_and_op(const T &data, const T &result, int dest, F op, const T &win)
    {
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Fetch_and_op(&data, &result, T, dest, disp, detail::get_op<T, F>(f).mpi_op, &win);
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::compare_and_swap(const T &data, const T &compare, const T &result, int dest, const T &win)
    {
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Compare_and_swap(&data, &compare, &result, T, dest, disp, &win);
    }
    
    // synchronization

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::flush(int dest, const T &win)
    {
        MPI_Win_flush(dest, &win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::flush_all(const T &win)
    {
        MPI_Win_flush_all(&win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::flush_local(int dest, const T &win)
    {
        MPI_Win_flush_local(dest, &win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::flush_local_all(const T &win)
    {
        MPI_Win_flush_local_all(&win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::fence(int assert, const T &win)
    {
        MPI_Win_fence(assert, &win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::lock(int lock_type, int dest, int assert, const T &win)
    {
        MPI_Win_lock(lock_type, dest, assert, &win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::lock_all(int assert, const T &win)
    {
        MPI_Win_lock_all(assert, &win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::unlock(int dest, const T &win)
    {
        MPI_Win_unlock(dest, &win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::unlock_all(const T &win)
    {
        MPI_Win_unlock_all(&win)
    }

    class RMA
    {
        private:
            Window<class T> win;
            
    };

    template <class T, class U>
    class External
    {
        public:
            int win(MPI_Info info, MPI_Comm comm, Kokkos::View<T>& view);
            void attach(External<T> win, const T &data, int size);
            void detach(External<T> win, const T &data);

            //communication
            void put(const T &data, int dest, const T &win);
            void get(const T &data, int dest, const T &win);
            template<typename T, typename F> void accumulate(const T &data, int dest, F op, const T &win);
            template<typename T, typename F> void get_accumulate(const T &data, const T &result, int dest, F op, const T &win);
            template<typename T, typename F> void fetch_and_op(const T &data, const T &result, int dest, F op, const T &win);
            void compare_and_swap(const T &data, const T &compare, const T &result, int dest, const T &win);
            impl::irequest rput(const T &data, int dest, const T &win);
            impl::irequest rget(const T &data, int dest, const T &win);
            template<typename T, typename F> impl::irequest racc(const T &data, int dest, F op, const T &win);
            template<typename T, typename F> impl::irequest rget_acc(const T &data, const T &result, int dest, F op, const T &win);

            //synchronization
            void flush(int dest, const T &win);
            void flush_all(const T &win);
            void flush_local(int dest, const T &win);
            void flush_local_all(const T &win);
            void fence(int assert, const T &win);
            void lock(int lock_type, int dest, int assert, const T &win);
            void lock_all(int assert, const T &win);
            void unlock(int dest, const T &win);
            void unlock_all(const T &win);
        private:
            MPI_Win win_address;
            U external_class;
            int num_dims;
            std::vector<int> strides;
            int calculate_disp(std::vector<int> displacement_vector, Kokkos::is_array_layout layout); // kokkos::is_array_layout could be replaced with other concept defined in kokkos.hpp
    };

    template<class T, class U>
    int External<T, U>::win(MPI_Info info = MPI_INFO_NULL, MPI_Comm comm = MPI_COMM_WORLD, U& view)         // would be U<T>???
    {
        // check in kokkos.hpp for valid constructor
        // assert()
                
        // path if valid view returned
        Kokkos::View<...> v = kksWindow(view); // kksWindow function in kokkos.hpp, will return view if valid format 

        void* baseptr = v.data(); // baseptr                

        assert(v.rank() > 0);
        std::vector<int> strides_vector;

        // size and stride vector setup        
        int size = 1;
        for(int i = 0; i < v.rank(); i++)
        {
            assert(v.extent(i) > 0);
            size *= v.extent(i);
            strides_vector.push_back(v.extent(i));
        }      
        
        this->num_dims = v.rank();
        this->strides = strides_vector;

        int disp_unit = sizeof(detail::datatype_traits<T>::get_datatype(&data)); // mpi_aint
        MPI_Win_create(&data, size, disp_unit, info, comm, &win_address);
    }

    // takes vector of chosen displacement eg.; <2, 1, 1> and does calculation based on layout type
    // may need to be broken into two functions (one for each layout type) to avoid if statement although
    // i think that may just move the if statement elsewhere in the program
    template<class T, class U>
    int External::calculate_disp(std::vector<int> displacement_vector) //change to size t
    {        
        // left = col, right = row
        if(typeid(this->external_class.layout()) == typeid(Kokkos::LayoutLeft))
        {       
            int offset = this->strides[this->num_dims-2] * displacement_vector[this->num_dims-1];            
            for(int i = this->num_dims - 1; i > 0; i--)
            {
                offset = offset * (displacement_vector[i-1] + this->strides[i-1]);
            }
        }
        else if(typeid(this->external_class.layout()) == typeid(Kokkos::LayoutRight))
        {
            int offset = displacement_vector[1] + (this->strides[1] * displacement_vector[0]);
            for(int i = 2; i < this->num_dims; i++)
            {
                offset = offset * displacement_vector[i] * this->strides[i];
            }
        }
    }

    // non contiguous: translate type to mpi data type and use mpi win allocate
    // is underlying storage for noncontig view contiguous?
    // in constructor: decide if contig or non; if non contig create a derived data type during put

    // if 1d: origin and target can be POD 
    // if nD: consider derived type
    // ex. if 2d, create a derived type for bulk communication

    // 1 make sure src and dest extents match
    // 2 calc displacement

    template<class T, class U>
    void External<T, U>::put(const T &data, int dest, std::vector<int> displacement_vector, const T &win)
    {
        int actual_displacement = calculate_disp(displacement_vector);
        MPI_Put(&data, 1, T, dest, actual_displacement, 1, T, &win);
    }

    template<class T, class U>
    void External<T, U>::attach(const T &win, const T &data, int size)
    {
        MPI_Win_attach(&win, &data, size);
    }

    template<class T, class U>
    void External<T, U>::detach(const T &win, const T &data)
    {
        MPI_Win_detach(&win, &data);
    }

    template<class T, class U>
    impl::irequest External<T, U>::rput(const T &data, int dest, std::vector<int> displacement_vector, const T &win)
    {
        MPI_Request req;
        int disp = calculate_disp(displacement_vector);
        MPI_Rput(&data, 1, T, dest, disp, 1, T, &win, &req);
        return impl::irequest(req);
    }

    template<class T, class U>
    void External<T, U>::get(const T &data, int dest, std::vector<int> displacement_vector, const T &win)
    {
        int disp = calculate_disp(displacement_vector);
        MPI_Get(&data, 1, T, dest, disp, 1, T, &win);
    }

    template<class T, class U>
    impl::irequest External<T, U>::rget(const T &data, int dest, std::vector<int> displacement_vector, const T &win)
    {
        MPI_Request req;
        int disp = calculate_disp(displacement_vector);
        MPI_Rget(&data, 1, T, dest, disp, 1, T, &win, &req);
        return impl::irequest(req);
    }

    template<class T, class U>
    void External<T, U>::accumulate(const T &data, int dest, std::vector<int> displacement_vector, F op, const T &win)
    {
        int disp = calculate_disp(displacement_vector);
        MPI_Accumulate(&data, 1, T, dest, disp, 1, T, detail::get_op<T, F>(f).mpi_op, &win);
    }

    template<class T, class U>
    impl::irequest External<T, U>::racc(const T &data, int dest, std::vector<int> displacement_vector, F op, const T &win)
    {
        MPI_Request req;
        int disp = calculate_disp(displacement_vector);
        MPI_Raccumulate(&data, 1, T, dest, disp, 1, T, detail::get_op<T, F>(f).mpi_op, &win, &req);
        return impl::irequest(req);
    }

    template<class T, class U>
    void External<T, U>::get_accumulate(const T &data, const T &result, int dest, std::vector<int> displacement_vector, F op, const T &win)
    {
        int disp = calculate_disp(displacement_vector);
        MPI_Get_accumulate(&data, 1, T, &result, 1, T, dest, disp, 1, T, detail::get_op<T, F>(f).mpi_op, &win);
    }

    template<class T, class U>
    impl::irequest External<T, U>::rget_acc(const T &data, const T &result, int dest, std::vector<int> displacement_vector, F op, const T &win)
    {
        MPI_Request req;
        int disp = calculate_disp(displacement_vector);
        MPI_Rget_accumulate(&data, 1, T, &result, 1, T, dest, disp, 1, T, detail::get_op<T, F>(f).mpi_op, &win, &req);
        return impl::irequest(req);
    }

    template<class T, class U>
    void External<T, U>::fetch_and_op(const T &data, const T &result, int dest, std::vector<int> displacement_vector, F op, const T &win)
    {
        int disp = calculate_disp(displacement_vector);
        MPI_Fetch_and_op(&data, &result, T, dest, disp, detail::get_op<T, F>(f).mpi_op, &win);
    }

    template<class T, class U>
    void External<T, U>::compare_and_swap(const T &data, const T &compare, const T &result, int dest, std::vector<int> displacement_vector, const T &win)
    {
        int disp = calculate_disp(displacement_vector);
        MPI_Compare_and_swap(&data, &compare, &result, T, dest, disp, &win);
    }

    template<class T, class U>
    void External<T, U>::flush(int dest, const T &win)
    {
        MPI_Win_flush(dest, &win)
    }

    template<class T, class U>
    void External<T, U>::flush_all(const T &win)
    {
        MPI_Win_flush_all(&win)
    }

    template<class T, class U>
    void External<T, U>::flush_local(int dest, const T &win)
    {
        MPI_Win_flush_local(dest, &win)
    }

    template<class T, class U>
    void External<T, U>::flush_local_all(const T &win)
    {
        MPI_Win_flush_local_all(&win)
    }

    template<class T, class U>
    void External<T, U>::fence(int assert, const T &win)
    {
        MPI_Win_fence(assert, &win)
    }

    template<class T, class U>
    void External<T, U>::lock(int lock_type, int dest, int assert, const T &win)
    {
        MPI_Win_lock(lock_type, dest, assert, &win)
    }

    template<class T, class U>
    void External<T, U>::lock_all(int assert, const T &win)
    {
        MPI_Win_lock_all(assert, &win)
    }

    template<class T, class U>
    void External<T, U>::unlock(int dest, const T &win)
    {
        MPI_Win_unlock(dest, &win)
    }

    template<class T, class U>
    void External<T, U>::unlock_all(const T &win)
    {
        MPI_Win_unlock_all(&win)
    }

}
