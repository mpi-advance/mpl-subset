#include <mpi.h>
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
            int win(int size, MPI_Info info, MPI_Comm comm, void* baseptr, T* base);                                // selector
            void attach(Window<T> win, const T &data, int size);
            void detach(Window<T> win, const T &data);
            
            //communication
            void put(const T &data, int dest, Window<T> win);
            void get(const T &data, int dest, Window<T> win);
            template<typename T, typename F> void accumulate(const T &data, int dest, F op, Window<T> win);
            template<typename T, typename F> void get_accumulate(const T &data, const T &result, int dest, F op, Window<T> win);
            template<typename T, typename F> void fetch_and_op(const T &data, const T &result, int dest, F op, Window<T> win);
            void compare_and_swap(const T &data, const T &compare, const T &result, int dest, Window<T> win);
            impl::irequest rput(const T &data, int dest, Window<T> win);
            impl::irequest rget(const T &data, int dest, Window<T> win);
            template<typename T, typename F> impl::irequest racc(const T &data, int dest, F op, Window<T> win);
            template<typename T, typename F> impl::irequest rget_acc(const T &data, const T &result, int dest, F op, Window<T> win);

            //synchronization
            void flush(int dest, Window<T> win);
            void flush_all(Window<T> win);
            void flush_local(int dest, Window<T> win);
            void flush_local_all(Window<T> win);
            void fence(int assert, Window<T> win);
            void lock(int lock_type, int dest, int assert, Window<T> win);
            void lock_all(int assert, Window<T> win);
            void unlock(int dest, Window<T> win);
            void unlock_all(Window<T> win);
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
    void Window<T, mode, share_flag>::attach(Window<T> win, const T &data, int size)
    {
        MPI_Win_attach(win, &data, size);
    }

    template<class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::detach(Window<T> win, const T &data)
    {
        MPI_Win_detach(win, &data);
    }

    // communication

    template<class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::put(const T &data, int dest, Window<T> win)
    {
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Put(&data, 1, T, dest, disp, 1, T, win);
    }

    template <class T, int mode, int share_flag>
    impl::irequest Window<T, mode, share_flag>::rput(const T &data, int dest, Window<T> win)
    {
        MPI_Request req;
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Rput(&data, 1, T, dest, disp, 1, T, win, &req);
        return impl::irequest(req);
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::get(const T &data, int dest, Window<T> win)
    {
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Get(&data, 1, T, dest, disp, 1, T, win);
    }

    template <class T, int mode, int share_flag>
    impl::irequest Window<T, mode, share_flag>::rget(const T &data, int dest, Window<T> win)
    {
        MPI_Request req;
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Rget(&data, 1, T, dest, disp, 1, T, win, &req);
        return impl::irequest(req);
    }

    template <typename T, typename F>
    void accumulate(const T &data, int dest, F op, Window<T> win)
    {
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Accumulate(&data, 1, T, dest, disp, 1, T, detail::get_op<T, F>(f).mpi_op, win);
    }

    template <typename T, typename F>
    impl::irequest racc(const T &data, int dest, F op, Window<T> win)
    {
        MPI_Request req;
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Raccumulate(&data, 1, T, dest, disp, 1, T, detail::get_op<T, F>(f).mpi_op, win, &req);
        return impl::irequest(req);
    }

    template <typename T, typename F>
    void get_accumulate(const T &data, const T &result, int dest, F op, Window<T> win)
    {
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Get_accumulate(&data, 1, T, &result, 1, T, dest, disp, 1, T, detail::get_op<T, F>(f).mpi_op, win);
    }

    template <typename T, typename F>
    impl::irequest rget_acc(const T &data, const T &result, int dest, F op, Window<T> win)
    {
        MPI_Request req;
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Rget_accumulate(&data, 1, T, &result, 1, T, dest, disp, 1, T, detail::get_op<T, F>(f).mpi_op, win, &req);
        return impl::irequest(req);
    }

    template <typename T, typename F>
    void fetch_and_op(const T &data, const T &result, int dest, F op, Window<T> win)
    {
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Fetch_and_op(&data, &result, T, dest, disp, detail::get_op<T, F>(f).mpi_op, win);
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::compare_and_swap(const T &data, const T &compare, const T &result, int dest, Window<T> win)
    {
        int disp = sizeof(detail::datatype_traits<T>::get_datatype(&data));
        MPI_Compare_and_swap(&data, &compare, &result, T, dest, disp, win);
    }
    
    // synchronization

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::flush(int dest, Window<T> win)
    {
        MPI_Win_flush(dest, win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::flush_all(Window<T> win)
    {
        MPI_Win_flush_all(win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::flush_local(int dest, Window<T> win)
    {
        MPI_Win_flush_local(dest, win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::flush_local_all(Window<T> win)
    {
        MPI_Win_flush_local_all(win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::fence(int assert, Window<T> win)
    {
        MPI_Win_fence(assert, win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::lock(int lock_type, int dest, int assert, Window<T> win)
    {
        MPI_Win_lock(lock_type, dest, assert, win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::lock_all(int assert, Window<T> win)
    {
        MPI_Win_lock_all(assert, win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::unlock(int dest, Window<T> win)
    {
        MPI_Win_unlock(dest, win)
    }

    template <class T, int mode, int share_flag>
    void Window<T, mode, share_flag>::unlock_all(Window<T> win)
    {
        MPI_Win_unlock_all(win)
    }

    class RMA
    {
        private:
            Window<class T> win;
            
    };
}
