#include <mpi.h>
#include <Kokkos_Core.hpp>

namespace mpl 
{
    template<typename T>
    concept is_layout_type = std::is_same<T, Kokkos::LayoutLeft>::value ||
                         std::is_same<T, Kokkos::LayoutRight>::value ||
                         std::is_same<T, Kokkos::LayoutStride>::value;

    template<typename T>
    concept is_memory_space = std::is_same<T, Kokkos::HostSpace>::value;
    // add other supported memory space types, cant figure out what they all are but should be easy to add

    template<typename T> 
    concept is_memory_trait = std::is_same<T, Kokkos::MemoryManaged>::value ||
                            std::is_same<T, Kokkos::MemoryUnmanaged>::value ||
                            std::is_same<T, Kokkos::MemoryRandomAccess>::value;

    template<class T, int a, is_layout_type LayoutType, is_memory_space MemorySpace, is_memory_trait MemoryTrait>
    class kokkos_view
    {
        public:
            Kokkos::View<T*> kksWindow(Kokkos::View<T*>& v);
            Kokkos::View<T*, LayoutType> kksWindow(Kokkos::View<T*, LayoutType>& v);
            Kokkos::View<T*, LayoutType, MemorySpace> kksWindow(Kokkos::View<T*, LayoutType, MemorySpace>& v);
            Kokkos::View<T*, LayoutType, MemorySpace, MemoryTrait> kksWindow(Kokkos::View<T*, LayoutType, MemorySpace, MemoryTrait>& v);
            Kokkos::View<T*, MemorySpace, MemoryTrait> kksWindow(Kokkos::View<T*, MemorySpace, MemoryTrait>& v);
            Kokkos::View<T*, MemorySpace> kksWindow(Kokkos::View<T*, MemorySpace>& v);
            Kokkos::View<T*, MemoryTrait> kksWindow(Kokkos::View<T*, MemoryTrait>& v);

            Kokkos::View<T[a]> kksWindow(Kokkos::View<T[a]>& v);
            Kokkos::View<T[a], LayoutType> kksWindow(Kokkos::View<T[a], LayoutType>& v);
            Kokkos::View<T[a], LayoutType, MemorySpace> kksWindow(Kokkos::View<T[a], LayoutType, MemorySpace>& v);
            Kokkos::View<T[a], LayoutType, MemorySpace, MemoryTrait> kksWindow(Kokkos::View<T[a], LayoutType, MemorySpace, MemoryTrait>& v);
            Kokkos::View<T[a], MemorySpace, MemoryTrait> kksWindow(Kokkos::View<T[a], MemorySpace, MemoryTrait>& v);
            Kokkos::View<T[a], MemorySpace> kksWindow(Kokkos::View<T[a], MemorySpace>& v);
            Kokkos::View<T[a], MemoryTrait> kksWindow(Kokkos::View<T[a], MemoryTrait>& v);
    }


    //view<dataype>
    template<typename T>
    Kokkos::View<T*> kksWindow(Kokkos::View<T*>& v) { 
    return v;
    };

    //view<datatype, layout>
    template<typename T, is_layout_type LayoutType>
    Kokkos::View<T*, LayoutType> kksWindow(Kokkos::View<T*, LayoutType>& v) { 
    return v;
    };

    //view<datatype, layout, space>
    template<typename T, is_layout_type LayoutType, is_memory_space MemorySpace>
    Kokkos::View<T*, LayoutType, MemorySpace> kksWindow(Kokkos::View<T*, LayoutType, MemorySpace>& v) { 
    return v;
    };

    //view<datatype, layout, space, memory traits>
    template<typename T, is_layout_type LayoutType, is_memory_space MemorySpace, is_memory_trait MemoryTrait>
    Kokkos::View<T*, LayoutType, MemorySpace, MemoryTrait> kksWindow(Kokkos::View<T*, LayoutType, MemorySpace, MemoryTrait>& v) { 
    return v;
    };

    //view<datatype, space, trait>
    template<typename T, is_memory_space MemorySpace, is_memory_trait MemoryTrait>
    Kokkos::View<T*, MemorySpace, MemoryTrait> kksWindow(Kokkos::View<T*, MemorySpace, MemoryTrait>& v) { 
    return v;
    };

    //view<datatype, space>
    template<typename T, is_memory_space MemorySpace>
    Kokkos::View<T*, MemorySpace> kksWindow(Kokkos::View<T*, MemorySpace>& v) { 
    return v;
    };

    //view<datatype, trait>
    template<typename T, is_memory_trait MemoryTrait>
    Kokkos::View<T*, MemoryTrait> kksWindow(Kokkos::View<T*, MemoryTrait>& v) { 
    return v;
    };

    //------SAME THING BUT FOR COMPILE TIME/MIXED DIM--------------------------------------------------------

    // works for compile time dims, also accepts mixed dims
    template<typename T, int a>
    Kokkos::View<T[a]> kksWindow(Kokkos::View<T[a]>& v) {
    return v;
    };

    //view<datatype, layout>
    template<typename T, int a, is_layout_type LayoutType>
    Kokkos::View<T[a], LayoutType> kksWindow(Kokkos::View<T[a], LayoutType>& v) { 
    return v;
    };

    //view<datatype, layout, space>
    template<typename T, int a, is_layout_type LayoutType, is_memory_space MemorySpace>
    Kokkos::View<T[a], LayoutType, MemorySpace> kksWindow(Kokkos::View<T[a], LayoutType, MemorySpace>& v) { 
    return v;
    };

    //view<datatype, layout, space, memory traits>
    template<typename T, int a, is_layout_type LayoutType, is_memory_space MemorySpace, is_memory_trait MemoryTrait>
    Kokkos::View<T[a], LayoutType, MemorySpace, MemoryTrait> kksWindow(Kokkos::View<T[a], LayoutType, MemorySpace, MemoryTrait>& v) { 
    return v;
    };

    //view<datatype, space, trait>
    template<typename T, int a, is_memory_space MemorySpace, is_memory_trait MemoryTrait>
    Kokkos::View<T[a], MemorySpace, MemoryTrait> kksWindow(Kokkos::View<T[a], MemorySpace, MemoryTrait>& v) { 
    return v;
    };

    //view<datatype, space>
    template<typename T, int a, is_memory_space MemorySpace>
    Kokkos::View<T[a], MemorySpace> kksWindow(Kokkos::View<T[a], MemorySpace>& v) { 
    return v;
    };

    //view<datatype, trait>
    template<typename T, int a, is_memory_trait MemoryTrait>
    Kokkos::View<T[a], MemoryTrait> kksWindow(Kokkos::View<T[a], MemoryTrait>& v) { 
    return v;
    };

}
