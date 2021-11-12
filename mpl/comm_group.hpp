#if !(defined MPL_COMM_GROUP_HPP)

#define MPL_COMM_GROUP_HPP

#include <mpi.h>
#include <type_traits>
#include <tuple>
#include <mpl/layout.hpp>

namespace mpl {

  class group;

  class communicator;

  namespace environment {

    namespace detail {

      class env;

    }

  }  // namespace environment

  //--------------------------------------------------------------------

  /// \brief Represents a group of processes.
  class group {
    MPI_Group gr{MPI_GROUP_EMPTY};

  public:
    /// \brief Group equality types.
    enum class equality_type {
      /// groups are identical, i.e., groups have same the members in same rank order
      identical = MPI_IDENT,
      /// groups are similar, i.e., groups have same tha members in different rank order
      similar = MPI_SIMILAR,
      /// groups are unequal, i.e., groups have different sets of members
      unequal = MPI_UNEQUAL
    };

    /// indicates that groups are identical, i.e., groups have same the members in same rank
    /// order
    static constexpr equality_type identical = equality_type::identical;
    /// indicates that groups are similar, i.e., groups have same tha members in different rank
    /// order
    static constexpr equality_type similar = equality_type::similar;
    /// indicates that groups are unequal, i.e., groups have different sets of members
    static constexpr equality_type unequal = equality_type::unequal;

    /// \brief Indicates the creation of a union of two groups.
    class Union_tag {};
    /// \brief Indicates the creation of a union of two groups.
    static constexpr Union_tag Union{};

    /// \brief Indicates the creation of an intersection of two groups.
    class intersection_tag {};
    /// \brief Indicates the creation of an intersection of two groups.
    static constexpr intersection_tag intersection{};

    /// \brief Indicates the creation of a difference of two groups.
    class difference_tag {};
    /// \brief Indicates the creation of a difference of two groups.
    static constexpr difference_tag difference{};

    /// \brief Indicates the creation of a subgroup by including members of an existing group.
    class include_tag {};
    /// \brief Indicates the creation of a subgroup by including members of an existing group.
    static constexpr include_tag include{};

    /// \brief Indicates the creation of a subgroup by excluding members of an existing group.
    class exclude_tag {};
    /// \brief Indicates the creation of a subgroup by excluding members of an existing group.
    static constexpr exclude_tag exclude{};

    /// \brief Creates an empty process group.
    group() = default;

    /// \brief Creates a new process group by copying an existing one.
    /// \param other the other group to copy from
    group(const group &other);

    /// \brief Move-constructs a process group.
    /// \param other the other group to move from
    group(group &&other) noexcept : gr{other.gr} { other.gr = MPI_GROUP_EMPTY; }

    /// \brief Creates a new group that consists of all processes of the given communicator.
    /// \param comm the communicator
    explicit group(const communicator &comm);

    /// \brief Creates a new group that consists of the union of two existing process groups.
    /// \param tag indicates the unification of two existing process groups
    /// \param other_1 first existing process group
    /// \param other_2 second existing process group
    explicit group(Union_tag tag, const group &other_1, const group &other_2);

    /// \brief Creates a new group that consists of the intersection of two existing process
    /// groups.
    /// \param tag indicates the intersection of two existing process groups
    /// \param other_1 first existing process group
    /// \param other_2 second existing process group
    explicit group(intersection_tag tag, const group &other_1, const group &other_2);

    /// \brief Creates a new group that consists of the difference of two existing process
    /// groups.
    /// \param tag indicates the difference of two existing process groups
    /// \param other_1 first existing process group
    /// \param other_2 second existing process group
    explicit group(difference_tag tag, const group &other_1, const group &other_2);

    /// \brief Creates a new group by including members of an existing process group.
    /// \param tag indicates inclusion from an existing process group
    /// \param other existing process group
    /// \param rank set of ranks to include
    //explicit group(include_tag tag, const group &other, const ranks &rank);

    /// \brief Creates a new group by excluding members of an existing process group.
    /// \param tag indicates exclusion from an existing process group
    /// \param other existing process group
    /// \param rank set of ranks to exclude
    //explicit group(exclude_tag tag, const group &other, const ranks &rank);

    /// \brief Destructs a process group.
    ~group() {
      int result;
      MPI_Group_compare(gr, MPI_GROUP_EMPTY, &result);
      if (result != MPI_IDENT)
        MPI_Group_free(&gr);
    }

    /// \brief Copy-assigns a process group.
    /// \param other the other group to move from
    /// \return this group
    group &operator=(const group &other) {
      if (this != &other) {
        int result;
        MPI_Group_compare(gr, MPI_GROUP_EMPTY, &result);
        if (result != MPI_IDENT)
          MPI_Group_free(&gr);
        MPI_Group_excl(other.gr, 0, nullptr, &gr);
      }
      return *this;
    }

    /// \brief Move-assigns a process group.
    /// \param other the other group to move from
    /// \return this group
    group &operator=(group &&other) noexcept {
      if (this != &other) {
        int result;
        MPI_Group_compare(gr, MPI_GROUP_EMPTY, &result);
        if (result != MPI_IDENT)
          MPI_Group_free(&gr);
        gr = other.gr;
        other.gr = MPI_GROUP_EMPTY;
      }
      return *this;
    }

    /// \brief Determines the size of a process group.
    /// \return the size of the group
    [[nodiscard]] int size() const {
      int result;
      MPI_Group_size(gr, &result);
      return result;
    }

    /// \brief Determines the rank within a process group.
    /// \return the rank of the calling process in the group
    [[nodiscard]] int rank() const {
      int result;
      MPI_Group_rank(gr, &result);
      return result;
    }

    /// \brief Determines the relative numbering of the same process in two different groups.
    /// \param rank a valid rank in the given process group
    /// \param other process group
    /// \return corresponding rank in this process group
    [[nodiscard]] int translate(int rank, const group &other) const {
      int other_rank;
      MPI_Group_translate_ranks(gr, 1, &rank, other.gr, &other_rank);
      return other_rank;
    }

    /// \brief Tests for identity of process groups.
    /// \return true if identical
    bool operator==(const group &other) const {
      int result;
      MPI_Group_compare(gr, other.gr, &result);
      return result == MPI_IDENT;
    }

    /// \brief Tests for identity of process groups.
    /// \return true if not identical
    bool operator!=(const group &other) const {
      int result;
      MPI_Group_compare(gr, other.gr, &result);
      return result != MPI_IDENT;
    }

    /// \brief Compares to another process group.
    /// \return equality type
    [[nodiscard]] equality_type compare(const group &other) const {
      int result;
      MPI_Group_compare(gr, other.gr, &result);
      return static_cast<equality_type>(result);
    }

    friend class communicator;
  };

  //--------------------------------------------------------------------

  /// \brief Specifies the communication context for a communication operation.
  class communicator {
    struct isend_irecv_state {
      MPI_Request req{};
      int source{MPI_ANY_SOURCE};
      int tag{MPI_ANY_TAG};
      MPI_Datatype datatype{MPI_DATATYPE_NULL};
      int count{MPI_UNDEFINED};
    };

    static int isend_irecv_query(void *state, MPI_Status *s) {
      isend_irecv_state *sendrecv_state{reinterpret_cast<isend_irecv_state *>(state)};
      MPI_Status_set_elements(s, sendrecv_state->datatype, sendrecv_state->count);
      MPI_Status_set_cancelled(s, 0);
      s->MPI_SOURCE = sendrecv_state->source;
      s->MPI_TAG = sendrecv_state->tag;
      return MPI_SUCCESS;
    }

    static int isend_irecv_free(void *state) {
      isend_irecv_state *sendrecv_state{reinterpret_cast<isend_irecv_state *>(state)};
      delete sendrecv_state;
      return MPI_SUCCESS;
    }

    static int isend_irecv_cancel(void *state, int complete) { return MPI_SUCCESS; }

  protected:
    MPI_Comm comm{MPI_COMM_NULL};

  public:
    /// \brief Equality types for communicator comparison.
    enum class equality_type {
      /// communicators are identical, i.e., communicators represent the same communication
      /// context
      identical = MPI_IDENT,
      /// communicators are identical, i.e., communicators have same the members in same rank
      /// order but different context
      congruent = MPI_CONGRUENT,
      /// communicators are similar, i.e., communicators have same tha members in different rank
      /// order
      similar = MPI_SIMILAR,
      /// communicators are unequal, i.e., communicators have different sets of members
      unequal = MPI_UNEQUAL
    };

    /// indicates that communicators are identical, i.e., communicators represent the same
    /// communication context
    static constexpr equality_type identical = equality_type::identical;
    /// indicates that communicators are identical, i.e., communicators have same the members in
    /// same rank order but different context
    static constexpr equality_type congruent = equality_type::congruent;
    /// indicates that communicators are similar, i.e., communicators have same tha members in
    /// different rank order
    static constexpr equality_type similar = equality_type::similar;
    /// indicates that communicators are unequal, i.e., communicators have different sets of
    /// members
    static constexpr equality_type unequal = equality_type::unequal;

    /// \brief Indicates the creation of a new communicator by a call that in collective for all
    /// processes in the given communicator.
    class comm_collective_tag {};
    /// \brief Indicates the creation of a new communicator by a call that in collective for all
    /// processes in the given communicator.
    static constexpr comm_collective_tag comm_collective{};

    /// \brief Indicates the creation of a new communicator by a call that in collective for all
    /// processes in the given group.
    class group_collective_tag {};
    /// \brief Indicates the creation of a new communicator by a call that in collective for all
    /// processes in the given group.
    static constexpr group_collective_tag group_collective{};

    /// \brief Indicates the creation of a new communicator by spitting an existing communicator
    /// into disjoint subgroups.
    class split_tag {};
    /// \brief Indicates the creation of a new communicator by spitting an existing communicator
    /// into disjoint subgroups.
    static constexpr split_tag split{};

    /// \brief Indicates the creation of a new communicator by spitting an existing communicator
    /// into disjoint subgroups each of which can create a shared memory region.
    class split_shared_memory_tag {};
    /// \brief Indicates the creation of a new communicator by spitting an existing communicator
    /// into disjoint subgroups each of which can create a shared memory region.
    static constexpr split_shared_memory_tag split_shared_memory{};

  private:
    void check_dest(int dest) const {
#if defined MPL_DEBUG
      if (dest != proc_null and (dest < 0 or dest >= size()))
        throw invalid_rank();
#endif
    }

    void check_source(int source) const {
#if defined MPL_DEBUG
      if (source != proc_null and source != any_source and (source < 0 or source >= size()))
        throw invalid_rank();
#endif
    }

    void check_send_tag(tag t) const {
#if defined MPL_DEBUG
      if (static_cast<int>(t) < 0 or static_cast<int>(t) > static_cast<int>(tag::up()))
        throw invalid_tag();
#endif
    }

    void check_recv_tag(tag t) const {
#if defined MPL_DEBUG
      if (static_cast<int>(t) != static_cast<int>(tag::any()) and
          (static_cast<int>(t) < 0 or static_cast<int>(t) > static_cast<int>(tag::up())))
        throw invalid_tag();
#endif
    }

    void check_root(int root_rank) const {
#if defined MPL_DEBUG
      if (root_rank < 0 or root_rank >= size())
        throw invalid_rank();
#endif
    }

    void check_nonroot(int root_rank) const {
#if defined MPL_DEBUG
      if (root_rank < 0 or root_rank >= size() or root_rank == rank())
        throw invalid_rank();
#endif
    }

    template<typename T>
    void check_size(const layouts<T> &l) const {
#if defined MPL_DEBUG
      if (static_cast<int>(l.size()) > size())
        throw invalid_size();
#endif
    }

    void check_count(int count) const {
#if defined MPL_DEBUG
      if (count == MPI_UNDEFINED)
        throw invalid_count();
#endif
    }

    template<typename T>
    void check_container_size(const T &container, detail::basic_or_fixed_size_type) const {}

    template<typename T>
    void check_container_size(const T &container, detail::stl_container) const {
#if defined MPL_DEBUG
      if (container.size() > std::numeric_limits<int>::max())
        throw invalid_count();
#endif
    }

    template<typename T>
    void check_container_size(const T &container) const {
      check_container_size(container,
                           typename detail::datatype_traits<T>::data_type_category{});
    }

  protected:
    explicit communicator(MPI_Comm comm) : comm(comm) {}

  public:
    /// \brief Creates an empty communicator with no associated process.
    communicator() = default;

    /// \brief Creates a new communicator which is equivalent to an existing one.
    /// \param other the other communicator to copy from
    communicator(const communicator &other) { MPI_Comm_dup(other.comm, &comm); }

    /// \brief Move-constructs a communicator.
    /// \param other the other communicator to move from
    communicator(communicator &&other) noexcept : comm{other.comm} {
      other.comm = MPI_COMM_NULL;
    }

    /// \brief Constructs a new communicator from an existing one with a specified communication group.
    /// \param comm_collective tag to indicate the mode of construction
    /// \param other the communicator
    /// \param gr the group that determines the new communicator's structure
    /// \note This is a collective operation that needs to be carried out by all processes of the communicator other.
    explicit communicator(comm_collective_tag comm_collective, const communicator &other, const group &gr) {
      MPI_Comm_create(other.comm, gr.gr, &comm);
    }

    /// \brief Constructs a new communicator from an existing one with a specified communication group.
    /// \param group_collective tag to indicate the mode of construction
    /// \param other the communicator
    /// \param gr the group that determines the new communicator's structure
    /// \param t tag to distinguish between different parallel operations in different threads
    /// \note This is a collective operation that needs to be carried out by all processes of the given group.
    explicit communicator(group_collective_tag group_collective, const communicator &other, const group &gr,
                          tag t = tag(0)) {
      MPI_Comm_create_group(other.comm, gr.gr, static_cast<int>(t), &comm);
    }

    /// \brief Constructs a new communicator from an existing one with a specified communication group.
    /// \param split tag to indicate the mode of construction
    /// \param other the communicator
    /// \param color control of subset assignment
    /// \param key  control of rank assignment
    /// \tparam color_type color type, must be integral type
    /// \tparam key_type key type, must be integral type
    template<typename color_type, typename key_type = int>
    explicit communicator(split_tag split, const communicator &other, color_type color,
                          key_type key = 0) {
      static_assert(detail::is_valid_color_v<color_type>,
                    "not an enumeration type or underlying enumeration type too large");
      static_assert(detail::is_valid_key_v<key_type>,
                    "not an enumeration type or underlying enumeration type too large");
      MPI_Comm_split(other.comm, detail::underlying_type<color_type>::value(color),
                     detail::underlying_type<key_type>::value(key), &comm);
    }

    /// \brief Constructs a new communicator from an existing one by spitting the communicator into disjoint subgroups each of which can create a shared memory region.
    /// \param split_shared_memory tag to indicate the mode of construction
    /// \param other the communicator
    /// \param key  control of rank assignment
    /// \tparam color_type color type, must be integral type
    template<typename key_type = int>
    explicit communicator(split_shared_memory_tag split_shared_memory, const communicator &other,
                          key_type key = 0) {
      static_assert(detail::is_valid_tag_v<key_type>,
                    "not an enumeration type or underlying enumeration type too large");
      MPI_Comm_split_type(other.comm, MPI_COMM_TYPE_SHARED,
                          detail::underlying_type<key_type>::value(key), MPI_INFO_NULL, &comm);
    }

    /// \brief Destructs a communicator.
    ~communicator() {
      if (is_valid()) {
        int result1;
        MPI_Comm_compare(comm, MPI_COMM_WORLD, &result1);
        int result2;
        MPI_Comm_compare(comm, MPI_COMM_SELF, &result2);
        if (result1 != MPI_IDENT and result2 != MPI_IDENT)
          MPI_Comm_free(&comm);
      }
    }

    void operator=(const communicator &) = delete;

    communicator &operator=(communicator &&other) noexcept {
      if (this != &other) {
        if (is_valid()) {
          int result1;
          MPI_Comm_compare(comm, MPI_COMM_WORLD, &result1);
          int result2;
          MPI_Comm_compare(comm, MPI_COMM_SELF, &result2);
          if (result1 != MPI_IDENT and result2 != MPI_IDENT)
            MPI_Comm_free(&comm);
        }
        comm = other.comm;
        other.comm = MPI_COMM_NULL;
      }
      return *this;
    }

    [[nodiscard]] int size() const {
      int result;
      MPI_Comm_size(comm, &result);
      return result;
    }

    [[nodiscard]] int rank() const {
      int result;
      MPI_Comm_rank(comm, &result);
      return result;
    }

    bool operator==(const communicator &other) const {
      int result;
      MPI_Comm_compare(comm, other.comm, &result);
      return result == MPI_IDENT;
    }

    bool operator!=(const communicator &other) const {
      int result;
      MPI_Comm_compare(comm, other.comm, &result);
      return result != MPI_IDENT;
    }

    [[nodiscard]] equality_type compare(const communicator &other) const {
      int result;
      MPI_Comm_compare(comm, other.comm, &result);
      return static_cast<equality_type>(result);
    }

    [[nodiscard]] bool is_valid() const { return comm != MPI_COMM_NULL; }

    void abort(int err) const { MPI_Abort(comm, err); }

    friend class group;

    friend class environment::detail::env;

    // === point to point ==============================================

    // === standard send ===
    // --- blocking standard send ---
  private:
    template<typename T>
    void send(const T &data, int dest, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Send(&data, 1, detail::datatype_traits<T>::get_datatype(), dest, static_cast<int>(t),
               comm);
    }

    template<typename T>
    void send(const T &data, int dest, tag t, detail::contiguous_const_stl_container) const {
      using value_type = typename T::value_type;
      vector_layout<value_type> l(data.size());
      send(data.size() > 0 ? &data[0] : nullptr, l, dest, t);
    }


  public:
    template<typename T>
    void send(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      check_container_size(data);
      send(data, dest, t, typename detail::datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    void send(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Send(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), dest,
               static_cast<int>(t), comm);
    }

    template<typename iterT>
    void send(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        send(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        send(&(*begin), l, dest, t);
      }
    }

    // --- nonblocking standard send ---
  private:
    template<typename T>
    impl::irequest isend(const T &data, int dest, tag t,
                         detail::basic_or_fixed_size_type) const {
      MPI_Request req;
      MPI_Isend(&data, 1, detail::datatype_traits<T>::get_datatype(), dest, static_cast<int>(t),
                comm, &req);
      return impl::irequest(req);
    }

  public:
    template<typename T>
    irequest isend(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      check_container_size(data);
      return isend(data, dest, t, typename detail::datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    irequest isend(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Isend(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), dest,
                static_cast<int>(t), comm, &req);
      return impl::irequest(req);
    }

    template<typename iterT>
    irequest isend(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        return isend(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return isend(&(*begin), l, dest, t);
      }
    }

    // --- persistent standard send ---
    template<typename T>
    prequest send_init(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Send_init(&data, 1, detail::datatype_traits<T>::get_datatype(), dest,
                    static_cast<int>(t), comm, &req);
      return impl::prequest(req);
    }

    template<typename T>
    prequest send_init(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Send_init(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), dest,
                    static_cast<int>(t), comm, &req);
      return impl::prequest(req);
    }

    template<typename iterT>
    prequest send_init(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        return send_init(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return send_init(&(*begin), l, dest, t);
      }
    }

    // === buffered send ===
    // --- determine buffer size ---
    template<typename T>
    [[nodiscard]] int bsend_size() const {
      int pack_size{0};
      MPI_Pack_size(1, detail::datatype_traits<T>::get_datatype(), comm, &pack_size);
      return pack_size + MPI_BSEND_OVERHEAD;
    }

    template<typename T>
    int bsend_size(const layout<T> &l) const {
      int pack_size{0};
      MPI_Pack_size(1, detail::datatype_traits<layout<T>>::get_datatype(l), comm, &pack_size);
      return pack_size + MPI_BSEND_OVERHEAD;
    }

    // --- blocking buffered send ---
  private:
    template<typename T>
    void bsend(const T &data, int dest, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Bsend(&data, 1, detail::datatype_traits<T>::get_datatype(), dest, static_cast<int>(t),
                comm);
    }

    template<typename T>
    void bsend(const T &data, int dest, tag t, detail::contiguous_const_stl_container) const {
      using value_type = typename T::value_type;
      vector_layout<value_type> l(data.size());
      bsend(data.size() > 0 ? &data[0] : nullptr, l, dest, t);
    }

  public:
    /// \anchor communicator_bsend
    template<typename T>
    void bsend(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      check_container_size(data);
      bsend(data, dest, t, typename detail::datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    void bsend(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Bsend(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), dest,
                static_cast<int>(t), comm);
    }

    template<typename iterT>
    void bsend(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        bsend(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        bsend(&(*begin), l, dest, t);
      }
    }

    // --- nonblocking buffered send ---
  private:
    template<typename T>
    irequest ibsend(const T &data, int dest, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Request req;
      MPI_Ibsend(&data, 1, detail::datatype_traits<T>::get_datatype(), dest,
                 static_cast<int>(t), comm, &req);
      return impl::irequest(req);
    }

  public:
    template<typename T>
    irequest ibsend(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      check_container_size(data);
      return ibsend(data, dest, t, typename detail::datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    irequest ibsend(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Ibsend(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), dest,
                 static_cast<int>(t), comm, &req);
      return impl::irequest(req);
    }

    template<typename iterT>
    irequest ibsend(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        return ibsend(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return ibsend(&(*begin), l, dest, t);
      }
    }

    // --- persistent buffered send ---
    template<typename T>
    prequest bsend_init(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Bsend_init(&data, 1, detail::datatype_traits<T>::get_datatype(), dest,
                     static_cast<int>(t), comm, &req);
      return impl::prequest(req);
    }

    template<typename T>
    prequest bsend_init(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Bsend_init(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), dest,
                     static_cast<int>(t), comm, &req);
      return impl::prequest(req);
    }

    template<typename iterT>
    prequest bsend_init(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        return bsend_init(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return bsend_init(&(*begin), l, dest, t);
      }
    }

    // === synchronous send ===
    // --- blocking synchronous send ---
  private:
    template<typename T>
    void ssend(const T &data, int dest, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Ssend(&data, 1, detail::datatype_traits<T>::get_datatype(), dest, static_cast<int>(t),
                comm);
    }

    template<typename T>
    void ssend(const T &data, int dest, tag t, detail::contiguous_const_stl_container) const {
      using value_type = typename T::value_type;
      vector_layout<value_type> l(data.size());
      ssend(data.size() > 0 ? &data[0] : nullptr, l, dest, t);
    }

  public:
    template<typename T>
    void ssend(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      ssend(data, dest, t, typename detail::datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    void ssend(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Ssend(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), dest,
                static_cast<int>(t), comm);
    }

    template<typename iterT>
    void ssend(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        ssend(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        ssend(&(*begin), l, dest, t);
      }
    }

    // --- nonblocking synchronous send ---
  private:
    template<typename T>
    irequest issend(const T &data, int dest, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Request req;
      MPI_Issend(&data, 1, detail::datatype_traits<T>::get_datatype(), dest,
                 static_cast<int>(t), comm, &req);
      return impl::irequest(req);
    }

  public:
    template<typename T>
    irequest issend(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      check_container_size(data);
      return issend(data, dest, t, typename detail::datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    irequest issend(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Issend(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), dest,
                 static_cast<int>(t), comm, &req);
      return impl::irequest(req);
    }

    template<typename iterT>
    irequest issend(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        return issend(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return issend(&(*begin), l, dest, t);
      }
    }

    // --- persistent synchronous send ---
    template<typename T>
    prequest ssend_init(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Ssend_init(&data, 1, detail::datatype_traits<T>::get_datatype(), dest,
                     static_cast<int>(t), comm, &req);
      return impl::prequest(req);
    }

    template<typename T>
    prequest ssend_init(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Ssend_init(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), dest,
                     static_cast<int>(t), comm, &req);
      return impl::prequest(req);
    }

    template<typename iterT>
    prequest ssend_init(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        return ssend_init(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return ssend_init(&(*begin), l, dest, t);
      }
    }

    // === ready send ===
    // --- blocking ready send ---
  private:
    template<typename T>
    void rsend(const T &data, int dest, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Rsend(&data, 1, detail::datatype_traits<T>::get_datatype(), dest, static_cast<int>(t),
                comm);
    }

    template<typename T>
    void rsend(const T &data, int dest, tag t, detail::contiguous_const_stl_container) const {
      using value_type = typename T::value_type;
      vector_layout<value_type> l(data.size());
      rsend(data.size() > 0 ? &data[0] : nullptr, l, dest, t);
    }

  public:
    template<typename T>
    void rsend(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      check_container_size(data);
      rsend(data, dest, t, typename detail::datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    void rsend(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Rsend(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), dest,
                static_cast<int>(t), comm);
    }

    template<typename iterT>
    void rsend(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        rsend(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        rsend(&(*begin), l, dest, t);
      }
    }

    // --- nonblocking ready send ---
  private:
    template<typename T>
    irequest irsend(const T &data, int dest, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Request req;
      MPI_Irsend(&data, 1, detail::datatype_traits<T>::get_datatype(), dest,
                 static_cast<int>(t), comm, &req);
      return impl::irequest(req);
    }

  public:
    template<typename T>
    irequest irsend(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      check_container_size(data);
      return irsend(data, dest, t, typename detail::datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    irequest irsend(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Irsend(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), dest,
                 static_cast<int>(t), comm, &req);
      return impl::irequest(req);
    }

    template<typename iterT>
    irequest irsend(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        return irsend(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return irsend(&(*begin), l, dest, t);
      }
    }

    // --- persistent ready send ---
    template<typename T>
    prequest rsend_init(const T &data, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Rsend_init(&data, 1, detail::datatype_traits<T>::get_datatype(), dest,
                     static_cast<int>(t), comm, &req);
      return impl::prequest(req);
    }

    template<typename T>
    prequest rsend_init(const T *data, const layout<T> &l, int dest, tag t = tag(0)) const {
      check_dest(dest);
      check_send_tag(t);
      MPI_Request req;
      MPI_Rsend_init(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), dest,
                     static_cast<int>(t), comm, &req);
      return impl::prequest(req);
    }

    template<typename iterT>
    prequest rsend_init(iterT begin, iterT end, int dest, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        return rsend_init(&(*begin), l, dest, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return rsend_init(&(*begin), l, dest, t);
      }
    }

    // === receive ===
    // --- blocking receive ---
  private:
    template<typename T>
    status recv(T &data, int source, tag t, detail::basic_or_fixed_size_type) const {
      status s;
      MPI_Recv(&data, 1, detail::datatype_traits<T>::get_datatype(), source,
               static_cast<int>(t), comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    template<typename T>
    status recv(T &data, int source, tag t, detail::contiguous_stl_container) const {
      using value_type = typename T::value_type;
      status s;
      auto *ps{reinterpret_cast<MPI_Status *>(&s)};
      MPI_Message message;
      MPI_Mprobe(source, static_cast<int>(t), comm, &message, ps);
      int count{0};
      MPI_Get_count(ps, detail::datatype_traits<value_type>::get_datatype(), &count);
      check_count(count);
      data.resize(count);
      MPI_Mrecv(data.size() > 0 ? &data[0] : nullptr, count,
                detail::datatype_traits<value_type>::get_datatype(), &message, ps);
      return s;
    }

  public:
    /// \anchor communicator_recv
    template<typename T>
    status recv(T &data, int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      return recv(data, source, t, typename detail::datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    status recv(T *data, const layout<T> &l, int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      status s;
      MPI_Recv(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), source,
               static_cast<int>(t), comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    template<typename iterT>
    status recv(iterT begin, iterT end, int source, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        return recv(&(*begin), l, source, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return recv(&(*begin), l, source, t);
      }
    }

    // --- nonblocking receive ---
  private:
    template<typename T>
    irequest irecv(T &data, int source, tag t, detail::basic_or_fixed_size_type) const {
      MPI_Request req;
      MPI_Irecv(&data, 1, detail::datatype_traits<T>::get_datatype(), source,
                static_cast<int>(t), comm, &req);
      return impl::irequest(req);
    }

  public:
    template<typename T>
    irequest irecv(T &data, int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      return irecv(data, source, t, typename detail::datatype_traits<T>::data_type_category{});
    }

    template<typename T>
    irequest irecv(T *data, const layout<T> &l, int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      MPI_Request req;
      MPI_Irecv(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), source,
                static_cast<int>(t), comm, &req);
      return impl::irequest(req);
    }

    template<typename iterT>
    irequest irecv(iterT begin, iterT end, int source, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        return irecv(&(*begin), l, source, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return irecv(&(*begin), l, source, t);
      }
    }

    // --- persistent receive ---
    template<typename T>
    prequest recv_init(T &data, int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      MPI_Request req;
      MPI_Recv_init(&data, 1, detail::datatype_traits<T>::get_datatype(), source,
                    static_cast<int>(t), comm, &req);
      return impl::prequest(req);
    }

    template<typename T>
    prequest recv_init(T *data, const layout<T> &l, int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      MPI_Request req;
      MPI_Recv_init(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), source,
                    static_cast<int>(t), comm, &req);
      return impl::prequest(req);
    }

    template<typename iterT>
    prequest recv_init(iterT begin, iterT end, int source, tag t = tag(0)) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        return recv_init(&(*begin), l, source, t);
      } else {
        iterator_layout<value_type> l(begin, end);
        return recv_init(&(*begin), l, source, t);
      }
    }

    // === probe ===
    // --- blocking probe ---
    [[nodiscard]] status probe(int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      status s;
      MPI_Probe(source, static_cast<int>(t), comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    // --- nonblocking probe ---
    [[nodiscard]] std::pair<bool, status> iprobe(int source, tag t = tag(0)) const {
      check_source(source);
      check_recv_tag(t);
      int result;
      status s;
      MPI_Iprobe(source, static_cast<int>(t), comm, &result,
                 reinterpret_cast<MPI_Status *>(&s));
      return std::make_pair(static_cast<bool>(result), s);
    }

    // === matching receive ===
    // --- blocking matching receive ---

    // --- nonblocking matching receive ---

    // === send and receive ===
    // --- send and receive ---
    template<typename T>
    status sendrecv(const T &senddata, int dest, tag sendtag, T &recvdata, int source,
                    tag recvtag) const {
      check_dest(dest);
      check_source(source);
      check_send_tag(sendtag);
      check_recv_tag(recvtag);
      status s;
      MPI_Sendrecv(&senddata, 1, detail::datatype_traits<T>::get_datatype(), dest,
                   static_cast<int>(sendtag), &recvdata, 1,
                   detail::datatype_traits<T>::get_datatype(), source,
                   static_cast<int>(recvtag), comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    template<typename T>
    status sendrecv(const T *senddata, const layout<T> &sendl, int dest, tag sendtag,
                    T *recvdata, const layout<T> &recvl, int source, tag recvtag) const {
      check_dest(dest);
      check_source(source);
      check_send_tag(sendtag);
      check_recv_tag(recvtag);
      status s;
      MPI_Sendrecv(senddata, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl), dest,
                   static_cast<int>(sendtag), recvdata, 1,
                   detail::datatype_traits<layout<T>>::get_datatype(recvl), source,
                   static_cast<int>(recvtag), comm, reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    template<typename iterT1, typename iterT2>
    status sendrecv(iterT1 begin1, iterT1 end1, int dest, tag sendtag, iterT2 begin2,
                    iterT2 end2, int source, tag recvtag) const {
      using value_type1 = typename std::iterator_traits<iterT1>::value_type;
      using value_type2 = typename std::iterator_traits<iterT2>::value_type;
      if constexpr (detail::is_contiguous_iterator_v<iterT1> and
                    detail::is_contiguous_iterator_v<iterT2>) {
        vector_layout<value_type1> l1(std::distance(begin1, end1));
        vector_layout<value_type2> l2(std::distance(begin2, end2));
        return sendrecv(&(*begin1), l1, dest, sendtag, &(*begin2), l2, source, recvtag);
      } else if constexpr (detail::is_contiguous_iterator_v<iterT1>) {
        vector_layout<value_type1> l1(std::distance(begin1, end1));
        iterator_layout<value_type2> l2(begin2, end2);
        return sendrecv(&(*begin1), l1, dest, sendtag, &(*begin2), l2, source, recvtag);
      } else if constexpr (detail::is_contiguous_iterator_v<iterT2>) {
        iterator_layout<value_type2> l1(begin1, end1);
        vector_layout<value_type2> l2(std::distance(begin2, end2));
        return sendrecv(&(*begin1), l1, dest, sendtag, &(*begin2), l2, source, recvtag);
      } else {
        iterator_layout<value_type1> l1(begin1, end1);
        iterator_layout<value_type2> l2(begin2, end2);
        return sendrecv(&(*begin1), l1, dest, sendtag, &(*begin2), l2, source, recvtag);
      }
    }

    // --- send, receive and replace ---
    template<typename T>
    status sendrecv_replace(T &data, int dest, tag sendtag, int source, tag recvtag) const {
      check_dest(dest);
      check_source(source);
      check_send_tag(sendtag);
      check_recv_tag(recvtag);
      status s;
      MPI_Sendrecv_replace(&data, 1, detail::datatype_traits<T>::get_datatype(), dest,
                           static_cast<int>(sendtag), source, static_cast<int>(recvtag), comm,
                           reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    template<typename T>
    status sendrecv_replace(T *data, const layout<T> &l, int dest, tag sendtag, int source,
                            tag recvtag) const {
      check_dest(dest);
      check_source(source);
      check_send_tag(sendtag);
      check_recv_tag(recvtag);
      status s;
      MPI_Sendrecv_replace(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), dest,
                           static_cast<int>(sendtag), source, static_cast<int>(recvtag), comm,
                           reinterpret_cast<MPI_Status *>(&s));
      return s;
    }

    template<typename iterT>
    status sendrecv_replace(iterT begin, iterT end, int dest, tag sendtag, int source,
                            tag recvtag) const {
      using value_type = typename std::iterator_traits<iterT>::value_type;
      if (detail::is_contiguous_iterator_v<iterT>) {
        vector_layout<value_type> l(std::distance(begin, end));
        return sendrecv_replace(&(*begin), l, dest, sendtag, source, recvtag);
      } else {
        iterator_layout<value_type> l(begin, end);
        return sendrecv_replace(&(*begin), l, dest, sendtag, source, recvtag);
      }
    }

    // === collective ==================================================
    // === barrier ===
    // --- blocking barrier ---
    void barrier() const { MPI_Barrier(comm); }

    // --- nonblocking barrier ---
    [[nodiscard]] irequest ibarrier() const {
      MPI_Request req;
      MPI_Ibarrier(comm, &req);
      return impl::irequest(req);
    }

    // === broadcast ===
    // --- blocking broadcast ---
    template<typename T>
    void bcast(int root_rank, T &data) const {
      check_root(root_rank);
      MPI_Bcast(&data, 1, detail::datatype_traits<T>::get_datatype(), root_rank, comm);
    }

    template<typename T>
    void bcast(int root_rank, T *data, const layout<T> &l) const {
      check_root(root_rank);
      MPI_Bcast(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), root_rank, comm);
    }

    // --- nonblocking broadcast ---
    template<typename T>
    irequest ibcast(int root_rank, T &data) const {
      check_root(root_rank);
      MPI_Request req;
      MPI_Ibcast(&data, 1, detail::datatype_traits<T>::get_datatype(), root_rank, comm, &req);
      return impl::irequest(req);
    }

    template<typename T>
    irequest ibcast(int root_rank, T *data, const layout<T> &l) const {
      check_root(root_rank);
      MPI_Request req;
      MPI_Ibcast(data, 1, detail::datatype_traits<layout<T>>::get_datatype(l), root_rank, comm,
                 &req);
      return impl::irequest(req);
    }

    // === gather ===
    // === root gets a single value from each rank and stores in contiguous memory
    // --- blocking gather ---
    template<typename T>
    void gather(int root_rank, const T &senddata, T *recvdata) const {
      check_root(root_rank);
      MPI_Gather(&senddata, 1, detail::datatype_traits<T>::get_datatype(), recvdata, 1,
                 detail::datatype_traits<T>::get_datatype(), root_rank, comm);
    }

    template<typename T>
    void gather(int root_rank, const T *senddata, const layout<T> &sendl, T *recvdata,
                const layout<T> &recvl) const {
      check_root(root_rank);
      MPI_Gather(senddata, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl), recvdata,
                 1, detail::datatype_traits<layout<T>>::get_datatype(recvl), root_rank, comm);
    }

    // --- nonblocking gather ---
    template<typename T>
    irequest igather(int root_rank, const T &senddata, T *recvdata) const {
      check_root(root_rank);
      MPI_Request req;
      MPI_Igather(&senddata, 1, detail::datatype_traits<T>::get_datatype(), recvdata, 1,
                  detail::datatype_traits<T>::get_datatype(), root_rank, comm, &req);
      return impl::irequest(req);
    }

    template<typename T>
    irequest igather(int root_rank, const T *senddata, const layout<T> &sendl, T *recvdata,
                     const layout<T> &recvl) const {
      check_root(root_rank);
      MPI_Request req;
      MPI_Igather(senddata, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl),
                  recvdata, 1, detail::datatype_traits<layout<T>>::get_datatype(recvl),
                  root_rank, comm, &req);
      return impl::irequest(req);
    }

    // --- blocking gather, non-root variant ---
    template<typename T>
    void gather(int root_rank, const T &senddata) const {
      check_nonroot(root_rank);
      MPI_Gather(&senddata, 1, detail::datatype_traits<T>::get_datatype(), 0, 0,
                 MPI_DATATYPE_NULL, root_rank, comm);
    }

    template<typename T>
    void gather(int root_rank, const T *senddata, const layout<T> &sendl) const {
      check_nonroot(root_rank);
      MPI_Gather(senddata, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl), 0, 0,
                 MPI_DATATYPE_NULL, root_rank, comm);
    }

    // --- nonblocking gather, non-root variant ---
    template<typename T>
    irequest igather(int root_rank, const T &senddata) const {
      check_nonroot(root_rank);
      MPI_Request req;
      MPI_Igather(&senddata, 1, detail::datatype_traits<T>::get_datatype(), 0, 0,
                  MPI_DATATYPE_NULL, root_rank, comm, &req);
      return impl::irequest(req);
    }

    template<typename T>
    irequest igather(int root_rank, const T *senddata, const layout<T> &sendl) const {
      check_nonroot(root_rank);
      MPI_Request req;
      MPI_Igather(senddata, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl), 0, 0,
                  MPI_DATATYPE_NULL, root_rank, comm, &req);
      return impl::irequest(req);
    }
    
    // === root gets varying amount of data from each rank and stores in non-contiguous memory
    // --- blocking gather ---
    /// \brief Gather messages with a variable amount of data from all processes at a single
    /// root process.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param root_rank rank of the receiving process
    /// \param senddata data to send
    /// \param sendl memory layout of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages, may be a null
    /// pointer at non-root processes
    /// \param recvls memory layouts of the data to receive by the root rank
    /// \param recvdispls std::vector<int> of the data to receive by the root rank
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    void gatherv(int root_rank, const T *senddata, const layout<T> &sendl, T *recvdata,
                 const layouts<T> &recvls, const std::vector<int> &recvdispls) const {
      int N(size());
      std::vector<int> senddispls(N);
      layouts<T> sendls(N);
      sendls[root_rank] = sendl;
      if (rank() == root_rank)
        alltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
      else
        alltoallv(senddata, sendls, senddispls, recvdata, mpl::layouts<T>(N), recvdispls);
    }

    /// \brief Gather messages with a variable amount of data from all processes at a single
    /// root process.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param root_rank rank of the receiving process
    /// \param senddata data to send
    /// \param sendl memory layout of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages, may be a null
    /// pointer at non-root processes
    /// \param recvls memory layouts of the data to receive by the root rank
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    void gatherv(int root_rank, const T *senddata, const layout<T> &sendl, T *recvdata,
                 const layouts<T> &recvls) const {
      gatherv(root_rank, senddata, sendl, recvdata, recvls, std::vector<int>(size()));
    }

    // --- non-blocking gather ---
    /// \brief Gather messages with a variable amount of data from all processes at a single
    /// root process in a non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param root_rank rank of the receiving process
    /// \param senddata data to send
    /// \param sendl memory layout of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages, may be a null
    /// pointer at non-root processes
    /// \param recvls memory layouts of the data to receive by the root rank
    /// \param recvdispls std::vector<int> of the data to receive by the root rank
    /// \return request representing the ongoing message transfer
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    irequest igatherv(int root_rank, const T *senddata, const layout<T> &sendl, T *recvdata,
                      const layouts<T> &recvls, const std::vector<int> &recvdispls) const {
      check_root(root_rank);
      int N(size());
      std::vector<int> senddispls(N);
      layouts<T> sendls(N);
      sendls[root_rank] = sendl;
      if (rank() == root_rank)
        return ialltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
      else
        return ialltoallv(senddata, sendls, senddispls, recvdata, mpl::layouts<T>(N),
                          recvdispls);
    }

    /// \brief Gather messages with a variable amount of data from all processes at a single
    /// root process in a non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param root_rank rank of the receiving process
    /// \param senddata data to send
    /// \param sendl memory layout of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages, may be a null
    /// pointer at non-root processes
    /// \param recvls memory layouts of the data to receive by the root rank
    /// \return request representing the ongoing message transfer
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    irequest igatherv(int root_rank, const T *senddata, const layout<T> &sendl, T *recvdata,
                      const layouts<T> &recvls) const {
      return igatherv(root_rank, senddata, sendl, recvdata, recvls, std::vector<int>(size()));
    }

    // --- blocking gather, non-root variant ---
    /// \brief Gather messages with a variable amount of data from all processes at a single
    /// root process.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param root_rank rank of the receiving process
    /// \param senddata data to send
    /// \param sendl memory layout of the data to send
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator. This particular overload can only be
    /// called by non-root processes.
    template<typename T>
    void gatherv(int root_rank, const T *senddata, const layout<T> &sendl) const {
      check_nonroot(root_rank);
      int N(size());
      std::vector<int> sendrecvdispls(N);
      layouts<T> sendls(N);
      sendls[root_rank] = sendl;
      alltoallv(senddata, sendls, sendrecvdispls, static_cast<T *>(nullptr), mpl::layouts<T>(N),
                sendrecvdispls);
    }

    // --- non-blocking gather, non-root variant ---
    /// \brief Gather messages with a variable amount of data from all processes at a single
    /// root process in a non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param root_rank rank of the receiving process
    /// \param senddata data to send
    /// \param sendl memory layout of the data to send
    /// \return request representing the ongoing message transfer
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator. This particular overload can only be
    /// called by non-root processes.
    template<typename T>
    irequest igatherv(int root_rank, const T *senddata, const layout<T> &sendl) const {
      check_nonroot(root_rank);
      int N(size());
      std::vector<int> sendrecvdispls(N);
      layouts<T> sendls(N);
      sendls[root_rank] = sendl;
      return ialltoallv(senddata, sendls, sendrecvdispls, static_cast<T *>(nullptr),
                        mpl::layouts<T>(N), sendrecvdispls);
    }

    // === allgather ===
    // === get a single value from each rank and stores in contiguous memory
    // --- blocking allgather ---
    template<typename T>
    void allgather(const T &senddata, T *recvdata) const {
      MPI_Allgather(&senddata, 1, detail::datatype_traits<T>::get_datatype(), recvdata, 1,
                    detail::datatype_traits<T>::get_datatype(), comm);
    }

    template<typename T>
    void allgather(const T *senddata, const layout<T> &sendl, T *recvdata,
                   const layout<T> &recvl) const {
      MPI_Allgather(senddata, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl),
                    recvdata, 1, detail::datatype_traits<layout<T>>::get_datatype(recvl), comm);
    }

    // --- nonblocking allgather ---
    template<typename T>
    irequest iallgather(const T &senddata, T *recvdata) const {
      MPI_Request req;
      MPI_Iallgather(&senddata, 1, detail::datatype_traits<T>::get_datatype(), recvdata, 1,
                     detail::datatype_traits<T>::get_datatype(), comm, &req);
      return impl::irequest(req);
    }

    template<typename T>
    irequest iallgather(const T *senddata, const layout<T> &sendl, T *recvdata,
                        const layout<T> &recvl) const {
      MPI_Request req;
      MPI_Iallgather(senddata, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl),
                     recvdata, 1, detail::datatype_traits<layout<T>>::get_datatype(recvl), comm,
                     &req);
      return impl::irequest(req);
    }
    
    // === get varying amount of data from each rank and stores in non-contiguous memory
    // --- blocking allgather ---
    /// \brief Gather messages with a variable amount of data from all processes and distribute
    /// result to all processes.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param senddata data to send
    /// \param sendl memory layout of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvls memory layouts of the data to receive
    /// \param recvdispls std::vector<int> of the data to receive
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    void allgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                    const layouts<T> &recvls, const std::vector<int> &recvdispls) const {
      int N(size());
      std::vector<int> senddispls(N);
      layouts<T> sendls(N, sendl);
      alltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
    }

    /// \brief Gather messages with a variable amount of data from all processes and distribute
    /// result to all processes.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param senddata data to send
    /// \param sendl memory layout of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvls memory layouts of the data to receive
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    void allgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                    const layouts<T> &recvls) const {
      allgatherv(senddata, sendl, recvdata, recvls, std::vector<int>(size()));
    }

    // --- non-blocking allgather ---
    /// \brief Gather messages with a variable amount of data from all processes and distribute
    /// result to all processes in a non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param senddata data to send
    /// \param sendl memory layout of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvls memory layouts of the data to receive
    /// \param recvdispls std::vector<int> of the data to receive
    /// \return request representing the ongoing message transfer
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    irequest iallgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                         const layouts<T> &recvls, const std::vector<int> &recvdispls) const {
      int N(size());
      std::vector<int> senddispls(N);
      layouts<T> sendls(N, sendl);
      return ialltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
    }

    /// \brief Gather messages with a variable amount of data from all processes and distribute
    /// result to all processes in a non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param senddata data to send
    /// \param sendl memory layout of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvls memory layouts of the data to receive
    /// \return request representing the ongoing message transfer
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    irequest iallgatherv(const T *senddata, const layout<T> &sendl, T *recvdata,
                         const layouts<T> &recvls) const {
      return iallgatherv(senddata, sendl, recvdata, recvls, std::vector<int>(size()));
    }

    // === scatter ===
    // === root sends a single value from contiguous memory to each rank
    // --- blocking scatter ---
    template<typename T>
    void scatter(int root_rank, const T *senddata, T &recvdata) const {
      check_root(root_rank);
      MPI_Scatter(senddata, 1, detail::datatype_traits<T>::get_datatype(), &recvdata, 1,
                  detail::datatype_traits<T>::get_datatype(), root_rank, comm);
    }

    template<typename T>
    void scatter(int root_rank, const T *senddata, const layout<T> &sendl, T *recvdata,
                 const layout<T> &recvl) const {
      check_root(root_rank);
      MPI_Scatter(senddata, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl),
                  recvdata, 1, detail::datatype_traits<layout<T>>::get_datatype(recvl),
                  root_rank, comm);
    }

    // --- nonblocking scatter ---
    template<typename T>
    irequest iscatter(int root_rank, const T *senddata, T &recvdata) const {
      check_root(root_rank);
      MPI_Request req;
      MPI_Iscatter(senddata, 1, detail::datatype_traits<T>::get_datatype(), &recvdata, 1,
                   detail::datatype_traits<T>::get_datatype(), root_rank, comm, &req);
      return impl::irequest(req);
    }

    template<typename T>
    irequest iscatter(int root_rank, const T *senddata, const layout<T> &sendl, T *recvdata,
                      const layout<T> &recvl) const {
      check_root(root_rank);
      MPI_Request req;
      MPI_Iscatter(senddata, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl),
                   recvdata, 1, detail::datatype_traits<layout<T>>::get_datatype(recvl),
                   root_rank, comm, &req);
      return impl::irequest(req);
    }

    // --- blocking scatter, non-root variant ---
    template<typename T>
    void scatter(int root_rank, T &recvdata) const {
      check_nonroot(root_rank);
      MPI_Scatter(0, 0, MPI_DATATYPE_NULL, &recvdata, 1,
                  detail::datatype_traits<T>::get_datatype(), root_rank, comm);
    }

    template<typename T>
    void scatter(int root_rank, T *recvdata, const layout<T> &recvl) const {
      check_root(root_rank);
      MPI_Scatter(0, 0, MPI_DATATYPE_NULL, recvdata, 1,
                  detail::datatype_traits<layout<T>>::get_datatype(recvl), root_rank, comm);
    }

    // --- nonblocking scatter, non-root variant ---
    template<typename T>
    irequest iscatter(int root_rank, T &recvdata) const {
      check_nonroot(root_rank);
      MPI_Request req;
      MPI_Iscatter(0, 0, MPI_DATATYPE_NULL, &recvdata, 1,
                   detail::datatype_traits<T>::get_datatype(), root_rank, comm, &req);
      return impl::irequest(req);
    }

    template<typename T>
    irequest iscatter(int root_rank, T *recvdata, const layout<T> &recvl) const {
      check_nonroot(root_rank);
      MPI_Request req;
      MPI_Iscatter(0, 0, MPI_DATATYPE_NULL, recvdata, 1,
                   detail::datatype_traits<layout<T>>::get_datatype(recvl), root_rank, comm,
                   &req);
      return impl::irequest(req);
    }
    
    // === root sends varying amount of data from non-contiguous memory to each rank
    // --- blocking scatter ---
    /// \brief Scatter messages with a variable amount of data from a single root process to all
    /// processes.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param root_rank rank of the sending process
    /// \param senddata pointer to continuous storage for outgoing messages, may be a null
    /// pointer at non-root processes
    /// \param sendls memory layouts of the data to send
    /// \param senddispls std::vector<int> of the data to send by the root rank
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvl memory layout of the data to receive by the root rank
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    void scatterv(int root_rank, const T *senddata, const layouts<T> &sendls,
                  const std::vector<int> &senddispls, T *recvdata, const layout<T> &recvl) const {
      check_root(root_rank);
      const int N{size()};
      std::vector<int> recvdispls(N);
      layouts<T> recvls(N);
      recvls[root_rank] = recvl;
      if (rank() == root_rank)
        alltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
      else
        alltoallv(senddata, sendls, senddispls, recvdata, mpl::layouts<T>(N), recvdispls);
    }
    
    /// \brief Scatter messages with a variable amount of data from a single root process to all
    /// processes.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param root_rank rank of the sending process
    /// \param senddata pointer to continuous storage for outgoing messages, may be a null
    /// pointer at non-root processes
    /// \param sendls memory layouts of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvl memory layout of the data to receive by the root rank
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    void scatterv(int root_rank, const T *senddata, const layouts<T> &sendls, T *recvdata,
                  const layout<T> &recvl) const {
      scatterv(root_rank, senddata, sendls, std::vector<int>(size()), recvdata, recvl);
    }

    // --- non-blocking scatter ---
    /// \brief Scatter messages with a variable amount of data from a single root process to all
    /// processes in a non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param root_rank rank of the sending process
    /// \param senddata pointer to continuous storage for outgoing messages, may be a null
    /// pointer at non-root processes
    /// \param sendls memory layouts of the data to send
    /// \param senddispls std::vector<int> of the data to send by the root rank
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvl memory layout of the data to receive by the root rank
    /// \return request representing the ongoing message transfer
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    irequest iscatterv(int root_rank, const T *senddata, const layouts<T> &sendls,
                       const std::vector<int> &senddispls, T *recvdata,
                       const layout<T> &recvl) const {
      check_root(root_rank);
      const int N{size()};
      std::vector<int> recvdispls(N);
      layouts<T> recvls(N);
      recvls[root_rank] = recvl;
      if (rank() == root_rank)
        return ialltoallv(senddata, sendls, senddispls, recvdata, recvls, recvdispls);
      else
        return ialltoallv(senddata, sendls, senddispls, recvdata, mpl::layouts<T>(N),
                          recvdispls);
    }

    /// \brief Scatter messages with a variable amount of data from a single root process to all
    /// processes  in a non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param root_rank rank of the sending process
    /// \param senddata pointer to continuous storage for outgoing messages, may be a null
    /// pointer at non-root processes
    /// \param sendls memory layouts of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvl memory layout of the data to receive by the root rank
    /// \return request representing the ongoing message transfer
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    irequest iscatterv(int root_rank, const T *senddata, const layouts<T> &sendls, T *recvdata,
                       const layout<T> &recvl) const {
      return iscatterv(root_rank, senddata, sendls, std::vector<int>(size()), recvdata, recvl);
    }

    // --- blocking scatter, non-root variant ---
    /// \brief Scatter messages with a variable amount of data from a single root process to all
    /// processes.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param root_rank rank of the sending process
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvl memory layout of the data to receive by the root rank
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator. This particular overload can only be
    /// called by non-root processes.
    template<typename T>
    void scatterv(int root_rank, T *recvdata, const layout<T> &recvl) const {
      check_root(root_rank);
      const int N{size()};
      std::vector<int> sendrecvdispls(N);
      layouts<T> recvls(N);
      recvls[root_rank] = recvl;
      alltoallv(static_cast<const T *>(nullptr), mpl::layouts<T>(N), sendrecvdispls, recvdata,
                recvls, sendrecvdispls);
    }

    // --- non-blocking scatter, non-root variant ---
    /// \brief Scatter messages with a variable amount of data from a single root process to all
    /// processes in a non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param root_rank rank of the sending process
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvl memory layout of the data to receive by the root rank
    /// \return request representing the ongoing message transfer
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator. This particular overload can only be
    /// called by non-root processes.
    template<typename T>
    irequest iscatterv(int root_rank, T *recvdata, const layout<T> &recvl) const {
      check_root(root_rank);
      const int N{size()};
      std::vector<int> sendrecvdispls(N);
      layouts<T> recvls(N);
      recvls[root_rank] = recvl;
      return ialltoallv(static_cast<const T *>(nullptr), mpl::layouts<T>(N), sendrecvdispls,
                        recvdata, recvls, sendrecvdispls);
    }

    // === all-to-all ===
    // === each rank sends a single value to each rank
    // --- blocking all-to-all ---
    template<typename T>
    void alltoall(const T *senddata, T *recvdata) const {
      MPI_Alltoall(senddata, 1, detail::datatype_traits<T>::get_datatype(), recvdata, 1,
                   detail::datatype_traits<T>::get_datatype(), comm);
    }

    template<typename T>
    void alltoall(const T *senddata, const layout<T> &sendl, T *recvdata,
                  const layout<T> &recvl) const {
      MPI_Alltoall(senddata, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl), recvdata, 1,
                   detail::datatype_traits<layout<T>>::get_datatype(recvl), comm);
    }

    // --- nonblocking all-to-all ---
    template<typename T>
    irequest ialltoall(const T *senddata, T *recvdata) const {
      MPI_Request req;
      MPI_Ialltoall(senddata, 1, detail::datatype_traits<T>::get_datatype(), recvdata, 1,
                    detail::datatype_traits<T>::get_datatype(), comm, &req);
      return impl::irequest(req);
    }

    template<typename T>
    irequest ialltoall(const T *senddata, const layout<T> &sendl, T *recvdata,
                       const layout<T> &recvl) const {
      MPI_Request req;
      MPI_Ialltoall(senddata, 1, detail::datatype_traits<layout<T>>::get_datatype(sendl), recvdata,
                    1, detail::datatype_traits<layout<T>>::get_datatype(recvl), comm, &req);
      return impl::irequest(req);
    }

    // --- blocking all-to-all, in place ---
    template<typename T>
    void alltoall(T *recvdata) const {
      MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvdata, 1,
                   detail::datatype_traits<T>::get_datatype(), comm);
    }

    template<typename T>
    void alltoall(T *recvdata, const layout<T> &recvl) const {
      MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvdata, 1,
                   detail::datatype_traits<layout<T>>::get_datatype(recvl), comm);
    }

    // --- nonblocking all-to-all, in place ---
    template<typename T>
    irequest ialltoall(T *recvdata) const {
      MPI_Request req;
      MPI_Ialltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvdata, 1,
                    detail::datatype_traits<T>::get_datatype(), comm, &req);
      return impl::irequest(req);
    }

    template<typename T>
    irequest ialltoall(T *recvdata, const layout<T> &recvl) const {
      MPI_Request req;
      MPI_Ialltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvdata, 1,
                    detail::datatype_traits<layout<T>>::get_datatype(recvl), comm, &req);
      return impl::irequest(req);
    }

    // === each rank sends a varying number of values to each rank with possibly different
    // layouts
    // --- blocking all-to-all ---
    /// \brief Sends messages with a variable amount of data to all processes and receives
    /// messages with a variable amount of data from all processes.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param senddata pointer to continuous storage for outgoing messages
    /// \param sendls memory layouts of the data to send
    /// \param senddispls std::vector<int> of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvls memory layouts of the data to receive
    /// \param recvdispls std::vector<int> of the data to receive
    /// \details Each process in the communicator sends elements of type T to each process
    /// (including itself) and receives elements of type T from each process.  Send- and
    /// receive-data are stored in consecutive blocks of variable size in the buffers senddata
    /// and recvdata, respectively. The i-th memory block with the layout sendls[i] in the array
    /// senddata starts senddispls[i] bytes after the address given in senddata. The i-th memory
    /// block is sent to the i-th process. The i-th memory block with the layout recvls[i] in
    /// the array recvdata starts recvdispls[i] bytes after the address given in recvdata.
    /// When the function has finished, the i-th memory block in the array recvdata was
    /// received from the i-th process.
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    void alltoallv(const T *senddata, const layouts<T> &sendls, const std::vector<int> &senddispls,
                   T *recvdata, const layouts<T> &recvls,
                   const std::vector<int> &recvdispls) const {
      const std::vector<int> counts(recvls.size(), 1);
      const std::vector<int> senddispls_int(senddispls.begin(), senddispls.end());
      const std::vector<int> recvdispls_int(recvdispls.begin(), recvdispls.end());
      static_assert(
          sizeof(decltype(*sendls())) == sizeof(MPI_Datatype),
          "compiler adds some unexpected padding, reinterpret cast will yield wrong results");
      MPI_Alltoallw(senddata, counts.data(), senddispls_int.data(),
                    reinterpret_cast<const MPI_Datatype *>(sendls()), recvdata, counts.data(),
                    recvdispls_int.data(), reinterpret_cast<const MPI_Datatype *>(recvls()),
                    comm);
    }

    /// \brief Sends messages with a variable amount of data to all processes and receives
    /// messages with a variable amount of data from all processes.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param senddata pointer to continuous storage for outgoing messages
    /// \param sendls memory layouts of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvls memory layouts of the data to receive
    /// \details Each process in the communicator sends elements of type T to each process
    /// (including itself) and receives elements of type T from each process.  Send- and
    /// receive-data are stored in consecutive blocks of variable size in the buffers senddata
    /// and recvdata, respectively. The i-th memory block with the layout sendls[i] in the array
    /// senddata starts at the address given in senddata. The i-th memory block is sent to the
    /// i-th process. The i-th memory block with the layout recvls[i] in the array recvdata
    /// starts at the address given in recvdata.  Note that the memory layouts need to include
    /// appropriate holes at the beginning in order to avoid overlapping send- or receive
    /// blocks. When the function has finished, the i-th memory block in the array recvdata
    /// was received from the i-th process.
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    void alltoallv(const T *senddata, const layouts<T> &sendls, T *recvdata,
                   const layouts<T> &recvls) const {
      const std::vector<int> sendrecvdispls(size());
      alltoallv(senddata, sendls, sendrecvdispls, recvdata, recvls, sendrecvdispls);
    }
    
    // --- non-blocking all-to-all ---
  public:

    /// \brief Sends messages with a variable amount of data to all processes and receives
    /// messages with a variable amount of data from all processes in a non-blocking manner.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param senddata pointer to continuous storage for outgoing messages
    /// \param sendls memory layouts of the data to send
    /// \param recvdata pointer to continuous storage for incoming messages
    /// \param recvls memory layouts of the data to receive
    /// \return request representing the ongoing message transfer
    /// \details Each process in the communicator sends elements of type T to each process
    /// (including itself) and receives elements of type T from each process.  Send- and
    /// receive-data are stored in consecutive blocks of variable size in the buffers senddata
    /// and recvdata, respectively. The i-th memory block with the layout sendls[i] in the array
    /// senddata starts at the address given in senddata. The i-th memory block is sent to the
    /// i-th process. The i-th memory block with the layout recvls[i] in the array recvdata
    /// starts at the address given in recvdata.  Note that the memory layouts need to include
    /// appropriate holes at the beginning in order to avoid overlapping send- or receive
    /// blocks. When the function has finished, the i-th memory block in the array recvdata
    /// was received from the i-th process.
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    irequest ialltoallv(const T *senddata, const layouts<T> &sendls, T *recvdata,
                        const layouts<T> &recvls) const {
      const std::vector<int> sendrecvdispls(size());
      return ialltoallv(senddata, sendls, sendrecvdispls, recvdata, recvls, sendrecvdispls);
    }

    // --- blocking all-to-all, in place ---
    /// \brief Sends messages with a variable amount of data to all processes and receives
    /// messages with a variable amount of data from all processes, in-place variant.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param sendrecvdata pointer to continuous storage for outgoing and incoming messages
    /// \param sendrecvls memory layouts of the data to send and to receive
    /// \param sendrecvdispls std::vector<int> of the data to send and to receive
    /// \details Each process in the communicator sends elements of type T to each process
    /// (including itself) and receives elements of type T from each process.  Send- and
    /// receive-data are stored in consecutive blocks of variable size in the buffer
    /// sendecvdata. The i-th memory block with the layout sendlrecvs[i] in the array
    /// sendrecvdata starts sendrecvdispls[i] bytes after the address given in sendrecvdata. The
    /// i-th memory block is sent to the i-th process. When the function has finished, the i-th
    /// memory block in the array sendrecvdata was received from the i-th process.
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    void alltoallv(T *sendrecvdata, const layouts<T> &sendrecvls,
                   const std::vector<int> &sendrecvdispls) const {
      const std::vector<int> counts(sendrecvls.size(), 1);
      const std::vector<int> sendrecvdispls_int(sendrecvdispls.begin(), sendrecvdispls.end());
      MPI_Alltoallw(MPI_IN_PLACE, 0, 0, 0, sendrecvdata, counts.data(),
                    sendrecvdispls_int.data(),
                    reinterpret_cast<const MPI_Datatype *>(sendrecvls()), comm);
    }

    /// \brief Sends messages with a variable amount of data to all processes and receives
    /// messages with a variable amount of data from all processes, in-place variant.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param sendrecvdata pointer to continuous storage for incoming and outgoing messages
    /// \param sendrecvls memory layouts of the data to send and to receive
    /// \details Each process in the communicator sends elements of type T to each process
    /// (including itself) and receives elements of type T from each process.  Send- and
    /// receive-data are stored in consecutive blocks of variable size in the buffer
    /// sendrecvdata. The i-th memory block with the layout sendrecvls[i] in the array
    /// sendrecvdata starts at the address given in sendrecvdata. The i-th memory block is sent
    /// to the i-th process. Note that the memory layouts need to include appropriate holes at
    /// the beginning in order to avoid overlapping send-receive blocks. When the function has
    /// finished, the i-th memory block in the array sendrecvdata was received from the
    /// i-th process.
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    void alltoallv(T *sendrecvdata, const layouts<T> &sendrecvls) const {
      alltoallv(sendrecvdata, sendrecvls, std::vector<int>(size()));
    }

    /// \brief Sends messages with a variable amount of data to all processes and receives
    /// messages with a variable amount of data from all processes in a non-blocking manner,
    /// in-place variant.
    /// \tparam T type of the data to send, must meet the requirements as described in the \ref
    /// data_types "data types" section
    /// \param sendrecvdata pointer to continuous storage for incoming and outgoing messages
    /// \param sendrecvls memory layouts of the data to send and to receive
    /// \details Each process in the communicator sends elements of type T to each process
    /// (including itself) and receives elements of type T from each process.  Send- and
    /// receive-data are stored in consecutive blocks of variable size in the buffer
    /// sendrecvdata. The i-th memory block with the layout sendrecvls[i] in the array
    /// sendrecvdata starts at the address given in sendrecvdata. The i-th memory block is sent
    /// to the i-th process. Note that the memory layouts need to include appropriate holes at
    /// the beginning in order to avoid overlapping send-receive blocks. When the function has
    /// finished, the i-th memory block in the array sendrecvdata was received from the
    /// i-th process.
    /// \note This is a collective operation and must be called (possibly by utilizing anther
    /// overload) by all processes in the communicator.
    template<typename T>
    irequest ialltoallv(T *sendrecvdata, const layouts<T> &sendrecvls) const {
      return ialltoallv(sendrecvdata, sendrecvls, std::vector<int>(size()));
    }

    // === reduce ===
    // --- blocking reduce ---
    template<typename T, typename F>
    void reduce(F f, int root_rank, const T &senddata, T &recvdata) const {
      check_root(root_rank);
      MPI_Reduce(&senddata, &recvdata, 1, detail::datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, root_rank, comm);
    }

    /// \anchor communicator_reduce_contiguous_layout
    template<typename T, typename F>
    void reduce(F f, int root_rank, const T *senddata, T *recvdata,
                const contiguous_layout<T> &l) const {
      check_root(root_rank);
      MPI_Reduce(senddata, recvdata, l.size(), detail::datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, root_rank, comm);
    }

    // --- non-blocking reduce ---
    template<typename T, typename F>
    irequest ireduce(F f, int root_rank, const T &senddata, T &recvdata) const {
      check_root(root_rank);
      MPI_Request req;
      MPI_Ireduce(&senddata, &recvdata, 1, detail::datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, root_rank, comm, &req);
      return impl::irequest(req);
    }

    template<typename T, typename F>
    irequest ireduce(F f, int root_rank, const T *senddata, T *recvdata,
                     const contiguous_layout<T> &l) const {
      check_root(root_rank);
      MPI_Request req;
      MPI_Ireduce(senddata, recvdata, l.size(), detail::datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, root_rank, comm, &req);
      return impl::irequest(req);
    }

    // --- blocking reduce, in place ---
    template<typename T, typename F>
    void reduce(F f, int root_rank, T &sendrecvdata) const {
      check_root(root_rank);
      if (rank() == root_rank)
        MPI_Reduce(MPI_IN_PLACE, &sendrecvdata, 1, detail::datatype_traits<T>::get_datatype(),
                   detail::get_op<T, F>(f).mpi_op, root_rank, comm);
      else
        MPI_Reduce(&sendrecvdata, nullptr, 1, detail::datatype_traits<T>::get_datatype(),
                   detail::get_op<T, F>(f).mpi_op, root_rank, comm);
    }

    template<typename T, typename F>
    void reduce(F f, int root_rank, const T &senddata) const {
      check_nonroot(root_rank);
      MPI_Reduce(&senddata, nullptr, 1, detail::datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, root_rank, comm);
    }

    template<typename T, typename F>
    void reduce(F f, int root_rank, T *sendrecvdata, const contiguous_layout<T> &l) const {
      if (rank() == root_rank)
        MPI_Reduce(MPI_IN_PLACE, sendrecvdata, l.size(),
                   detail::datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
                   root_rank, comm);
      else
        MPI_Reduce(sendrecvdata, nullptr, l.size(), detail::datatype_traits<T>::get_datatype(),
                   detail::get_op<T, F>(f).mpi_op, root_rank, comm);
    }

    template<typename T, typename F>
    void reduce(F f, int root_rank, const T *sendrecvdata,
                const contiguous_layout<T> &l) const {
      check_nonroot(root_rank);
      MPI_Reduce(sendrecvdata, nullptr, l.size(), detail::datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, root_rank, comm);
    }

    // --- non-blocking reduce, in place ---
    template<typename T, typename F>
    irequest ireduce(F f, int root_rank, T &sendrecvdata) const {
      check_root(root_rank);
      MPI_Request req;
      if (rank() == root_rank)
        MPI_Ireduce(MPI_IN_PLACE, &sendrecvdata, 1, detail::datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, root_rank, comm, &req);
      else
        MPI_Ireduce(&sendrecvdata, nullptr, 1, detail::datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, root_rank, comm, &req);
      return impl::irequest(req);
    }

    template<typename T, typename F>
    irequest ireduce(F f, int root_rank, const T &sendrecvdata) const {
      check_nonroot(root_rank);
      MPI_Request req;
      MPI_Ireduce(&sendrecvdata, nullptr, 1, detail::datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, root_rank, comm, &req);
      return impl::irequest(req);
    }

    template<typename T, typename F>
    irequest ireduce(F f, int root_rank, T *sendrecvdata, const contiguous_layout<T> &l) const {
      check_root(root_rank);
      MPI_Request req;
      if (rank() == root_rank)
        MPI_Ireduce(MPI_IN_PLACE, sendrecvdata, l.size(),
                    detail::datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
                    root_rank, comm, &req);
      else
        MPI_Ireduce(sendrecvdata, nullptr, l.size(), detail::datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, root_rank, comm, &req);
      return impl::irequest(req);
    }

    template<typename T, typename F>
    irequest ireduce(F f, int root_rank, const T *sendrecvdata,
                     const contiguous_layout<T> &l) const {
      check_nonroot(root_rank);
      MPI_Request req;
      MPI_Ireduce(sendrecvdata, nullptr, l.size(), detail::datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, root_rank, comm, &req);
      return impl::irequest(req);
    }

    // === all-reduce ===
    // --- blocking all-reduce ---
    template<typename T, typename F>
    void allreduce(F f, const T &senddata, T &recvdata) const {
      MPI_Allreduce(&senddata, &recvdata, 1, detail::datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, comm);
    }

    template<typename T, typename F>
    void allreduce(F f, const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Allreduce(senddata, recvdata, l.size(), detail::datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, comm);
    }

    // --- non-blocking all-reduce ---
    template<typename T, typename F>
    irequest iallreduce(F f, const T &senddata, T &recvdata) const {
      MPI_Request req;
      MPI_Iallreduce(&senddata, &recvdata, 1, detail::datatype_traits<T>::get_datatype(),
                     detail::get_op<T, F>(f).mpi_op, comm, &req);
      return impl::irequest(req);
    }

    template<typename T, typename F>
    irequest iallreduce(F f, const T *senddata, T *recvdata,
                        const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Iallreduce(senddata, recvdata, l.size(), detail::datatype_traits<T>::get_datatype(),
                     detail::get_op<T, F>(f).mpi_op, comm, &req);
      return impl::irequest(req);
    }

    // --- blocking all-reduce, in place ---
    template<typename T, typename F>
    void allreduce(F f, T &sendrecvdata) const {
      MPI_Allreduce(MPI_IN_PLACE, &sendrecvdata, 1, detail::datatype_traits<T>::get_datatype(),
                    detail::get_op<T, F>(f).mpi_op, comm);
    }

    template<typename T, typename F>
    void allreduce(F f, T *sendrecvdata, const contiguous_layout<T> &l) const {
      MPI_Allreduce(MPI_IN_PLACE, sendrecvdata, l.size(),
                    detail::datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
                    comm);
    }

    // --- non-blocking all-reduce, in place ---
    template<typename T, typename F>
    irequest iallreduce(F f, T &sendrecvdata) const {
      MPI_Request req;
      MPI_Iallreduce(MPI_IN_PLACE, &sendrecvdata, 1, detail::datatype_traits<T>::get_datatype(),
                     detail::get_op<T, F>(f).mpi_op, comm, &req);
      return impl::irequest(req);
    }

    template<typename T, typename F>
    irequest iallreduce(F f, T *sendrecvdata, const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Iallreduce(MPI_IN_PLACE, sendrecvdata, l.size(),
                     detail::datatype_traits<T>::get_datatype(), detail::get_op<T, F>(f).mpi_op,
                     comm, &req);
      return impl::irequest(req);
    }

    // === reduce-scatter-block ===
    // --- blocking reduce-scatter-block ---
    template<typename T, typename F>
    void reduce_scatter_block(F f, const T *senddata, T &recvdata) const {
      MPI_Reduce_scatter_block(senddata, &recvdata, 1,
                               detail::datatype_traits<T>::get_datatype(),
                               detail::get_op<T, F>(f).mpi_op, comm);
    }

    template<typename T, typename F>
    void reduce_scatter_block(F f, const T *senddata, T *recvdata,
                              const contiguous_layout<T> &l) const {
      MPI_Reduce_scatter_block(senddata, recvdata, l.size(),
                               detail::datatype_traits<T>::get_datatype(),
                               detail::get_op<T, F>(f).mpi_op, comm);
    }

    // --- non-blocking reduce-scatter-block ---
    template<typename T, typename F>
    irequest ireduce_scatter_block(F f, const T *senddata, T &recvdata) const {
      MPI_Request req;
      MPI_Ireduce_scatter_block(senddata, &recvdata, 1,
                                detail::datatype_traits<T>::get_datatype(),
                                detail::get_op<T, F>(f).mpi_op, comm, &req);
      return impl::irequest(req);
    }

    template<typename T, typename F>
    irequest ireduce_scatter_block(F f, const T *senddata, T *recvdata,
                                   const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Ireduce_scatter_block(senddata, recvdata, l.size(),
                                detail::datatype_traits<T>::get_datatype(),
                                detail::get_op<T, F>(f).mpi_op, comm, &req);
      return impl::irequest(req);
    }

    // === reduce-scatter ===
    // --- blocking reduce-scatter ---
    template<typename T, typename F>
    void reduce_scatter(F f, const T *senddata, T *recvdata,
                        const contiguous_layouts<T> &recvcounts) const {
      MPI_Reduce_scatter(senddata, recvdata, recvcounts.sizes(),
                         detail::datatype_traits<T>::get_datatype(),
                         detail::get_op<T, F>(f).mpi_op, comm);
    }

    // --- non-blocking reduce-scatter ---
    template<typename T, typename F>
    irequest ireduce_scatter(F f, const T *senddata, T *recvdata,
                             contiguous_layouts<T> &recvcounts) const {
      MPI_Request req;
      MPI_Ireduce_scatter(senddata, recvdata, recvcounts.sizes(),
                          detail::datatype_traits<T>::get_datatype(),
                          detail::get_op<T, F>(f).mpi_op, comm, &req);
      return impl::irequest(req);
    }

    // === scan ===
    // --- blocking scan ---
    template<typename T, typename F>
    void scan(F f, const T &senddata, T &recvdata) const {
      MPI_Scan(&senddata, &recvdata, 1, detail::datatype_traits<T>::get_datatype(),
               detail::get_op<T, F>(f).mpi_op, comm);
    }

    template<typename T, typename F>
    void scan(F f, const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Scan(senddata, recvdata, l.size(), detail::datatype_traits<T>::get_datatype(),
               detail::get_op<T, F>(f).mpi_op, comm);
    }

    // --- non-blocking scan ---
    template<typename T, typename F>
    irequest iscan(F f, const T &senddata, T &recvdata) const {
      MPI_Request req;
      MPI_Iscan(&senddata, &recvdata, 1, detail::datatype_traits<T>::get_datatype(),
                detail::get_op<T, F>(f).mpi_op, comm, &req);
      return impl::irequest(req);
    }

    template<typename T, typename F>
    irequest iscan(F f, const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Iscan(senddata, recvdata, l.size(), detail::datatype_traits<T>::get_datatype(),
                detail::get_op<T, F>(f).mpi_op, comm, &req);
      return impl::irequest(req);
    }

    // --- blocking scan, in place ---
    template<typename T, typename F>
    void scan(F f, T &recvdata) const {
      MPI_Scan(MPI_IN_PLACE, &recvdata, 1, detail::datatype_traits<T>::get_datatype(),
               detail::get_op<T, F>(f).mpi_op, comm);
    }

    template<typename T, typename F>
    void scan(F f, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Scan(MPI_IN_PLACE, recvdata, l.size(), detail::datatype_traits<T>::get_datatype(),
               detail::get_op<T, F>(f).mpi_op, comm);
    }

    // --- non-blocking scan, in place ---
    template<typename T, typename F>
    irequest iscan(F f, T &recvdata) const {
      MPI_Request req;
      MPI_Iscan(MPI_IN_PLACE, &recvdata, 1, detail::datatype_traits<T>::get_datatype(),
                detail::get_op<T, F>(f).mpi_op, comm, &req);
      return impl::irequest(req);
    }

    template<typename T, typename F>
    irequest iscan(F f, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Iscan(MPI_IN_PLACE, recvdata, l.size(), detail::datatype_traits<T>::get_datatype(),
                detail::get_op<T, F>(f).mpi_op, comm, &req);
      return impl::irequest(req);
    }

    // === exscan ===
    // --- blocking exscan ---
    template<typename T, typename F>
    void exscan(F f, const T &senddata, T &recvdata) const {
      MPI_Exscan(&senddata, &recvdata, 1, detail::datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, comm);
    }

    template<typename T, typename F>
    void exscan(F f, const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Exscan(senddata, recvdata, l.size(), detail::datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, comm);
    }

    // --- non-blocking exscan ---
    template<typename T, typename F>
    irequest iexscan(F f, const T &senddata, T &recvdata) const {
      MPI_Request req;
      MPI_Iexscan(&senddata, &recvdata, 1, detail::datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, comm, &req);
      return impl::irequest(req);
    }

    template<typename T, typename F>
    irequest iexscan(F f, const T *senddata, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Iexscan(senddata, recvdata, l.size(), detail::datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, comm, &req);
      return impl::irequest(req);
    }

    // --- blocking exscan, in place ---
    template<typename T, typename F>
    void exscan(F f, T &recvdata) const {
      MPI_Exscan(MPI_IN_PLACE, &recvdata, 1, detail::datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, comm);
    }

    template<typename T, typename F>
    void exscan(F f, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Exscan(MPI_IN_PLACE, recvdata, l.size(), detail::datatype_traits<T>::get_datatype(),
                 detail::get_op<T, F>(f).mpi_op, comm);
    }

    // --- non-blocking exscan, in place ---
    template<typename T, typename F>
    irequest iexscan(F f, T &recvdata) const {
      MPI_Request req;
      MPI_Iexscan(MPI_IN_PLACE, &recvdata, 1, detail::datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, comm, &req);
      return impl::irequest(req);
    }

    template<typename T, typename F>
    irequest iexscan(F f, T *recvdata, const contiguous_layout<T> &l) const {
      MPI_Request req;
      MPI_Iexscan(MPI_IN_PLACE, recvdata, l.size(), detail::datatype_traits<T>::get_datatype(),
                  detail::get_op<T, F>(f).mpi_op, comm, &req);
      return impl::irequest(req);
    }
  };  // namespace mpl

  //--------------------------------------------------------------------

  inline group::group(const group &other) { MPI_Group_excl(other.gr, 0, nullptr, &gr); }

  inline group::group(const communicator &comm) { MPI_Comm_group(comm.comm, &gr); }

  inline group::group(group::Union_tag, const group &other_1, const group &other_2) {
    MPI_Group_union(other_1.gr, other_2.gr, &gr);
  }

  inline group::group(group::intersection_tag, const group &other_1, const group &other_2) {
    MPI_Group_intersection(other_1.gr, other_2.gr, &gr);
  }

  inline group::group(group::difference_tag, const group &other_1, const group &other_2) {
    MPI_Group_difference(other_1.gr, other_2.gr, &gr);
  }

}  // namespace mpl

#endif
