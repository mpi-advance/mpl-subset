#if !(defined MPL_ERROR_HPP)

#define MPL_ERROR_HPP

#include <exception>

namespace mpl {

  /// Base class for all MPL exception classes that will be thrown in case of run-time errors.
  class error : public ::std::exception {
  protected:
    const char *const str;

  public:
    /// \param str error message that will be returned by #what method
    explicit error(const char *const str = "unknown") : str(str) {}

    /// \return character pointer to error message
    [[nodiscard]] const char *what() const noexcept override { return str; }
  };

  /// Will be thrown when an error occurs while manipulating layouts.
  class invalid_datatype_bound : public error {
  public:
    invalid_datatype_bound() : error("invalid datatype bound") {}
  };

}  // namespace mpl

#endif