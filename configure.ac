#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([dimbl],[0.18],[lamasoftware@science.ru.nl]) #also adapt in codemeta.json!
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([configure.ac])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])

AX_REQUIRE_DEFINED([AX_CXX_COMPILE_STDCXX_17])

# Checks for programs.
AC_PROG_CXX( [g++] )
AX_CXX_COMPILE_STDCXX_17

# use libtool
LT_INIT

# when running tests, use CXX
AC_LANG([C++])

# check OpenMP support
AC_OPENMP
if test "x$ac_cv_prog_cxx_openmp" != "x"; then
  if test "x$ac_cv_prog_cxx_openmp" != "xunsupported"; then
    CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS"
    AC_DEFINE([HAVE_OPENMP], [1] , [Define to 1 if you have OpenMP] )
  else
    AC_MSG_NOTICE([We don't have OpenMP. Multithreaded operation is disabled])
  fi
fi

# Checks for header files.
AC_CHECK_HEADERS([sys/time.h])

PKG_PROG_PKG_CONFIG
AC_MSG_NOTICE([PKG_CONFIG_PATH=$PKG_CONFIG_PATH])
if test "x$prefix" = "xNONE"; then
   prefix="/usr/local"
fi

if test "x$PKG_CONFIG_PATH" = x; then
    export PKG_CONFIG_PATH="$prefix/lib/pkgconfig"
else
    export PKG_CONFIG_PATH="$prefix/lib/pkgconfig:$PKG_CONFIG_PATH"
fi

AC_MSG_NOTICE([PKG_CONFIG_PATH=$PKG_CONFIG_PATH])
AC_OSX_PKG( [icu4c] )

# Checks for timbl library.
PKG_CHECK_MODULES([timbl], [timbl >= 6.8] )
CXXFLAGS="$CXXFLAGS $timbl_CFLAGS"
LIBS="$LIBS $timbl_LIBS"

# Checks for ticcutils library.
PKG_CHECK_MODULES([ticcutils], [ticcutils >= 0.22] )
CXXFLAGS="$CXXFLAGS $ticcutils_CFLAGS"
LIBS="$LIBS $ticcutils_LIBS"

PKG_CHECK_MODULES([ICU], [icu-uc >= 50 icu-io] )
CXXFLAGS="$CXXFLAGS $ICU_CFLAGS"
LIBS="$ICU_LIBS $LIBS"

AC_CHECK_FUNCS([gettimeofday])
AC_C_INLINE
AC_HEADER_STDBOOL
AC_TYPE_SIZE_T

AC_CONFIG_FILES([
  Makefile
  m4/Makefile
  docs/Makefile
  src/Makefile
  include/Makefile
  include/dimbl/Makefile
  examples/Makefile
])
AC_OUTPUT
