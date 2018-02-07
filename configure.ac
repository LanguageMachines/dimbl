#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.61)
AC_INIT([dimbl],[0.16],[lamasoftware@science.ru.nl])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([configure.ac])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADER([config.h])

if test x"${CXXFLAGS+set}" = xset; then
  # the user set CXXFLAGS; don't override it.
  cxx_flags_were_set=true
else
  cxx_flags_were_set=false
fi

if $cxx_flags_were_set; then
  CXXFLAGS=$CXXFLAGS
fi


# Checks for programs.
AC_PROG_CXX( [g++] )
AC_PROG_LIBTOOL
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
if test "x$PKG_CONFIG_PATH" = x; then
    export PKG_CONFIG_PATH="$prefix/lib/pkgconfig"
else
    export PKG_CONFIG_PATH="$prefix/lib/pkgconfig:$PKG_CONFIG_PATH"
fi

# Checks for timbl library.
PKG_CHECK_MODULES([timbl], [timbl >= 6.4.4] )
CXXFLAGS="$CXXFLAGS $timbl_CFLAGS"
LIBS="$LIBS $timbl_LIBS"

# Checks for ticcutils library.
PKG_CHECK_MODULES([ticcutils], [ticcutils >= 0.10] )
CXXFLAGS="$CXXFLAGS $ticcutils_CFLAGS"
LIBS="$LIBS $ticcutils_LIBS"

AC_CHECK_FUNCS([gettimeofday])
AC_C_INLINE
AC_HEADER_STDBOOL
AC_TYPE_SIZE_T

AC_OUTPUT([
  Makefile
  m4/Makefile
  docs/Makefile
  src/Makefile
  include/Makefile
  include/dimbl/Makefile
  examples/Makefile
])
