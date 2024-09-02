AM_CPPFLAGS = -I@top_srcdir@/include
AM_CXXFLAGS = -std=c++17 -O3 -W -Wall --pedantic

bin_PROGRAMS = dimbl

dimbl_SOURCES = Dimbl.cxx DimProcs.cxx

TESTS = tst.sh

EXTRA_DIST = tst.sh tst.ok
CLEANFILES = tst.out dimin.train
