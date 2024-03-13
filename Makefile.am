ACLOCAL_AMFLAGS = -I m4
AM_CXXFLAGS = -fsycl -g  -I$(srcdir)/data 

bin_PROGRAMS = bench_aos bench_mos


bench_aos_SOURCES = src/bench_aos.cpp
bench_mos_SOURCES = src/bench_mos.cpp


clean-local: 
	-rm -rf autom4te.cache/ m4/
	-rm -f aclocal.m4 
	-rm -f config.* configure 
	-rm -f Makefile  Makefile.in
