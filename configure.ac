#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([qmckl_bench_gpu], [0.0.1], [])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_SRCDIR([configure.ac])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([include/config.h])
AM_INIT_AUTOMAKE([ foreign subdir-objects color-tests parallel-tests silent-rules 1.11])

AM_PROG_AR
AM_MAINTAINER_MODE()

# Checks for programs.
AC_LANG(C++)
AC_PROG_CXX([icpx clang++])
AM_PROG_CC_C_O
AC_PROG_INSTALL
AC_PROG_LN_S

AC_ARG_WITH(qmckl_gpu, [AS_HELP_STRING([--with-qmckl_gpu],[Specify installation directory of QMCKL GPU])],
            with_qmckl_gpu=$withval, with_qmckl_gpu=yes)


####
## Checks for TREXio / libmath
####
AC_CHECK_LIB([m], [sqrt])
AC_CHECK_LIB([trexio], [trexio_open], [],
      [ AC_MSG_ERROR([TREXIO not found]) ])

####
## Check for QMCkl GPU
#####
AS_IF([test "$with_qmckl_gpu" = "yes"],
  [
    AC_CHECK_LIB([qmckl_gpu], [qmckl_context_create_device], [],
      [ AC_MSG_ERROR([QMCKL GPU not found]) ]
    )
  ], [
    QMCKL_GPU_LIBS="${with_qmckl_gpu}/lib/libqmckl_gpu.a" # Always using the static lib to avoid issues
    QMCKL_GPU_CFLAGS="-I${with_qmckl_gpu}/include -lm"
    QMCKL_GPU_LDFLAGS="-L=${with_qmckl_gpu}/lib -lm"
  ])

LIBS="${QMCKL_GPU_LIBS} ${LIBS}"
LDFLAGS="${QMCKL_GPU_LDFLAGS} ${LDFLAGS}"
CXXFLAGS="${QMCKL_GPU_CFLAGS} ${CXXFLAGS} "
AC_DEFINE_UNQUOTED(DATA_DIR, ["`cd ${srcdir}/data && echo $(pwd -P)`"], [Path to the data files])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h qmckl_gpu.h])



# Auto offloading flags
AC_ARG_ENABLE(autoflags, [AS_HELP_STRING([--disable-autoflags],[By default, configure will try to infer and add offloading compiler flags by itself. Disable this option and set your flags manually if you encounter issues.])], have_autoflags=no, have_autoflags=yes)
AS_IF([test "x$have_autoflags" = "xyes"], [
	echo "Autoflags have been kept enabled"
], [
	echo "Autoflags have been disabled"
])



AC_CONFIG_FILES([Makefile])

AC_OUTPUT

####
# Print configure status
####

AC_OUTPUT

echo \
"-------------------------------------------------
${PACKAGE_NAME} Version ${PACKAGE_VERSION}

Prefix: '${prefix}'

CXX.................: ${CXX}
CXXFLAGS.............: ${CXXFLAGS}
LDFLAGS:...........: ${LDFLAGS}

Autoflags .........: ${have_autoflags}

--------------------------------------------------"