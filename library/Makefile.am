lib_LTLIBRARIES = libvwjni.la

libvwjni_la_SOURCES = VwJniBinding.cc

libvwjni_la_LIBADD = ../vowpalwabbit/libvw.la ../vowpalwabbit/liballreduce.la

INCLUDES = ${BOOST_CPPFLAGS} ${ZLIB_CPPFLAGS} ${PTHREAD_CFLAGS}
AM_LDFLAGS = ${BOOST_LDFLAGS} ${BOOST_PROGRAM_OPTIONS_LIB} ${ZLIB_LDFLAGS} ${PTHREAD_LIBS}
