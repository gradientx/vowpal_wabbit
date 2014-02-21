#!/bin/sh
AC_PATH=/usr/share
[ -x "`which g++`" ] && CXX=g++
[ -x "`which clang++`" ] && CXX=clang++
case $( uname -s ) in
 Darwin)  alias vwlibtool=glibtoolize
<<<<<<< Upstream, based on upstream/master
	 if [ -d /opt/local/share ]; then
     AC_PATH="/opt/local/share"
  else
     AC_PATH="/usr/local/share"
  fi;;
=======
	AC_PATH=/usr/local/share;;
>>>>>>> b955e93 Adding some regressor dumping functionality to VW 7.4
 *)	alias vwlibtool=libtoolize;;
esac

vwlibtool -f -c && aclocal -I ./acinclude.d -I $AC_PATH/aclocal && autoheader && touch README && automake -ac -Woverride && autoconf && ./configure "$@" CXX=$CXX
