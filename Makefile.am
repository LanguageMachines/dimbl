ACLOCAL_AMFLAGS = -I m4 --install

SUBDIRS = src include examples m4 docs

EXTRA_DIST = bootstrap.sh AUTHORS TODO NEWS

ChangeLog: NEWS
	git pull; git2cl > ChangeLog
