### MAKEFILE FOR CREATING PARGRID DISTRIBUTION ###
###               DO NOT EDIT                  ###

# Distribution package name
DIST=pargrid_v01_000.tar

# Build targets

clean:
	rm -rf *~ *.tar.gz

dist:
	tar -rf ${DIST} Makefile
	tar -rf ${DIST} COPYING
	tar -rf ${DIST} *.h
	tar -rf ${DIST} *.pdf
	gzip -9 ${DIST}
