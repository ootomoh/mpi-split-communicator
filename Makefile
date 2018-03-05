MPICPP=mpic++
MPICPPFLAGS=-std=c++11
BIN=mpi-split-com

$(BIN):main.cpp
	$(MPICPP) $(MPICPPFLAGS) -o $@ @<
