#include <iostream>
#include <mpi.h>

int main(int argc,char** argv){
	int w_rank,w_nprocs;
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &w_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &w_nprocs);

	std::cout<<w_rank<<std::endl;

	MPI_Finalize();
}
