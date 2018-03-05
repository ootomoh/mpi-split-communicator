#include <iostream>
#include <mpi.h>


int main(int argc,char** argv){
	MPI_Init(&argc, &argv);

	MPI_Group world_group;
	MPI_Group grad_group;
	MPI_Group inv_group;

	int world_rank,world_nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_nprocs);
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);

	int num_invs = 3;
	int num_grads = world_nprocs - num_invs + 1;
	int group_rank;

	int *inv_ranks = new int [world_nprocs];
	for(int i = 0;i < num_invs;i++){
		inv_ranks[i] = i;
	}
	std::string message = "world_rank = " + std::to_string(world_rank);

	MPI_Group_incl(world_group, num_invs, inv_ranks,&inv_group);
	MPI_Group_excl(world_group, num_invs-1, inv_ranks+1,&grad_group);
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Group_rank(grad_group, &group_rank);
	if( group_rank != MPI_UNDEFINED){
		message += ", grad_rank = " + std::to_string(group_rank);
	}
	MPI_Group_rank(inv_group, &group_rank);
	if( group_rank != MPI_UNDEFINED){
		message += ", inv_rank = " + std::to_string(group_rank);
	}

	std::cout<<message<<std::endl;

	MPI_Finalize();
	delete [] inv_ranks;
}
