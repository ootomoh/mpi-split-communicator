#include <iostream>
#include <mpi.h>

const int num_invs = 4;

class MPIGroup{
public:
	MPI_Comm comm;
	MPI_Group group;
	int rank;
	int size;
};

int main(int argc,char **argv){
	MPI_Init(&argc,&argv);

	MPIGroup g_world,g_invs,g_masters,g_grads;

	MPI_Comm_rank(MPI_COMM_WORLD, &g_world.rank);
	MPI_Comm_size(MPI_COMM_WORLD, &g_world.size);
	MPI_Comm_group(MPI_COMM_WORLD, &g_world.group);

	// 0 -> grad master
	// 1 -> inv master
	if( g_world.size < 2){
		std::cerr<<"もっとプロセスを食わせろ"<<std::endl;
		MPI_Finalize();
		return 1;
	}

	int masters_ranks[] = {0,1};
	MPI_Group_incl(g_world.group, 2, masters_ranks, &g_masters.group);
	MPI_Comm_create(MPI_COMM_WORLD, g_masters.group, &g_masters.comm);

	int *invs_ranks = new int [num_invs];
	for(int i = 0;i < num_invs;i++)invs_ranks[i] = i+1;
	MPI_Group_incl(g_world.group, num_invs, invs_ranks, &g_invs.group);
	MPI_Comm_create( MPI_COMM_WORLD, g_invs.group, &g_invs.comm);

	MPI_Group_excl(g_world.group, num_invs, invs_ranks, &g_grads.group);
	MPI_Comm_create( MPI_COMM_WORLD, g_grads.group, &g_grads.comm);


	// 各グループでの自分のグループ番号を取得
	MPI_Group_rank(g_masters.group, &g_masters.rank);
	MPI_Group_rank(g_invs.group, &g_invs.rank);
	MPI_Group_rank(g_grads.group, &g_grads.rank);

	if( g_masters.rank != MPI_UNDEFINED ){
		int val = g_masters.rank;
		int sum;
		MPI_Reduce( &val, &sum,1, MPI_INT, MPI_SUM, 0, g_masters.comm);
		if(g_masters.rank == 0){
			std::cout<<"masters reduce ; "<<sum<<std::endl;
		}
	}
	MPI_Barrier( MPI_COMM_WORLD );

	if( g_invs.rank != MPI_UNDEFINED ){
		int val = g_invs.rank;
		int sum;
		MPI_Reduce( &val, &sum,1, MPI_INT, MPI_SUM, 0, g_invs.comm);
		if(g_invs.rank == 0){
			std::cout<<"invs reduce ; "<<sum<<std::endl;
		}
	}
	if( g_grads.rank != MPI_UNDEFINED ){
		int val = g_grads.rank;
		int sum;
		MPI_Reduce( &val, &sum,1, MPI_INT, MPI_SUM, 0, g_grads.comm);
		if(g_grads.rank == 0){
			std::cout<<"grads reduce ; "<<sum<<std::endl;
		}
	}

	delete [] invs_ranks;
	MPI_Finalize();
}
