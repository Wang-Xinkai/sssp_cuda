# sssp_cuda
This is a repo for an unfinished [competition](https://cas-pra.sugon.com/com_q.html).Due to some reasons I quit the competition.

`csr_impl` is the implementation of others in our team, which is a previous work. It transforms sparse matrix into csr format to accelerate computing. 

`matrix_impl` is my trivial implementation, which is just a transformation of [cuda implementation](https://github.com/OlegKonings/CUDA_Floyd_Warshall_.git). I just transform them to the hip environment. There are two versions of sssp: one is simple and the other is blocked version. 

`run.sh` is some commands for testing on the competition environment, which equips hipcc(Nvidia YES!)
