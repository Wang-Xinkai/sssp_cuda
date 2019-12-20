rocm-smi --showmeminfo vram
sinfo |grep "caspra"
salloc -p caspra  -N 1

hipcc shortest_path.cpp test2.cpp -o b;srun -p caspra -n 1 ./b
hipcc shortest_path.cpp test.cpp -o a;srun -p caspra -n 1 ./a

hipcc shortest_path.cpp test.cpp -o a -I . -I /opt/rocm/hip/include/
srun -p caspra -n 1 ./a