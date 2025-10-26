Ampere A100 GPU has 108 SMs with 64 cores each, totaling 6912 cores in the entire GPU.
 The SM (Streaming Multiprocessor) counts and the number of cores per SM for the NVIDIA H100, H200, and B200 GPUs are as follows:

| GPU    | SM Count | Cores per SM | Total CUDA Cores |
|--------|----------|--------------|------------------|
| H100   | 132      | 128          | 16,896           |
| H200   | 132      | 128          | 16,896           |
| B200   | 216      | 128          | 27,648           |

- Both the H100 and H200 have 132 SMs, each with 128 CUDA cores, totaling 16,896 CUDA cores per GPU.[1][2][3][4]
- The B200 increases both the SM count (216) and maintains 128 CUDA cores per SM, totaling 27,648 CUDA cores.[5][6]

This table highlights NVIDIA's increase in parallel processing capability with each new GPU generation as it moves from Hopper (H100/H200) to Blackwell (B200).

[1](https://docs.nvidia.com/launchpad/ai/h100-mig/latest/h100-mig-gpu.html)
[2](https://www.hyperstack.cloud/technical-resources/performance-benchmarks/comparing-nvidia-h100-pcie-vs-sxm-performance-use-cases-and-more)
[3](https://www.runpod.io/articles/guides/nvidia-h200-gpu)
[4](https://2crsi.com/nvidia-h200-gpu-launch)
[5](https://glennklockwood.com/garden/processors/B200)
[6](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
[7](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
[8](https://www.nvidia.com/en-us/data-center/h100/)
[9](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf)
[10](https://www.pny.com/nvidia-h100)
[11](https://www.trgdatacenters.com/resource/nvidia-h200/)
[12](https://www.nvidia.com/en-us/data-center/h200/)
[13](https://www.serversimply.com/blog/technical-analysis-of-the-blackwell-b200)
[14](https://www.centralcomputer.com/pny-nvidia-h100-tensor-core-gpu-accelerator-80gb-hbm2e-nvh100tcgpu-kit.html)
[15](https://www.nvidia.com/en-us/data-center/dgx-b200/)
[16](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth)
[17](https://docs.nvidia.com/ai-enterprise/reference-architecture/latest/compute-node-hardware.html)
[18](https://www.civo.com/blog/comparing-nvidia-b200-and-h100)
[19](https://taknet.sg/nvidia-h200-tensor-core-gpu/)
[20](https://www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf)

Barrier synchronization imposes execution constraints on threads within a block.

In most implementations to date, once a block has been assigned to an SM, it is further divided into 32-thread units called warps.
A warp is the unit of thread scheduling in SMs.

An SM is designed to execute all threads in a warp following the single-instruction, multiple-data (SIMD) model. That is, at any instant in time, one instruction is fetched and executed for all threads in the warp (see the “Warps and SIMD Hardware” sidebar).
The advantage of SIMD is that the cost of the control hardware, such as the instruction fetch/dispatch unit, is shared across many execution units. This
when threads within a warp take different control flow paths, the SIMD hardware will take multiple passes through these paths, one pass for each path.


An important implication of control divergence is that one cannot assume that all threads in a warp have the same execution timing. Therefore if all threads in a warp must complete a phase of their execution before any of them can move on, one must use a barrier synchronization mechanism such as __syncwarp() to ensure correctness.

post office customers as warps and the clerk as a hardware execution unit.

A100 - 32 blocks per SM , 64 WARPS (or 2048 threads) per sm, 1024 thread per block

the thread slots are dynamically partionied among blocks, the number of threads per block is dynamically assigned
the block size we choose should  divide the max threads per block , otherwise the SMs will not have full 
occupancy

A100 has a max 65,536 registers per SM
The number of registers needed per thread can affect occupancy. Watch out for the number of declated autmatic 
variables.

terhe is a cuda occupancy calculator on;irr

the compute capability of a GPU is the amount of resources available in a SM, A100 - 8.0

cuda run time will tell the host program how many cuda devices are available
and the device properties 

the number of SMs in the device is given in devProp.maxThreadsPerBlock

the ratio of number of threads assigned to a SM to the max number it supports is called occupancy.


Chapter 5
- how to reduce global memory access
- tiling techniques by which barrier sync is used to improve locality and reduce global mem access, 
    - needs boundary checks
- shared mem usage and register usage can affect how many thread blocks can be accomodated in a 
streamign procesor
- DRAM is off-chip and has long access latency - 100s of clock cycles
- FLOP/Byte = compute to global access ratio - number of flops performaed for each byte access from the global memory with a region of a program. a.k.a arithmeatic intersity/computational intensity
- local memory is placed in global memory and has similar access laency as global mem but not shared
- registers and shared mems are onchip memories
- registers hold thread private frequently accessed varaible
- shared mem holds varaibles shared across threads in a thread block
- when a operand is in register, there is no extra step like load which needs to be called when the operand is in global memory
- energy consumed is less when accessing register
- when operand is in shared memory, we still need to load the operand from memory
- tiling : divide the data into smaller subsets called tiles so that each tile fits into the shared memory. Shared memory is small
- 

chapter 6 
- memory coalescing 
    - used in conjunction of tiling
    - DRAM reading takes tens of nanoseconds due to capacitive line detection.
    - Modern DRAM uses parallelism with multiple sensors accessing consecutive locations (DRAM bursts) simultaneously to go faster
    - Optimal memory access occurs when all threads in a warp access consecutive global memory locations
    - elements that appear vertically adjacent (like M0,0 and M1,0) are actually separated by the row width in linear memory becuase of row major placement in CUDA
    - In matrix multiplication, consecutive threads in a warp accessing consecutive columns create coalesced memory access because the array index k*Width+col has the same k and Width values across threads, while col (blockIdx.x*blockDim.x+threadIdx.x) varies consecutively, resulting in consecutive memory addresses.
    - if the matrix is stored in column major (like if the matrix is transposed), then the acess is not favarable for coalescing
    - two strategies when the coalasing is not natural 
        - rearrange how threads are mapped to data
        - rerrange the layout of the data
        - tranfer data from global to shared in coalased manner and do the unfavorable access patterns operations in shared mem aka corner tuning
    - gloal mem is DRAM , shared mem is SRAM
        - This transistor-based design means SRAM keeps data in memory as long as power is supplied to the system, without needing periodic refresh cycles like DRAM
    - Memory coalescing: threads combine simultaneous accesses into single DRAM requests.
    - 
- memory latency hiding
- thread granularity coalescing
- checklist of common perf optimizations
- when trading one resource to another - one must know if the traded one does alliveate the perf bottleneck. otherwise it will be guess work
- 


