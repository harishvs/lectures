number of 
kernel<<<gridSize, blockSize>>>(args);
kernel<<<dimGrid,dimBlock>>>
k<<<8,128>>>
8 blocks 
128 threads in each block

a. 128/32 = 4
b. 8 * 4 = 32
c.1 = 32 , all warps are active. The threads have diverged , but warp it self is still active.
c.ii = 16
    Per block (128 threads = 4 warps):

    Warp 0 (threads 0-31): All execute line 04 (all < 40) → NOT divergent

    Warp 1 (threads 32-63): All execute line 04 (32-39 < 40, rest are 40-63) → Wait, let me recalculate...

    Threads 32-39: execute (< 40) ✓
    Threads 40-63: DON'T execute (40-103 range) ✗
    → DIVERGENT
    Warp 2 (threads 64-95): None execute line 04 (all in 40-103 range) → NOT divergent

    Warp 3 (threads 96-127):

    Threads 96-103: DON'T execute (in 40-103 range) ✗
    Threads 104-127: execute (>= 104) ✓
    → DIVERGENT
    Per block: 2 divergent warps (Warp 1 and Warp 3)

    Across entire grid:

    8 blocks × 2 divergent warps/block = 16 divergent warps

SIMD Efficiency = (Average active threads per warp) / (Warp size)
c.iii  = Warp 0 (threads 0-31): 32/32 active = 100% efficient 100
c.iv = Warp 1 (threads 32-63): 8/32 active (only 32-39 execute) = 25% efficient
c.v = Warp 3 (threads 96-127): 24/32 active (104-127 execute) = 75% efficient

d.i = all 32 are active
d.ii = 32
d.iii = 50%
    Setup:

    Block size: 128 threads
    Grid size: (N + 128)/128 = (1024 + 128)/128 = 9 blocks
    Warp size: 32 threads (standard CUDA)
    Threads per block: 128, so 128/32 = 4 warps per block
    Line 07 analysis: a[i] = b[i]*2;

    This statement is inside the condition if(i%2 == 0), which means only threads with even indices execute this line.

    d.i. How many warps in the grid are active?

    Total warps in grid = 9 blocks × 4 warps/block = 36 warps

    All 36 warps are active because they all reach this statement (even though not all threads within each warp execute it).

    d.ii. How many warps in the grid are divergent?

    Within each warp (32 consecutive threads), half will have even threadIdx and half will have odd threadIdx. Since the condition is i%2 == 0, threads alternate between executing and not executing.

    All 36 warps are divergent because each warp contains both threads that satisfy the condition (even i) and threads that don't (odd i).

    d.iii. What is the SIMD efficiency (in %) of warp 0 of block 0?

    Warp 0 of block 0 contains threads 0-31.

    Threads with even i: 0, 2, 4, ..., 30 = 16 threads execute
    Threads with odd i: 1, 3, 5, ..., 31 = 16 threads don't execute
    SIMD efficiency = (active threads / total threads) × 100 = (16/32) × 100 = 50%

e. 
        Looking at line 09 from the original code:

        for(unsigned int j = 0; j < 5 - (i%3); ++j) {
            b[i] = j;
        }
        The loop bound depends on i%3, which gives different iteration counts for different threads:

        Iteration count per thread:

        If i % 3 == 0: loop runs 5 - 0 = 5 iterations
        If i % 3 == 1: loop runs 5 - 1 = 4 iterations
        If i % 3 == 2: loop runs 5 - 2 = 3 iterations
        Within a warp (32 consecutive threads):

        Since threads in a warp have consecutive 
        README.md
        values, and i % 3 cycles through 0, 1, 2, 0, 1, 2..., every warp will have threads with different iteration counts.

        Analyzing divergence per iteration:

        Iteration j=0: All threads (5, 4, and 3 iteration threads) execute → NO divergence
        Iteration j=1: All threads (5, 4, and 3 iteration threads) execute → NO divergence
        Iteration j=2: All threads (5, 4, and 3 iteration threads) execute → NO divergence
        Iteration j=3: Only threads with 5 and 4 iterations execute (threads with i%3==2 exit) → DIVERGENCE
        Iteration j=4: Only threads with 5 iterations execute (threads with i%3==1 exit) → DIVERGENCE
        Answer:

        e.i. Iterations with NO divergence: 3 (j = 0, 1, 2)
        e.ii. Iterations with divergence: 2 (j = 3, 4)

2. ans = 2048 
3. total warps = 2048/32 = 64 warps total
1 warp
4. 
        Let me calculate the barrier waiting time:

    Thread execution times (μs):

    Thread 0: 2.0
    Thread 1: 2.3
    Thread 2: 3.0 ← slowest (determines barrier time)
    Thread 3: 2.8
    Thread 4: 2.4
    Thread 5: 1.9 ← fastest
    Thread 6: 2.6
    Thread 7: 2.9
    Barrier synchronization: All threads must wait for the slowest thread (3.0 μs) before proceeding.

    Wait time per thread:

    Thread 0: 3.0 - 2.0 = 1.0 μs
    Thread 1: 3.0 - 2.3 = 0.7 μs
    Thread 2: 3.0 - 3.0 = 0.0 μs (no wait)
    Thread 3: 3.0 - 2.8 = 0.2 μs
    Thread 4: 3.0 - 2.4 = 0.6 μs
    Thread 5: 3.0 - 1.9 = 1.1 μs
    Thread 6: 3.0 - 2.6 = 0.4 μs
    Thread 7: 3.0 - 2.9 = 0.1 μs
    Total wait time: 1.0 + 0.7 + 0.0 + 0.2 + 0.6 + 1.1 + 0.4 + 0.1 = 4.1 μs

    Total execution time: 8 threads × 3.0 μs = 24.0 μs

    Percentage waiting:

    (4.1 / 24.0) × 100 = 17.08%
    Answer: ~17.1% of total execution time is spent waiting for the barrier

    This shows the inefficiency of barrier synchronization when threads have imbalanced workloads - nearly 1/6 of the time is wasted waiting.

    4. 
    The programmer is thinking: "32 threads = 1 warp, and warps execute in lockstep (SIMT), so they're already synchronized."

    In summary, never omit barrier synchronization merely because your block size matches the warp size. Synchronization should be dictated by the need to avoid race conditions and ensure correct inter-thread communication, not just hardware lockstepping.

    5.
            Constraints:

        Max threads per SM: 1536
        Max blocks per SM: 4
        Analysis:

        a. 128 threads per block

        Blocks that fit: min(4, floor(1536/128)) = min(4, 12) = 4 blocks
        Total threads: 4 × 128 = 512 threads
        b. 256 threads per block

        Blocks that fit: min(4, floor(1536/256)) = min(4, 6) = 4 blocks
        Total threads: 4 × 256 = 1024 threads
        c. 512 threads per block

        Blocks that fit: min(4, floor(1536/512)) = min(4, 3) = 3 blocks
        Total threads: 3 × 512 = 1536 threads ✓
        d. 1024 threads per block

        Blocks that fit: min(4, floor(1536/1024)) = min(4, 1) = 1 block
        Total threads: 1 × 1024 = 1024 threads
        Answer: c. 512 threads per block gives the most threads (1536)

9.
    CUDA Thread Hierarchy
    Threads: The smallest unit of execution; each runs the same kernel code.

    Thread Block: A group of threads that can cooperate via shared memory and synchronization.

    Example: A 16×16 thread block = 256 threads.

    Grid: A collection of many blocks that together cover all the data you need to process.

    When launching a CUDA kernel, you specify both:

    cpp
    dim3 threadsPerBlock(16, 16);  // 256 threads per block
    dim3 numBlocks(64, 64);         // 4,096 blocks in total
    myKernel<<<numBlocks, threadsPerBlock>>>(...);
    This setup means:

    Each block has 256 threads.

    The entire grid has 
    64
    ×
    64
    =
    4096
    64×64=4096 blocks.

    The total number of threads is 
    4096
    ×
    256
    =
    1
    ,
    048
    ,
    576
    4096×256=1,048,576 — enough to compute a 1024×1024 matrix.

    So in short:

    “16×16 block” = 256 threads in one block.

    Number of blocks is specified separately in the grid configuration.**

    This separation allows CUDA to handle huge problems efficiently while respecting per-block hardware limits.​


