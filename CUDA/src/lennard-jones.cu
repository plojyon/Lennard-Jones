#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA headers
#include <cuda_runtime.h>
#include <cuda.h>

#include "gifenc.h"
#include "lennard-jones.h"

// plotting functions
#if GENERATE_GIF
uint8_t palette[] = {
    0, 0, 0,
    255, 255, 0};

void set_pixel(uint8_t *img, int w, int h, int x, int y, uint8_t index)
{
    if (x < 0 || y < 0 || x >= w || y >= h)
    {
        return;
    }
    size_t idx = (size_t)y * (size_t)w + (size_t)x;
    img[idx] = index;
}

void render_frame_gif(ge_GIF *gif, const Particle *particles, unsigned int n, double box_size)
{

    memset(gif->frame, 0, FRAME_WIDTH * FRAME_HEIGHT);

    for (unsigned int i = 0; i < n; ++i)
    {

        int px = (int)(particles[i].x / box_size * (double)(FRAME_WIDTH - 1));
        int py = (int)(particles[i].y / box_size * (double)(FRAME_HEIGHT - 1));
        py = (FRAME_HEIGHT - 1) - py;

        for (int dy = -FRAME_PARTICLE_RADIUS; dy <= FRAME_PARTICLE_RADIUS; ++dy)
        {
            for (int dx = -FRAME_PARTICLE_RADIUS; dx <= FRAME_PARTICLE_RADIUS; ++dx)
            {
                if (dx * dx + dy * dy <= FRAME_PARTICLE_RADIUS * FRAME_PARTICLE_RADIUS)
                {
                    set_pixel(gif->frame, FRAME_WIDTH, FRAME_HEIGHT, px + dx, py + dy, 1);
                }
            }
        }
    }
}
#endif
double random_double(void)
{
    return (double)rand() / (double)RAND_MAX;
}

// __device__ double atomicAddPreSM60(double* address, double val)
// {
//     unsigned long long int* address_as_ull = (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;
//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed,
//                 __double_as_longlong(val + __longlong_as_double(assumed)));
//     } while (assumed != old);
//     return __longlong_as_double(old);
// }

// from dotprod4.cu
__global__ void compute_ke_kernel(const Particle *d_particles, unsigned int n, double *result)
{
    extern __shared__ double part[];
    part[threadIdx.x] = 0.0;

    // Each thread sums some partial values to produce a THREADS sized array of partial sums
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    while (tid < n)
    {
        const Particle *p = &d_particles[tid];
        part[threadIdx.x] += 0.5 * (p->vx * p->vx + p->vy * p->vy);
        tid += blockDim.x * gridDim.x;
    }

    __syncthreads();

    // On each step reduce the partial sum array by half by assigning each thread to sum two values
    int idxStep;
    for (idxStep = blockDim.x / 2; idxStep > 0; idxStep /= 2)
    {
        if (threadIdx.x < idxStep)
        {
            part[threadIdx.x] += part[threadIdx.x + idxStep];
        }
        __syncthreads();
    }

    // The last thread has to add the result to result (this is a block result in a grid of blocks)
    if (threadIdx.x == 0)
    {
        atomicAdd(result, part[0]);
    }
}

double compute_ke(const Particle *d_particles, unsigned int n, int threads)
{
    double *d_ke;
    double ke = 0.0;

    cudaMalloc(&d_ke, sizeof(double));
    cudaMemset(d_ke, 0, sizeof(double));

    int blocks = (n + threads - 1) / threads;
    compute_ke_kernel<<<blocks, threads, threads>>>(d_particles, n, d_ke);
    cudaDeviceSynchronize();
    cudaMemcpy(&ke, d_ke, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_ke);

    return ke;
}

int initialize_particles(Particle *particles, unsigned int n, double box_size, double placement_fraction, unsigned int seed, double temperature)
{

    srand(seed);
    unsigned int n_side = (unsigned int)ceil(sqrt((double)n));
    double placement_size = placement_fraction * box_size;
    double offset = 0.5 * (box_size - placement_size);
    double delta = placement_size / (double)n_side;

    double mean_vx = 0.0;
    double mean_vy = 0.0;
    // place particles in the middle of the grid with some random jitter and assign random velocities
    for (unsigned int k = 0; k < n; k++)
    {
        double x0 = offset + (0.5 + (double)(k % n_side)) * delta;
        double y0 = offset + (0.5 + (double)(k / n_side)) * delta;

        particles[k].x = x0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].y = y0 + (2.0 * random_double() - 1.0) * JITTER * delta;

        particles[k].vx = 2.0 * random_double() - 1.0;
        particles[k].vy = 2.0 * random_double() - 1.0;

        mean_vx += particles[k].vx;
        mean_vy += particles[k].vy;
    }

    mean_vx /= (double)n;
    mean_vy /= (double)n;
    double ke = 0.0;
    // subtract mean velocity to ensure zero net momentum and compute initial kinetic energy
    for (unsigned int k = 0; k < n; k++)
    {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        ke += 0.5 * (particles[k].vx * particles[k].vx +
                     particles[k].vy * particles[k].vy);
    }

    double current_temperature = ke / (double)n;
    if (current_temperature <= 0.0)
    {
        return 0;
    }

    // scale velocities to match the desired initial temperature of the system
    double scale = sqrt(temperature / current_temperature);
    for (unsigned int k = 0; k < n; k++)
    {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
    }

    return 1;
}

// apply periodic boundary conditions to ensure particles stay within the simulation box
__global__ void wrap_positions_kernel(Particle *particles, unsigned int n, double box_size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    Particle *p = &particles[i];

    double wx = fmod(p->x, box_size);
    double wy = fmod(p->y, box_size);

    if (wx < 0.0)
        wx += box_size;

    if (wy < 0.0)
        wy += box_size;

    p->x = wx;
    p->y = wy;
}
// shift potential to ensure it goes to zero at the cutoff distance, improving energy conservation
double compute_v_shift(void)
{
    return 4.0 * EPSILON * (pow(SIGMA / R_CUT, 12.0) - pow(SIGMA / R_CUT, 6.0));
}

__device__ double v_shift;
__global__ void compute_forces_internal(Particle *particles, unsigned int n, double box_size, double *pe_out)
{
    extern __shared__ double shared[];
    double *sh_x = shared;
    double *sh_y = shared + blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    double fxi = 0.0;
    double fyi = 0.0;

    Particle *pi = &particles[i];
    double pe = 0.0;

    pi->fx = 0.0;
    pi->fy = 0.0;

    for (int tile = 0; tile < n; tile += blockDim.x)
    {
        // We care about the details
        if (tile != 0)
            __syncthreads();

        int shared_j = tile + threadIdx.x;

        sh_x[threadIdx.x] = 0.0;
        sh_y[threadIdx.x] = 0.0;

        // load tile into shared memory
        if (shared_j < n)
        {
            sh_x[threadIdx.x] = particles[shared_j].x;
            sh_y[threadIdx.x] = particles[shared_j].y;
        }

        __syncthreads();

        int tile_size = min(blockDim.x, n - tile);
        for (int k = 0; k < tile_size; k++)
        {
            int j = tile + k;
            if (j == i)
            {
                continue;
            }

            // compute distance between particles with periodic boundary conditions
            double dx = pi->x - sh_x[k];
            double dy = pi->y - sh_y[k];

            dx -= box_size * nearbyint(dx / box_size);
            dy -= box_size * nearbyint(dy / box_size);

            // compute Lennard-Jones force and potential energy contribution if particles are within the cutoff distance
            double r = dx * dx + dy * dy;
            if (r >= R_CUT * R_CUT || r == 0.0)
            {
                continue;
            }
            double sr = SIGMA * SIGMA / r;

            double fij = 24.0 * EPSILON * (2.0 * pow(sr, 6.0) - pow(sr, 3.0)) / r;
            double fx = fij * dx;
            double fy = fij * dy;

            fxi += fx;
            fyi += fy;

            double vij = 4.0 * EPSILON * (pow(sr, 6.0) - pow(sr, 3.0)) - v_shift;
            pe += 0.5 * vij;
        }
    }

    pi->fx = fxi;
    pi->fy = fyi;

    // Reduce per block
    shared[threadIdx.x] = pe;

    __syncthreads();

    for (int idxStep = blockDim.x / 2; idxStep > 0; idxStep /= 2)
    {
        if (threadIdx.x < idxStep)
        {
            shared[threadIdx.x] += shared[threadIdx.x + idxStep];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(pe_out, pe); // replace with atomicAdd if compute capability > 6.0
    }
}

double compute_forces(Particle *d_particles, unsigned int n, double box_size, int threads)
{
    int blocks = (n + threads - 1) / threads;
    size_t shared_mem_size = 2 * threads * sizeof(double);

    // Initialize potential energy on device to 0
    double *d_pe;
    cudaMalloc((void **)&d_pe, sizeof(double));
    cudaMemset(d_pe, 0, sizeof(double));

    // Launch kernel
    compute_forces_internal<<<blocks, threads, shared_mem_size>>>(
        d_particles, n, box_size, d_pe);

    cudaDeviceSynchronize();

    // Copy result back
    double pe;
    cudaMemcpy(&pe, d_pe, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_pe);
    return pe;
}

__global__ void half_leapfrog(Particle *d_particles, unsigned int n, int do_drift)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    Particle *p = &d_particles[i];

    p->vx += 0.5 * DT * p->fx;
    p->vy += 0.5 * DT * p->fy;

    if (do_drift)
    {
        p->x += DT * p->vx;
        p->y += DT * p->vy;
    }
}

double leapfrog_step(Particle *d_particles, unsigned int n, double box_size, int threads)
{
    int blocks = (n + threads - 1) / threads;

    // update velocities by half a time step, then update positions by a full time step,
    // and finally update velocities by another half time step to complete the leapfrog integration step
    half_leapfrog<<<blocks, threads>>>(d_particles, n, 1);
    wrap_positions_kernel<<<blocks, threads>>>(d_particles, n, box_size);
    double pe = compute_forces(d_particles, n, box_size, threads);
    half_leapfrog<<<blocks, threads>>>(d_particles, n, 0);

    return pe;
}

SimulationResult run_simulation(Particle *particles, unsigned int n, unsigned int nsteps, double box_size, int log_steps, int threads)
{
    // Move particles to device
    Particle *d_particles;
    cudaMalloc((void **)&d_particles, n * sizeof(Particle));
    cudaMemcpy(d_particles, particles, n * sizeof(Particle), cudaMemcpyHostToDevice);

    // Initialize v_shift on device
    double cpu_v_shift = compute_v_shift();
    cudaMemcpyToSymbol(v_shift, &cpu_v_shift, sizeof(double));

    SimulationResult out;
    out.start_potential = compute_forces(d_particles, n, box_size, threads);
    out.start_kinetic = compute_ke(d_particles, n, threads);
    out.start_total = out.start_kinetic + out.start_potential;

#if GENERATE_GIF
    ge_GIF *gif = NULL;

    gif = ge_new_gif(GIF_FILE, (uint16_t)FRAME_WIDTH, (uint16_t)FRAME_HEIGHT, palette, 8, -1, 0);
    if (!gif)
    {
        fprintf(stderr, "Warning: failed to create GIF output %s\n", GIF_FILE);
    }
    else
    {
        cudaDeviceSynchronize();
        cudaMemcpy(particles, d_particles, n * sizeof(Particle), cudaMemcpyDeviceToHost);
        render_frame_gif(gif, particles, n, box_size);
        ge_add_frame(gif, FRAME_DELAY);
    }
#endif

    for (unsigned int step = 0; step < nsteps; step++)
    {
        out.final_potential = leapfrog_step(d_particles, n, box_size, threads);
        out.final_kinetic = compute_ke(d_particles, n, threads);
        out.final_total = out.final_kinetic + out.final_potential;
        if (log_steps)
        {
            printf(
                "step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                step,
                out.final_kinetic,
                out.final_potential,
                out.final_total);
        }

#if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0)
        {
            cudaDeviceSynchronize();
            cudaMemcpy(particles, d_particles, n * sizeof(Particle), cudaMemcpyDeviceToHost);
            render_frame_gif(gif, particles, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }

#if GENERATE_GIF
    if (gif)
    {
        ge_close_gif(gif);
    }
#endif

    cudaDeviceSynchronize();
    cudaMemcpy(particles, d_particles, n * sizeof(Particle), cudaMemcpyDeviceToHost);
    out.n = n;
    out.particles = particles;
    return out;
}