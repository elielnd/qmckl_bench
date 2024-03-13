#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <qmckl_gpu.hpp>
#include <trexio.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <unistd.h>
#include <sys/time.h>

#include <time.h>


#define ITERMAX 10
#define DEVICE_ID 0

using namespace sycl;

int main(int argc, char **argv)
{
    queue q;
    try
    {
        // Try to initialize the queue with the GPU selector
        q = queue(gpu_selector_v);
        std::cout << "Running on GPU: " << q.get_device().get_info<info::device::name>() << "\n";
    }
    catch (const sycl::exception &accelerator_exception)
    {
        try
        {
            // If initialization with the GPU selector fails as well, use the accelerator selector
            q = queue(accelerator_selector_v);
            std::cout << "Running on ACCELERATOR: " << q.get_device().get_info<info::device::name>() << "\n";
        }
        catch (const sycl::exception &gpu_exception)
        {
            // If initialization with the accelerator selector fails as well, use the host selector
            q = queue();
            std::cerr << "Warning: No device found";
            std::cout << "Running on Host: " << q.get_device().get_info<info::device::name>() << "\n";
        }
    };

    long start, end;
    struct timeval timecheck;

    qmckl_context_device context;
    context = qmckl_context_create_device(q);
    qmckl_exit_code_device rc;
    if (argc < 2)
    {
        fprintf(stderr, "Syntax: %s FILE.    (FILEs are data/*.h5)\n", argv[0]);
        exit(-1);
    }
    char *file_name = argv[1];
    trexio_t *trexio_file = trexio_open(file_name, 'r', TREXIO_HDF5, &rc);
    assert(rc == TREXIO_SUCCESS);

    int walk_num, elec_num;
    rc = trexio_read_electron_num(trexio_file, &elec_num);
    assert(rc == TREXIO_SUCCESS);
    rc = trexio_read_qmc_num(trexio_file, &walk_num);
    assert(rc == TREXIO_SUCCESS);

    double *elec_coord = (double *)malloc(sizeof(double) * walk_num * elec_num * 3);
    double *elec_coord_device = (double *)qmckl_malloc_device(context, sizeof(double) * walk_num * elec_num * 3);

    assert(elec_coord != NULL);
    rc = trexio_read_qmc_point(trexio_file, elec_coord);
    assert(rc == TREXIO_SUCCESS);
    trexio_close(trexio_file);

    qmckl_memcpy_H2D(context, elec_coord_device, elec_coord,
                     sizeof(double) * walk_num * elec_num * 3);

    printf("Reading %s.\n", file_name);
    rc = qmckl_trexio_read_device(context, file_name, 255);
    assert(rc == QMCKL_SUCCESS_DEVICE);

    int64_t mo_num;
    rc = qmckl_get_mo_basis_mo_num_device(context, &mo_num);
    assert(rc == QMCKL_SUCCESS_DEVICE);

    const int64_t size_max = 5 * walk_num * elec_num * mo_num;
    double *mo_vgl = (double *)qmckl_malloc_device(context, size_max * sizeof(double));
    assert(mo_vgl != NULL);

    rc = qmckl_set_electron_coord_device(
        context, 'N', walk_num, elec_coord_device, walk_num * elec_num * 3);
    assert(rc == QMCKL_SUCCESS_DEVICE);

    rc = qmckl_get_mo_basis_mo_vgl_device(context, mo_vgl, size_max);
    assert(rc == QMCKL_SUCCESS_DEVICE);

    gettimeofday(&timecheck, NULL);
    start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    for (int i = 0; i < ITERMAX; ++i)
    {
        rc =
            qmckl_get_mo_basis_mo_vgl_inplace_device(context, mo_vgl, size_max);
        qmckl_context_touch_device(context);
    }
    gettimeofday(&timecheck, NULL);
    end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    printf("Time for the calculation of 1 step (ms): %10.1f\n",
           (double)(end - start) / (double)ITERMAX);
    qmckl_free_device(context, elec_coord_device);
    qmckl_free_device(context, mo_vgl);
    rc = qmckl_context_destroy_device(context);
}