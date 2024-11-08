from .base import *
from .mpi import BackendMPITeam


def BackendMPI(*args, **kwargs):
    # import and setup module mpimanager
    from . import mpimanager
    master_node_ranks = [0]
    process_per_model = 1
    if 'master_node_ranks' in kwargs:
        master_node_ranks = kwargs['master_node_ranks']
    if 'process_per_model' in kwargs:
        process_per_model = kwargs['process_per_model']
    mpimanager.create_mpi_manager(master_node_ranks, process_per_model)

    # import BackendMPI and return an instance
    from .mpi import BackendMPI
    return BackendMPI(*args, **kwargs)


def BackendMPITestHelper(*args, **kwargs):
    from .mpi import BackendMPITestHelper
    return BackendMPITestHelper(*args, **kwargs)


def BackendSpark(*args, **kwargs):
    from .spark import BackendSpark
    return BackendSpark(*args, **kwargs)