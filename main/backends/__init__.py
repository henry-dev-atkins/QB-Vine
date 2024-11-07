import logging
import time
from .base import *
import main.backends.mpimanager as mpimanager
from main.backends.mpi import BackendMPILeader, BackendMPIWorker


class BackendMPITeam(BackendMPILeader if mpimanager.get_mpi_manager().is_leader() else BackendMPIWorker):
    """
    A team is compounded by workers and a leader. One process per team is a leader, others are workers
    """

    OP_PARALLELIZE, OP_MAP, OP_COLLECT, OP_BROADCAST, OP_DELETEPDS, OP_DELETEBDS, OP_FINISH = [1, 2, 3, 4, 5, 6, 7]

    def __init__(self):
        # Define the vars that will hold the pds ids received from scheduler to operate on
        self._rec_pds_id = None
        self._rec_pds_id_result = None

        # Initialize a BDS store for both scheduler & team.
        self.bds_store = {}

        # print("In BackendMPITeam, rank : ", self.rank, ", model_rank_global : ", globals()['model_rank_global'])

        self.logger = logging.getLogger(__name__)
        super().__init__()


class BackendMPITeam(BackendMPILeader if mpimanager.get_mpi_manager().is_leader() else BackendMPIWorker):
    """"A backend parallelized by using MPI

    The backend conditionally inherits either the BackendMPIScheduler class
    or the BackendMPIteam class depending on it's rank. This lets
    BackendMPI have a uniform interface for the user but allows for a
    logical split between functions performed by the scheduler
    and the teams.
    """

    def __init__(self, scheduler_node_ranks=[0], process_per_model=1):
        """
        Parameters
        ----------
        scheduler_node_ranks: Python list
            list of scheduler nodes

        process_per_model: Integer
            number of MPI processes to allocate to each model
        """
        # get mpimanager instance from the mpimanager module (which has to be setup before calling the constructor)
        self.logger = logging.getLogger(__name__)
        self.mpimanager = mpimanager.get_mpi_manager()

        if self.mpimanager.get_world_size() < 2:
            raise ValueError('A minimum of 2 ranks are required for the MPI backend')

        # Set the global backend
        globals()['backend'] = self

        # Call the appropriate constructors and pass the required data
        super().__init__()

    def size(self):
        """ Returns world size """
        return self.mpimanager.get_world_size()

    def scheduler_node_ranks(self):
        """ Returns scheduler node ranks """
        return self.mpimanager.get_scheduler_node_ranks()

    @staticmethod
    def disable_nested(mpi_comm):
        if mpi_comm.Get_rank() != 0:
            mpi_comm.Barrier()

    @staticmethod
    def enable_nested(mpi_comm):
        if mpi_comm.Get_rank() == 0:
            mpi_comm.Barrier()


class PDSMPI(PDS):
    """
    This is an MPI wrapper for a Python parallel data set.
    """

    def __init__(self, python_list, pds_id, backend_obj):
        self.python_list = python_list
        self.pds_id = pds_id
        self.backend_obj = backend_obj

    def __del__(self):
        """
        Destructor to be called when a PDS falls out of scope and/or is being deleted.
        Uses the backend to send a message to destroy the teams' copy of the pds.
        """
        try:
            self.backend_obj.delete_remote_pds(self.pds_id)
        except AttributeError:
            # Catch "delete_remote_pds not defined" for teams and ignore.
            pass


class BDSMPI(BDS):
    """
    This is a wrapper for MPI's BDS class.
    """

    def __init__(self, object, bds_id, backend_obj):
        # The BDS data is no longer saved in the BDS object.
        # It will access & store the data only from the current backend
        self.bds_id = bds_id
        backend.bds_store[self.bds_id] = object

    def value(self):
        """
        This method returns the actual object that the broadcast data set represents.
        """
        return backend.bds_store[self.bds_id]

    def __del__(self):
        """
        Destructor to be called when a BDS falls out of scope and/or is being deleted.
        Uses the backend to send a message to destroy the teams' copy of the bds.
        """

        try:
            backend.delete_remote_bds(self.bds_id)
        except AttributeError:
            # Catch "delete_remote_pds not defined" for teams and ignore.
            pass


class BackendMPITestHelper:
    """
    Helper function for some of the test cases to be able to access and verify class members.
    """

    def check_pds(self, k):
        """Checks if a PDS exists in the pds data store. Used to verify deletion and creation
        """
        return k in backend.pds_store.keys()

    def check_bds(self, k):
        """Checks if a BDS exists in the BDS data store. Used to verify deletion and creation
        """
        return k in backend.bds_store.keys()