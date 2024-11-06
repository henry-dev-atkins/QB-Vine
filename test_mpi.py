try:
    from mpi4py import MPI
except:
    raise ImportError("mpi4py not found. Please install it using 'pip', or 'Windows' as described in the README.md file")

comm = MPI.COMM_WORLD
size = comm.Get_size()

if size < 2:
    raise ValueError('A minimum of 2 ranks are required for the MPI backend')

print(f"Running with {size} MPI processes")