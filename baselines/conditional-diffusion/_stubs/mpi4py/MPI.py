"""Single-process stand-in for mpi4py (world size 1)."""
class _Comm:
    rank = 0
    size = 1
    def Get_rank(self): return 0
    def Get_size(self): return 1
    def bcast(self, obj=None, root=0): return obj
COMM_WORLD = _Comm()
