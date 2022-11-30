import numpy as np
import multiprocessing as mp
import ctypes as c
class mp_funcs:
    def __init__(self, sz):
        
        self.sz=sz
    def arr_init(self):
        sz=self.sz
        mp_arr = mp.Array(c.c_double, sz**2)
        arr = np.frombuffer(mp_arr.get_obj())
        globals()['arr']=arr.reshape((sz, sz))
        
    def simulate(self,list_arg):
        print(list_arg)
        globals()['arr'][list_arg[0],:]=list_arg[1]
    def async_sim(self):
        self.arr_init()
        list_arg=enumerate([np.arange(0, 4)*i for i in range(10, 50, 10)])
        with mp.Pool(processes=4) as P:
            P.map(self.simulate,list_arg)
z=mp_funcs(4)


z.async_sim()
print(globals()['arr'])
