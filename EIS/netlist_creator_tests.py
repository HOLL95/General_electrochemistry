import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
import matplotlib.pyplot as plt
import numpy as np
from EIS_class import EIS
from EIS_optimiser import EIS_genetics, EIS_optimiser
from circuit_drawer import circuit_artist
import math
init_test={"z1":{"p1":["R1", {"p1_1":"C1", "p1_2":["R9", {"p1":"C8", "p2":["R14", {"p1":"C-1", "p2":"C-2"}]}]}, {"p1_1":"C2", "p1_2":"C3"}], "p2":"R2"}, "z2":{"p1":"R3", "p2":"C4"}, "z3":"R5"}
eis=EIS(circuit=init_test, construct_netlist=True)
