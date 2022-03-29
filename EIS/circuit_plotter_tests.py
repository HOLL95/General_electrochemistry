import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from EIS_class import EIS
import time
from circuit_drawer import circuit_artist
test_dict={'z1': {'p1': 'R1', 'p2': 'R2'}, 'z2': {'p1': {'p1': 'W1', 'p2': ('Q1', 'alpha1')}, 'p2': ["R4", {'p1': 'R3', 'p2': 'W2'}]}, 'z0': 'R0'}
"""distances_dict={"z1":{"x_pos":1,"y_pos":0.5,
                        "p1":
                            {"name":"R1", "y_pos":0.5+0.25, "x_pos":1},
                        "p2":
                            {"name":"R2", "y_pos":0.5-0.25, "x_pos":1}},
                'z2':{"x_pos": 2,"y_pos":0.5,
                        "p1":
                            [{"name":"R4", "y_pos":0.5+0.25, "x_pos":2-0.25}, {'p1':
                                {"name":"W1", "y_pos":0.5+0.25+(0.25/2), "x_pos":2+0.25},
                            'p2':
                                {"name":"CPE1", "y_pos":0.5+0.25-(0.25/2), "x_pos":2+0.25},
                            }],
                        'p2':
                            {'p1':
                             {"name":"R3", "y_pos":0.5-0.25+(0.25/2), "x_pos":2},
                             'p2':
                             {"name":"W2", "y_pos":0.5-0.25-(0.25/2), "x_pos":2}
                             }
                     }
                 }"""
test=circuit_artist(test_dict)
