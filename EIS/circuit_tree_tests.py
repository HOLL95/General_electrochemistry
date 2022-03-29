import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from EIS_class import EIS
import time
test=EIS()
from circuit_drawer import circuit_artist
hard_tree={"root_series":{"left_paralell":{
                        "left_series":{"left_element":"R", "right_element":"C"},
                        "right_paralell":{"left_series":{"right_element":"R", "left_element":"R"}, "right_element":"R"}},"right_element":"C"},
                        }
triple_randles={"root_series":{"left_paralell":{"left_element":"R", "right_element":"R"},
"right_series":
{"left_paralell":{"left_element":"R", "right_element":"R"}, "right_series":
{"left_paralell":{"left_element":"R", "right_element":"R"}, "right_paralell":{"left_element":"R", "right_element":"R"}}}}}


z=test.construct_dict_from_tree(triple_randles["root_series"], "root_series")

#print(test.random_circuit_tree(3))
random_tree=test.random_circuit_tree(4)
#top_key=list(random_tree.keys())[0]
#random_dict=test.construct_dict_from_tree(random_tree[top_key], top_key)
circuit_artist(test.translate_tree(random_tree))
top_key=test.get_top_key(random_tree)
#print(random_tree)
#print(random_dict)
#print(test.construct_circuit_from_tree_dict(random_dict))
import random
get_node=test.find_random_node(random_tree[top_key], random.randint(1, 4), random_tree)
new_tree=test.random_circuit_tree(3)
crossed_tree=test.crossover(random_tree, new_tree)
circuit_artist(test.translate_tree(crossed_tree))

#print("\n")
#print(new_tree)
#print(replace_tree)
#print(get_node)
