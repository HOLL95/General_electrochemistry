import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from EIS_class import EIS
class circuit_artist(EIS):
    def __init__(self, circuit_dictionary, **kwargs):
        if "ax" not in kwargs:
            fig,ax=plt.subplots()
        else:
            ax=kwargs["ax"]
        self.patch_library={}
        print(circuit_dictionary)
        element_width=0.5
        element_height=0.05
        circuit_height=0.25/2
        circuit_width=0.7/2
        start=1
        increment=1
        total_len=len(circuit_dictionary.keys())
        circuit_line=np.linspace(start-(circuit_width*2), start+(increment*total_len), 100)
        total_circuit=ax.plot(circuit_line, [0.5]*100, zorder=1)
        circuit_keys=list(circuit_dictionary.keys())
        distance_dict={}

        for i in range(0, total_len):
            current_key=circuit_keys[i]

            self.paralell_drawer(circuit_dictionary[current_key], level=1, target_y_pos=0.5, target_x_pos=start, height_sep=circuit_height, width_sep=circuit_width,element_width=element_width, element_height=element_height, ax=ax)
            start+=increment
        #compressed=super().dict_compress(distance_dict, parent_key="", sep="_", optional_escape="name", extract_lists=True)


        #ax.set_xlim(0, )
        ax.set_ylim(0, 1)
        naive_paralell={}
        #plt.show()

    def paralell_drawer(self, circuit_dict, level, target_y_pos, target_x_pos, height_sep, width_sep,element_width, element_height, ax):
        element=False
        if isinstance(circuit_dict, str):
            element=True
            name=circuit_dict
        elif isinstance(circuit_dict, tuple):
            element=True
            element_type=None
            for sub_element in circuit_dict:
                if  "alpha" in sub_element:
                    element_type="CPE"
                    element_number=sub_element[sub_element.index("p")+3:]
                    break
                elif "delta" in sub_element:
                    element_type="W_fin"
                    element_number=sub_element[sub_element.index("a")+2:]
                    break
            if element_type==None:
                raise ValueError("circuit"+ circuit_dict+ "element type not recognised")
            name=element_type+element_number

        if element==True:

            rect=patches.Rectangle(self.get_anchor_from_centre(target_x_pos,
                                                         target_y_pos,
                                                         element_width, element_height),
                                                         element_width, element_height, zorder=2)
            self.patch_library[name]=ax.add_patch(rect)
            ax.text(target_x_pos-element_width*0.5, target_y_pos+(element_height*0.6),name )
        if isinstance(circuit_dict, dict):
            rect=patches.Rectangle(self.get_anchor_from_centre(target_x_pos,
                                                            target_y_pos,
                                                            width_sep*2, height_sep*2),
                                                            width_sep*2, height_sep*2, facecolor=(1, 1, 1, 1), edgecolor="black", zorder=2)

            ax.add_patch(rect)
            if level>2:
                new_height=0.8*height_sep
                new_element_height=0.5*element_height
            else:
                new_height=0.5*height_sep
                new_element_height=element_height
            for key in ["p1", "p2"]:
                if key=="p1":
                    self.paralell_drawer(circuit_dict[key], level+1, target_y_pos+height_sep, target_x_pos, new_height, width_sep*0.8,element_width*0.8, new_element_height, ax=ax)
                elif key=="p2":
                    self.paralell_drawer(circuit_dict[key], level+1, target_y_pos-height_sep, target_x_pos, new_height, width_sep*0.8, element_width*0.8, new_element_height,ax=ax)
        if isinstance(circuit_dict, list):
            flat_list=super().list_flatten(circuit_dict)
            total_len=len(flat_list)
            fake_dict={}
            positions=np.linspace(target_x_pos-width_sep+(element_width/4), target_x_pos+width_sep-(element_width/4), total_len)

            for i in range(0, total_len):
                self.paralell_drawer(flat_list[i], level+1, target_y_pos, positions[i],height_sep, width_sep/total_len, element_width/total_len, element_height,ax=ax)


    def get_anchor_from_centre(self, centre_x,centre_y, width, height):
        anchor_x=centre_x-(width/2)
        anchor_y=centre_y-(height/2)
        return (anchor_x, anchor_y)
