def generate_distance_dict(self, circuit_dict, level, target_y_pos, target_x_pos, parent, parent_key, height_sep, width_sep):
        if isinstance(circuit_dict, str):
             return {"name":circuit_dict, "y_pos":target_y_pos, "x_pos":target_x_pos}
        elif isinstance(circuit_dict, tuple):
            element_type=None
            for element in circuit_dict:
                if  "alpha" in element:
                    element_type="CPE"
                    element_number=element[element.index("p")+3:]
                    break
                elif "delta" in element:
                    element_type="W_fin"
                    element_number=element[element.index("a")+2:]
                    break
            if element_type==None:
                raise ValueError("circuit"+ circuit_dict+ "element type not recognised")
            return {"name":element_type+element_number, "y_pos":target_y_pos, "x_pos":target_x_pos}
        elif isinstance(circuit_dict, dict):
            #initially only for circuit dicts generated via the tree method
            if level>2:
                new_height=0.8*height_sep
            else:
                new_height=0.5*height_sep
            return super().paralell_combinator(self.generate_distance_dict(circuit_dict["p1"], level+1, target_y_pos+height_sep, target_x_pos, circuit_dict, "p1", new_height, width_sep),
                                self.generate_distance_dict(circuit_dict["p2"], level+1, target_y_pos-height_sep, target_x_pos, circuit_dict, "p2", new_height, width_sep))
        elif isinstance(circuit_dict, list):
            flat_list=super().list_flatten(circuit_dict)
            total_len=len(flat_list)
            fake_dict={}
            positions=np.linspace(target_x_pos-width_sep+(self.element_width), target_x_pos+width_sep-(self.element_width), total_len)
            print(positions, "poslist2", target_x_pos)
            for i in range(0, total_len):
                fake_key="f{0}".format(i)
                fake_dict[fake_key]=self.generate_distance_dict(flat_list[i], level+1, target_y_pos, positions[i], fake_dict, fake_key, height_sep, width_sep)
            return [fake_dict["f{0}".format(i)] for i in range(0, total_len)]
