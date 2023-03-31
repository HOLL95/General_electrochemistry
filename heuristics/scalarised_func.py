def simulate(self, parameters, times, lazy_evaluation=False):
        if self.global_options["label"]=="cmaes":
            parameters=self.change_norm_group(parameters, "un_norm")
            #print(list(parameters))
        discriminator_params=[parameters[x] for x in self.discriminator_idx]
        discriminator_data=self.discriminator_class.simulate(discriminator_params, self.x_values[self.discriminator_key])
        if lazy_evaluation==True:
            if self.MAP is None:
                _,self.MAP, self.MAP_noise=self.get_MAP(self.discriminator_noise)
            continuation=False
            if self.pints_problem is None:
                raise ValueError("Need to specify pints.problem")
            else:
                discriminator_value=self.pints_problem(np.append(discriminator_params, self.MAP_noise))

                ratio=self.MAP/discriminator_value #if the fit is bad->0 if the fit is good ->1
                print(self.MAP, discriminator_value, ratio)
                u_rand=np.random.rand()
                if u_rand<ratio:
                    continuation=True
        if lazy_evaluation==False or continuation==True:
            data=[discriminator_data]
            for i in range(0, len(self.slow_keys)):
                key=self.slow_keys[i]
                slow_class=self.class_dict[key]
                slow_params=[parameters[self.other_idx[i][j]] for j in range(0, len(self.other_idx[i]))]
                slow_x_val=self.x_values[key]
                data.append(slow_class.simulate(slow_params, slow_x_val))
            #WEIGHTS GO HERE

            final_data=self.data_combiner(data)
            if self.global_options["test"]==True:
                print(self.MAP)
                print(self.common_optim_list)
                print(list(parameters))
                plt.plot(self.total_likelihood)
                plt.plot(final_data)
                plt.show()
        else:
            final_data=np.ones(len(times))*1e10
        return final_data