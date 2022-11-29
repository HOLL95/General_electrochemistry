import os
from sympy import *
import numpy as np
import pandas as pd
import sys
init_printing()
# initialize variables
class model_creator:
    def __init__(self, dt, end, source_func, param_dict):
        num_rlc = 0 # number of passive elements
        num_ind = 0 # number of inductors
        num_v = 0    # number of independent voltage sources
        num_i = 0    # number of independent current sources
        i_unk = 0  # number of current unknowns
        num_opamps = 0   # number of op amps
        num_vcvs = 0     # number of controlled sources of various types
        num_vccs = 0
        num_cccs = 0
        num_ccvs = 0
        num_cpld_ind = 0 # number of coupled inductors
        fn = 'temp_netlist'  #'RCL circuit test' #'RLC_series' #'RLC_parallel' # net list
        fd1 = open(fn+'.net','r')
        content = fd1.readlines()
        content = [x.strip() for x in content]  #remove leading and trailing white space
        # remove empty lines
        while '' in content:
            content.pop(content.index(''))

        # remove comment lines, these start with a asterisk *
        content = [n for n in content if not n.startswith('*')]
        # remove other comment lines, these start with a semicolon ;
        content = [n for n in content if not n.startswith(';')]
        # remove spice directives, these start with a period, .
        content = [n for n in content if not n.startswith('.')]
        # converts 1st letter to upper case
        #content = [x.upper() for x in content] <- this converts all to upper case
        content = [x.capitalize() for x in content]
        # removes extra spaces between entries
        content = [' '.join(x.split()) for x in content]

        self.line_cnt = len(content) # number of lines in the netlist
        branch_cnt = 0  # number of branches in the netlist
        # check number of entries on each line, count each element type
        num_R=0
        num_C=0
        for i in range(self.line_cnt):
            x = content[i][0]
            tk_cnt = len(content[i].split()) # split the line into a list of words

            if (x == 'R') or (x == 'L') or (x == 'C'):
                if tk_cnt != 4:
                    print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
                    print("had {:d} items and should only be 4".format(tk_cnt))
                num_rlc += 1
                branch_cnt += 1
                if x == 'L':
                    num_ind += 1
                if x == 'R':
                    num_R += 1
                if x == 'C':
                    num_C += 1
            elif x == 'V':
                if tk_cnt != 4:
                    print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
                    print("had {:d} items and should only be 4".format(tk_cnt))
                num_v += 1
                branch_cnt += 1
            elif x == 'I':
                if tk_cnt != 4:
                    print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
                    print("had {:d} items and should only be 4".format(tk_cnt))
                num_i += 1
                branch_cnt += 1
            elif x == 'O':
                if tk_cnt != 4:
                    print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
                    print("had {:d} items and should only be 4".format(tk_cnt))
                num_opamps += 1
            elif x == 'E':
                if (tk_cnt != 6):
                    print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
                    print("had {:d} items and should only be 6".format(tk_cnt))
                num_vcvs += 1
                branch_cnt += 1
            elif x == 'G':
                if (tk_cnt != 6):
                    print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
                    print("had {:d} items and should only be 6".format(tk_cnt))
                num_vccs += 1
                branch_cnt += 1
            elif x == 'F':
                if (tk_cnt != 5):
                    print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
                    print("had {:d} items and should only be 5".format(tk_cnt))
                num_cccs += 1
                branch_cnt += 1
            elif x == 'H':
                if (tk_cnt != 5):
                    print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
                    print("had {:d} items and should only be 5".format(tk_cnt))
                num_ccvs += 1
                branch_cnt += 1
            elif x == 'K':
                if (tk_cnt != 4):
                    print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
                    print("had {:d} items and should only be 4".format(tk_cnt))
                num_cpld_ind += 1
            else:
                print("unknown element type in branch {:d}, {:s}".format(i,content[i]))
        # build the pandas data frame
        self.df = pd.DataFrame(columns=['element','p node','n node','cp node','cn node',
            'Vout','value','Vname','Lname1','Lname2'])

        # this data frame is for branches with unknown currents
        self.df2 = pd.DataFrame(columns=['element','p node','n node'])
        # loads voltage or current sources into branch structure
        # load branch info into data frame
        for i in range(self.line_cnt):
            x = content[i][0]

            if (x == 'R') or (x == 'L') or (x == 'C'):
                self.rlc_element(i, content)
            elif (x == 'V') or (x == 'I'):
                self.indep_source(i, content)
            elif x == 'O':
                self.opamp_sub_network(i, content)
            elif x == 'E':
                self.vcvs_sub_network(i, content)
            elif x == 'G':
                self.vccs_sub_network(i, content)
            elif x == 'F':
                self.cccs_sub_network(i, content)
            elif x == 'H':
                self.ccvs_sub_network(i, content)
            elif x == 'K':
                self.cpld_ind_sub_network(i, content)
            else:
                print("unknown element type in branch {:d}, {:s}".format(i,content[i]))

        # count number of nodes
        num_nodes = self.count_nodes()

        # Build self.df2: consists of branches with current unknowns, used for C & D matrices
        # walk through data frame and find these parameters
        count = 0
        for i in range(len(self.df)):
            # process all the elements creating unknown currents
            x = self.df.loc[i,'element'][0]   #get 1st letter of element name
            if (x == 'L') or (x == 'V') or (x == 'O') or (x == 'E') or (x == 'H') or (x == 'F'):
                self.df2.loc[count,'element'] = self.df.loc[i,'element']
                self.df2.loc[count,'p node'] = self.df.loc[i,'p node']
                self.df2.loc[count,'n node'] = self.df.loc[i,'n node']
                count += 1
        # print a report
        print('Net list report')
        print('number of lines in netlist: {:d}'.format(self.line_cnt))
        print('number of branches: {:d}'.format(branch_cnt))
        print('number of nodes: {:d}'.format(num_nodes))
        # count the number of element types that affect the size of the B, C, D, E and J arrays
        # these are current unknows
        i_unk = num_v+num_opamps+num_vcvs+num_ccvs+num_cccs+num_ind
        print('number of unknown currents: {:d}'.format(i_unk))
        print('number of RLC (passive components): {:d}'.format(num_rlc))
        print('number of inductors: {:d}'.format(num_ind))
        print('number of independent voltage sources: {:d}'.format(num_v))
        print('number of independent current sources: {:d}'.format(num_i))
        print('number of op amps: {:d}'.format(num_opamps))
        print('number of E - VCVS: {:d}'.format(num_vcvs))
        print('number of G - VCCS: {:d}'.format(num_vccs))
        print('number of F - CCCS: {:d}'.format(num_cccs))
        print('number of H - CCVS: {:d}'.format(num_ccvs))
        print('number of K - Coupled inductors: {:d}'.format(num_cpld_ind))
        # initialize some symbolic matrix with zeros
        # A is formed by [[G, C] [B, D]]
        # Z = [I,E]
        # X = [V, J]
        V = zeros(num_nodes,1)
        I = zeros(num_nodes,1)
        G = zeros(num_nodes,num_nodes)  # also called Yr, the reduced nodal matrix
        s = 1 # the Laplace variable

        # count the number of element types that affect the size of the B, C, D, E and J arrays
        # these are element types that have unknown currents
        i_unk = num_v+num_opamps+num_vcvs+num_ccvs+num_ind+num_cccs
        # if i_unk == 0, just generate empty arrays
        B = zeros(num_nodes,i_unk)
        C = zeros(i_unk,num_nodes)
        D = zeros(i_unk,i_unk)
        Ev = zeros(i_unk,1)
        J = zeros(i_unk,1)
        # G matrix
        for i in range(len(self.df)):  # process each row in the data frame
            n1 = self.df.loc[i,'p node']
            n2 = self.df.loc[i,'n node']
            cn1 = self.df.loc[i,'cp node']
            cn2 = self.df.loc[i,'cn node']
            # process all the passive elements, save conductance to temp value
            x = self.df.loc[i,'element'][0]   #get 1st letter of element name
            if x == 'R':
                g = 1/sympify(self.df.loc[i,'element'])
            if x == 'C':
                g = s*sympify(self.df.loc[i,'element'])
            if x == 'G':   #vccs type element
                g = sympify(self.df.loc[i,'element'].lower())  # use a symbol for gain value

            if (x == 'R') or (x == 'C'):
                # If neither side of the element is connected to ground
                # then subtract it from the appropriate location in the matrix.
                if (n1 != 0) and (n2 != 0):
                    G[n1-1,n2-1] += -g
                    G[n2-1,n1-1] += -g

                # If node 1 is connected to ground, add element to diagonal of matrix
                if n1 != 0:
                    G[n1-1,n1-1] += g

                # same for for node 2
                if n2 != 0:
                    G[n2-1,n2-1] += g

            if x == 'G':    #vccs type element
                # check to see if any terminal is grounded
                # then stamp the matrix
                if n1 != 0 and cn1 != 0:
                    G[n1-1,cn1-1] += g

                if n2 != 0 and cn2 != 0:
                    G[n2-1,cn2-1] += g

                if n1 != 0 and cn2 != 0:
                    G[n1-1,cn2-1] -= g

                if n2 != 0 and cn1 != 0:
                    G[n2-1,cn1-1] -= g

        G  # display the G matrix
        # generate the B Matrix
        sn = 0   # count source number as code walks through the data frame
        for i in range(len(self.df)):
            n1 = self.df.loc[i,'p node']
            n2 = self.df.loc[i,'n node']
            n_vout = self.df.loc[i,'Vout'] # node connected to op amp output

            # process elements with input to B matrix
            x = self.df.loc[i,'element'][0]   #get 1st letter of element name
            if x == 'V':
                if i_unk > 1:  #is B greater than 1 by n?, V
                    if n1 != 0:
                        B[n1-1,sn] = 1
                    if n2 != 0:
                        B[n2-1,sn] = -1
                else:
                    if n1 != 0:
                        B[n1-1] = 1
                    if n2 != 0:
                        B[n2-1] = -1
                sn += 1   #increment source count
            if x == 'O':  # op amp type, output connection of the opamg goes in the B matrix
                B[n_vout-1,sn] = 1
                sn += 1   # increment source count
            if (x == 'H') or (x == 'F'):  # H: ccvs, F: cccs,
                if i_unk > 1:  #is B greater than 1 by n?, H, F
                    # check to see if any terminal is grounded
                    # then stamp the matrix
                    if n1 != 0:
                        B[n1-1,sn] = 1
                    if n2 != 0:
                        B[n2-1,sn] = -1
                else:
                    if n1 != 0:
                        B[n1-1] = 1
                    if n2 != 0:
                        B[n2-1] = -1
                sn += 1   #increment source count
            if x == 'E':   # vcvs type, only ik column is altered at n1 and n2
                if i_unk > 1:  #is B greater than 1 by n?, E
                    if n1 != 0:
                        B[n1-1,sn] = 1
                    if n2 != 0:
                        B[n2-1,sn] = -1
                else:
                    if n1 != 0:
                        B[n1-1] = 1
                    if n2 != 0:
                        B[n2-1] = -1
                sn += 1   #increment source count
            if x == 'L':
                if i_unk > 1:  #is B greater than 1 by n?, L
                    if n1 != 0:
                        B[n1-1,sn] = 1
                    if n2 != 0:
                        B[n2-1,sn] = -1
                else:
                    if n1 != 0:
                        B[n1-1] = 1
                    if n2 != 0:
                        B[n2-1] = -1
                sn += 1   #increment source count

        # check source count
        if sn != i_unk:
            print('source number, sn={:d} not equal to i_unk={:d} in matrix B'.format(sn,i_unk))

        B   # display the B matrix
        # find the the column position in the C and D matrix for controlled sources
        # needs to return the node numbers and branch number of controlling branch
        print('failed to find matching branch element in find_vname')
        # generate the C Matrix
        sn = 0   # count source number as code walks through the data frame
        for i in range(len(self.df)):
            n1 = self.df.loc[i,'p node']
            n2 = self.df.loc[i,'n node']
            cn1 = self.df.loc[i,'cp node'] # nodes for controlled sources
            cn2 = self.df.loc[i,'cn node']
            n_vout = self.df.loc[i,'Vout'] # node connected to op amp output

            # process elements with input to B matrix
            x = self.df.loc[i,'element'][0]   #get 1st letter of element name
            if x == 'V':
                if i_unk > 1:  #is B greater than 1 by n?, V
                    if n1 != 0:
                        C[sn,n1-1] = 1
                    if n2 != 0:
                        C[sn,n2-1] = -1
                else:
                    if n1 != 0:
                        C[n1-1] = 1
                    if n2 != 0:
                        C[n2-1] = -1
                sn += 1   #increment source count

            if x == 'O':  # op amp type, input connections of the opamp go into the C matrix
                # C[sn,n_vout-1] = 1
                if i_unk > 1:  #is B greater than 1 by n?, O
                    # check to see if any terminal is grounded
                    # then stamp the matrix
                    if n1 != 0:
                        C[sn,n1-1] = 1
                    if n2 != 0:
                        C[sn,n2-1] = -1
                else:
                    if n1 != 0:
                        C[n1-1] = 1
                    if n2 != 0:
                        C[n2-1] = -1
                sn += 1   # increment source count

            if x == 'F':  # need to count F (cccs) types
                sn += 1   #increment source count
            if x == 'H':  # H: ccvs
                if i_unk > 1:  #is B greater than 1 by n?, H
                    # check to see if any terminal is grounded
                    # then stamp the matrix
                    if n1 != 0:
                        C[sn,n1-1] = 1
                    if n2 != 0:
                        C[sn,n2-1] = -1
                else:
                    if n1 != 0:
                        C[n1-1] = 1
                    if n2 != 0:
                        C[n2-1] = -1
                sn += 1   #increment source count
            if x == 'E':   # vcvs type, ik column is altered at n1 and n2, cn1 & cn2 get value
                if i_unk > 1:  #is B greater than 1 by n?, E
                    if n1 != 0:
                        C[sn,n1-1] = 1
                    if n2 != 0:
                        C[sn,n2-1] = -1
                    # add entry for cp and cn of the controlling voltage
                    if cn1 != 0:
                        C[sn,cn1-1] = -sympify(self.df.loc[i,'element'].lower())
                    if cn2 != 0:
                        C[sn,cn2-1] = sympify(self.df.loc[i,'element'].lower())
                else:
                    if n1 != 0:
                        C[n1-1] = 1
                    if n2 != 0:
                        C[n2-1] = -1
                    vn1, vn2, df2_index = find_vname(self.df.loc[i,'Vname'])
                    if vn1 != 0:
                        C[vn1-1] = -sympify(self.df.loc[i,'element'].lower())
                    if vn2 != 0:
                        C[vn2-1] = sympify(self.df.loc[i,'element'].lower())
                sn += 1   #increment source count

            if x == 'L':
                if i_unk > 1:  #is B greater than 1 by n?, L
                    if n1 != 0:
                        C[sn,n1-1] = 1
                    if n2 != 0:
                        C[sn,n2-1] = -1
                else:
                    if n1 != 0:
                        C[n1-1] = 1
                    if n2 != 0:
                        C[n2-1] = -1
                sn += 1   #increment source count

        # check source count
        if sn != i_unk:
            print('source number, sn={:d} not equal to i_unk={:d} in matrix C'.format(sn,i_unk))

        C   # display the C matrix
        # generate the D Matrix
        sn = 0   # count source number as code walks through the data frame
        for i in range(len(self.df)):
            n1 = self.df.loc[i,'p node']
            n2 = self.df.loc[i,'n node']
            #cn1 = self.df.loc[i,'cp node'] # nodes for controlled sources
            #cn2 = self.df.loc[i,'cn node']
            #n_vout = self.df.loc[i,'Vout'] # node connected to op amp output

            # process elements with input to D matrix
            x = self.df.loc[i,'element'][0]   #get 1st letter of element name
            if (x == 'V') or (x == 'O') or (x == 'E'):  # need to count V, E & O types
                sn += 1   #increment source count

            if x == 'L':
                if i_unk > 1:  #is D greater than 1 by 1?
                    D[sn,sn] += -s*sympify(self.df.loc[i,'element'])
                else:
                    D[sn] += -s*sympify(self.df.loc[i,'element'])
                sn += 1   #increment source count

            if x == 'H':  # H: ccvs
                # if there is a H type, D is m by m
                # need to find the vn for Vname
                # then stamp the matrix
                vn1, vn2, df2_index = self.find_vname(self.df.loc[i,'Vname'])
                D[sn,df2_index] += -sympify(self.df.loc[i,'element'].lower())
                sn += 1   #increment source count

            if x == 'F':  # F: cccs
                # if there is a F type, D is m by m
                # need to find the vn for Vname
                # then stamp the matrix
                vn1, vn2, df2_index = self.find_vname(self.df.loc[i,'Vname'])
                D[sn,df2_index] += -sympify(self.df.loc[i,'element'].lower())
                D[sn,sn] = 1
                sn += 1   #increment source count

            if x == 'K':  # K: coupled inductors, KXX LYY LZZ value
                # if there is a K type, D is m by m
                vn1, vn2, ind1_index = self.find_vname(self.df.loc[i,'Lname1'])  # get i_unk position for Lx
                vn1, vn2, ind2_index = self.find_vname(self.df.loc[i,'Lname2'])  # get i_unk position for Ly
                # enter sM on diagonals = value*sqrt(LXX*LZZ)

                D[ind1_index,ind2_index] += -s*sympify('M{:s}'.format(self.df.loc[i,'element'].lower()[1:]))  # s*Mxx
                D[ind2_index,ind1_index] += -s*sympify('M{:s}'.format(self.df.loc[i,'element'].lower()[1:]))  # -s*Mxx

        # display the The D matrix
        D
        for i in range(num_nodes):
            V[i] = sympify('v{:d}'.format(i+1))

        V  # display the V matrix


        # The J matrix is an mx1 matrix, with one entry for each i_unk from a source
        #sn = 0   # count i_unk source number
        #oan = 0   #count op amp number
        for i in range(len(self.df2)):
            # process all the unknown currents
            J[i] = sympify('I_{:s}'.format(self.df2.loc[i,'element']))

        J  # diplay the J matrix
        # generate the I matrix, current sources have n2 = arrow end of the element
        for i in range(len(self.df)):
            n1 = self.df.loc[i,'p node']
            n2 = self.df.loc[i,'n node']
            # process all the passive elements, save conductance to temp value
            x = self.df.loc[i,'element'][0]   #get 1st letter of element name
            if x == 'I':
                g = sympify(self.df.loc[i,'element'])
                # sum the current into each node
                if n1 != 0:
                    I[n1-1] -= g
                if n2 != 0:
                    I[n2-1] += g

        I  # display the I matrix
        # generate the E matrix
        sn = 0   # count source number
        for i in range(len(self.df)):
            # process all the passive elements
            x = self.df.loc[i,'element'][0]   #get 1st letter of element name
            if x == 'V':
                Ev[sn] = sympify(self.df.loc[i,'element'])
                sn += 1

        Ev   # display the E matrix


        Z = I[:] + Ev[:]  # the + operator in python concatinates the lists
        Z  # display the Z matrix
        X = V[:] + J[:]  # the + operator in python concatinates the lists
        X  # display the X matrix
        n = num_nodes
        m = i_unk
        A = zeros(m+n,m+n)
        for i in range(n):
            for j in range(n):
                A[i,j] = G[i,j]

        if i_unk > 1:
            for i in range(n):
                for j in range(m):
                    A[i,n+j] = B[i,j]
                    A[n+j,i] = C[j,i]

            for i in range(m):
                for j in range(m):
                    A[n+i,n+j] = D[i,j]

        if i_unk == 1:
            for i in range(n):
                A[i,n] = B[i]
                A[n,i] = C[i]

        A  # display the A matrix

        pprint(A)
        pprint(X)
        pprint(Z)
        import re
        A_sz=shape(A)
        num_rows=A_sz[0]


        num_cols=A_sz[1]
        cpe_locs=[]
        for i in range(0, len(X)):
            if "I_H" in str(X[i]):
                cpe_locs.append(str(i))
            if "I_V1" in str(X[i]):
                current_idx=i
        deriv_array=np.zeros(len(X))
        deriv_mat=[["" for x in range(0, num_cols)] for y in range(0, num_rows)]
        for i in range(0, num_cols):
            for j in range(0, num_rows):

                str_entry=str(A[j,i])
                if "C" in str_entry:
                    #print(A[j,i], "sympy")
                    find_string=re.findall(r'-?C\d+?', str_entry)
                    deriv_mat[j][i]="+".join(find_string)
                    A[j,i]=sympify(str_entry+"-({0})".format(deriv_mat[j][i]))
                    deriv_array[j]=1

        print(deriv_mat)
        equation_list=[]
        for i in range(0, num_cols):
            z=[a*x for a,x in zip(A[i,:], X)]

            expression=sum(z)

            if Z[i]!=0:
                expression=expression-Z[i]
            str_exp=str(expression)


            print(str_exp)
            cpe_vals=re.findall(r'\*h\d+|h\d+\*', str_exp)
            str_variables=[str(x) for x in X]
            if len(cpe_vals)!=0:
                for j in range(0, len(cpe_vals)):
                    cpe_str=cpe_vals[j]
                    cpe_num_match=re.search(r"\d+", cpe_str)
                    cpe_num=int(cpe_num_match.group())
                    index=[x for x in range(0, len(str_variables)) if str_variables[x]=="I_H{0}".format(cpe_num)][0]
                    str_exp=re.sub(r'I_V1\*h{0}|h{0}\*I_V1'.format(cpe_num), "self.cpe_dict[\"{0}\"]".format(index), str_exp)
            if "V1" in str_exp:
                str_exp=re.sub("-V1", "self.source_func(t)", str_exp)
            for j in range(0, len(X)):
                var_name="x[{0}]".format(j)
                variable=str_variables[j]

                str_exp=re.sub(variable, var_name, str_exp)
                if deriv_mat[i][j]!='':
                    #if deriv_mat[i][j][0]!="-":
                    str_exp+="+("+deriv_mat[i][j]+")*xdot[{0}]".format(j)
                    #else:
                    #    str_exp+="-("+deriv_mat[i][j]+")*xdot[{0}]".format(j)
            str_exp="        result[{0}]=".format(i)+str_exp+"\n"
            equation_list.append(str_exp)
        deriv_list=[str(x) for x in np.where(deriv_array==1)[0]]
        deriv_locs="["+",".join(deriv_list)+"]"
        cpe_locs="["+",".join(cpe_locs)+"]"
        print(deriv_locs, cpe_locs)
        read_eq=open("dae_class.py", "r")
        temp_eq=open("temp_model.py", "w")
        num_cpe=num_ccvs
        source_func_count=-1
        equation_area_count=-1
        parameter_area_count=-1
        dt=dt
        t_end=end
        source_func=source_func
        print(source_func)
        params=param_dict
        for line in read_eq:
            if "self.num_cpe=0" in line:
                line="        self.num_cpe={0}\n".format(num_cpe)
            if "self.cpe_arrays={}" in line:
                if num_cpe>0:
                    line="        self.cpe_arrays={\"cpe_\"+str(x+1):np.zeros(len(self.t_array)) for x in range(0, self.num_cpe)}\n"
            if "self.cpe_potential_arrays={}" in line:
                if num_cpe>0:
                    line="        self.cpe_potential_arrays={\"cpe_\"+str(x+1):deque([0,0,0,0]) for x in range(0, self.num_cpe)}\n"
            if "def source(self, t)" in line:
                source_func_count=1
            if "result=[0,0,0,0]" in line:
                line="        result=["+",".join(["0" for x in X])+"]\n"
            if  "var_array=[0,0,0,0]" in line:
                line="        var_array=["+",".join(["0" for x in X])+"]\n"
            if equation_area_count==1:
                line=("").join(equation_list)+"\n"+line+"\n"
                equation_area_count=-1
            if "#equation_area" in line:
                equation_area_count=1
            if parameter_area_count==1:
                line=""#        R0=self.params[\"R0\"]\n"
                for i in range(0, num_R):
                    line+="        R{0}=self.params[\"R{0}\"]\n".format(i)
                for i in range(0, num_C):
                    line+="        C{0}=self.params[\"C{0}\"]\n".format(i+1)
                parameter_area_count=-1
            if "#parameter_area" in line:
                parameter_area_count=1
            if "dt=0" in line:
                line="dt="+str(dt)+"\n"

            if "sol_fs=solve_functions" in line:
                tabs="    "*9
                line=("    sol_fs=solve_functions(params="+params+",\n"
                                                +tabs+"derivative_vars="+deriv_locs+",\n"
                                                +tabs+"dt=dt,\n"+tabs+"end="+t_end+",\n"
                                                +tabs+"current_idx="+str(current_idx)+",\n"
                                                +tabs+"source_func=lambda t:"+source_func+",\n"
                                                +tabs+"cpe_locs="+cpe_locs+")\n")
            temp_eq.write(line)

        read_eq.close()
        temp_eq.close()
    def indep_source(self, line_nu, content):
        tk = content[line_nu].split()
        self.df.loc[line_nu,'element'] = tk[0]
        self.df.loc[line_nu,'p node'] = int(tk[1])
        self.df.loc[line_nu,'n node'] = int(tk[2])
        self.df.loc[line_nu,'value'] = float(tk[3])

    # loads passive elements into branch structure
    def rlc_element(self, line_nu, content):
        tk = content[line_nu].split()
        self.df.loc[line_nu,'element'] = tk[0]
        self.df.loc[line_nu,'p node'] = int(tk[1])
        self.df.loc[line_nu,'n node'] = int(tk[2])
        self.df.loc[line_nu,'value'] = float(tk[3])

    # loads multi-terminal elements into branch structure
    # O - Op Amps
    def opamp_sub_network(self, line_nu, content):
        tk = content[line_nu].split()
        self.df.loc[line_nu,'element'] = tk[0]
        self.df.loc[line_nu,'p node'] = int(tk[1])
        self.df.loc[line_nu,'n node'] = int(tk[2])
        self.df.loc[line_nu,'Vout'] = int(tk[3])

    # G - VCCS
    def vccs_sub_network(self, line_nu, content):
        tk = content[line_nu].split()
        self.df.loc[line_nu,'element'] = tk[0]
        self.df.loc[line_nu,'p node'] = int(tk[1])
        self.df.loc[line_nu,'n node'] = int(tk[2])
        self.df.loc[line_nu,'cp node'] = int(tk[3])
        self.df.loc[line_nu,'cn node'] = int(tk[4])
        self.df.loc[line_nu,'value'] = float(tk[5])

    # E - VCVS
    # in sympy E is the number 2.718, replacing E with Ea otherwise, sympify() errors out
    def vcvs_sub_network(self, line_nu, content):
        tk = content[line_nu].split()
        self.df.loc[line_nu,'element'] = tk[0].replace('E', 'Ea')
        self.df.loc[line_nu,'p node'] = int(tk[1])
        self.df.loc[line_nu,'n node'] = int(tk[2])
        self.df.loc[line_nu,'cp node'] = int(tk[3])
        self.df.loc[line_nu,'cn node'] = int(tk[4])
        self.df.loc[line_nu,'value'] = float(tk[5])

    # F - CCCS
    def cccs_sub_network(self, line_nu, content):
        tk = content[line_nu].split()
        self.df.loc[line_nu,'element'] = tk[0]
        self.df.loc[line_nu,'p node'] = int(tk[1])
        self.df.loc[line_nu,'n node'] = int(tk[2])
        self.df.loc[line_nu,'Vname'] = tk[3].capitalize()
        self.df.loc[line_nu,'value'] = float(tk[4])

    # H - CCVS
    def ccvs_sub_network(self, line_nu, content):
        #print(line_nu)
        tk = content[line_nu].split()
        self.df.loc[line_nu,'element'] = tk[0]
        self.df.loc[line_nu,'p node'] = int(tk[1])
        self.df.loc[line_nu,'n node'] = int(tk[2])
        self.df.loc[line_nu,'Vname'] = tk[3].capitalize()
        self.df.loc[line_nu,'value'] = float(tk[4])

    # K - Coupled inductors
    def cpld_ind_sub_network(self, line_nu, content):
        tk = content[line_nu].split()
        self.df.loc[line_nu,'element'] = tk[0]
        self.df.loc[line_nu,'Lname1'] = tk[1].capitalize()
        self.df.loc[line_nu,'Lname2'] = tk[2].capitalize()
        self.df.loc[line_nu,'value'] = float(tk[3])

    # function to scan self.df and get largest node number
    def count_nodes(self):
        # need to check that nodes are consecutive
        # fill array with node numbers
        p = np.zeros(self.line_cnt+1)
        for i in range(self.line_cnt):
            # need to skip coupled inductor 'K' statements
            if self.df.loc[i,'element'][0] != 'K': #get 1st letter of element name
                p[self.df['p node'][i]] = self.df['p node'][i]
                p[self.df['n node'][i]] = self.df['n node'][i]

        # find the largest node number
        if self.df['n node'].max() > self.df['p node'].max():
            largest = self.df['n node'].max()
        else:
            largest =  self.df['p node'].max()

        largest = int(largest)
        # check for unfilled elements, skip node 0
        for i in range(1,largest):
            if p[i] == 0:
                print('nodes not in continuous order, node {:.0f} is missing'.format(p[i-1]+1))

        return largest

    def find_vname(self, name):
        # need to walk through data frame and find these parameters
        for i in range(len(self.df2)):
            # process all the elements creating unknown currents
            if name == self.df2.loc[i,'element']:
                n1 = self.df2.loc[i,'p node']
                n2 = self.df2.loc[i,'n node']
                return n1, n2, i  # n1, n2 & col_num are from the branch of the controlling element



        #print()
