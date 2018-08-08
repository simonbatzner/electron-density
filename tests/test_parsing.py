import unittest

import numpy as np
from Jon_Production.parse import load_config_yaml,Structure_Config

#TODO Flesh out the Structure Tester
class Test_Structure_Parser(unittest.TestCase):

    def setUp(self):
        self.struc1dict =load_config_yaml('test_configs/test_01.yaml')
        self.struc2dict =load_config_yaml('test_configs/test_02.yaml')

    def test_load_01(self):

        self.assertEqual(self.struc1dict['structure_params']['alat'],1.0)
        self.assertEqual(self.struc1dict['structure_params']['pos'],[['H',[0,0,0]]])

    def test_structure_01(self):

        H_config = Structure_Config(self.struc1dict['structure_params'])

        H_struct = H_config.to_structure()

        self.assertTrue(np.array_equal(H_config['lattice'],np.eye(3)))
        self.assertEqual(H_config['elements'],['H'])
        self.assertTrue(np.array_equal(H_config['unit_cell'],np.eye(3)))
        self.assertEqual(H_config['mass'],0)

        self.assertEqual(H_struct.atoms[0].mass,1.0)


    def test_structure_02(self):

        Si_config = Structure_Config(self.struc2dict['structure_params'])
        Si_struct = Structure_Config(self.struc2dict['structure_params']).to_structure()

        print(Si_struct)

        self.assertEqual(Si_struct[0].mass,28.0855)
        self.assertEqual(Si_struct.get_species_mass('Si'),28.0855)

        self.assertTrue(np.array_equal(Si_struct.get_positions(),[[0,0,0],[2,2,2]]))
        self.assertTrue(np.array_equal(Si_struct.forces,[[13,13,13],[10583,12583,14627]]))




    def runTest(self):

        self.test_load_01()
        self.test_structure_01()



#TODO MD Config Parser Test

#TODO QE Config Parser Test

#TODO HPC Config Parser Test

# TODO ML Config Parser Test


if __name__ == '__main__':
    unittest.main()
