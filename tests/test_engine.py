import unittest
from Jon_Production.oo_MD_Engine import MD_Engine
from Jon_Production.parse import load_config_yaml,QE_Config,Structure_Config,ml_config,MD_Config

class Test_Structure_Parser(unittest.TestCase):

    def setUp(self):
        config = load_config_yaml('test_configs/test_03.yaml')
        qe_config = QE_Config(config['qe_params'], warn=True)
        structure = Structure_Config(config['structure_params']).to_structure()
        ml_config_ = ml_config(params=config['ml_params'], print_warn=True)
        md_config = MD_Config(params=config['md_params'], warn=True)
        self.engine = MD_Engine(structure, md_config, qe_config, ml_config_)



    def runTest(self):

        pass #Easy A
        #self.engine.run()



if __name__ == '__main__':
    unittest.main()
