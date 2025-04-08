from hydra import initialize, compose
import logging

from runnables.train_multi import main


class TestCTCancerSimulation:

    def test_train_evaluation(self):
        logging.basicConfig(level='info')
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=gluroo_sim",
                                                                 "+backbone=ct",
                                                                 "exp.seed=100",
                                                                 "exp.logging=False",
                                                                 "dataset.num_patients.train=256",
                                                                 "dataset.num_patients.val=10",
                                                                 "dataset.num_patients.test=10"])
            results_1 = main(args)
            results_2 = main(args)
            print(results_1, results_2)
            assert results_1 == results_2
