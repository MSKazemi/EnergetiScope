
import unittest
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from app.train_power import train_model

class TestTrainPower(unittest.TestCase):

    def test_train_model(self):
        # Create a dummy dataframe for testing
        data = {
            'features': [np.random.rand(10) for _ in range(10)],
            'total_energy_j': np.random.rand(10),
            'workload_name': ['workload' + str(i) for i in range(10)]
        }
        df = pd.DataFrame(data)

        # Train the model
        model = train_model(df, 'total_energy_j', 5)

        # Check if the model is a KNeighborsRegressor
        self.assertIsInstance(model, KNeighborsRegressor)

if __name__ == '__main__':
    unittest.main()
