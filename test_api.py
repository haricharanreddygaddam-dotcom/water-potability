"""
Verification test suite for Water Potability Flask Application
"""
import unittest
import json
from app import app

class WaterPotabilityTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'AQUASCAN 3D', response.data)

    def test_presets(self):
        response = self.app.get('/api/presets')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('mountain_spring', data)
        self.assertIn('municipal_tap', data)
        self.assertIn('industrial_polluted', data)

    def test_model_info(self):
        response = self.app.get('/api/model-info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('metrics', data)
        self.assertIn('features', data)
        self.assertIn('accuracy', data['metrics'])
        self.assertGreater(data['metrics']['accuracy'], 0.75)

    def test_predict_potable(self):
        sample = {
            "ph": 7.4,
            "Hardness": 120.0,
            "Solids": 2400.0,
            "Chloramines": 1.2,
            "Sulfate": 185.0,
            "Conductivity": 280.0,
            "Organic_carbon": 3.1,
            "Trihalomethanes": 12.0,
            "Turbidity": 1.8
        }
        response = self.app.post('/api/predict', data=json.dumps(sample), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['is_potable'])
        self.assertGreater(data['potability_probability'], 60.0)
        self.assertIn('visual_params', data)
        self.assertIn('turbidity_factor', data['visual_params'])

    def test_predict_non_potable(self):
        sample = {
            "ph": 3.4,
            "Hardness": 310.0,
            "Solids": 49500.0,
            "Chloramines": 11.5,
            "Sulfate": 475.0,
            "Conductivity": 720.0,
            "Organic_carbon": 27.8,
            "Trihalomethanes": 115.0,
            "Turbidity": 6.8
        }
        response = self.app.post('/api/predict', data=json.dumps(sample), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertFalse(data['is_potable'])
        self.assertGreater(len(data['violations']), 0)
        self.assertIn('recommendations', data)

    def test_what_if_optimization(self):
        sample = {
            "ph": 4.0,
            "Hardness": 280.0,
            "Solids": 35000.0,
            "Chloramines": 9.0,
            "Sulfate": 420.0,
            "Conductivity": 600.0,
            "Organic_carbon": 22.0,
            "Trihalomethanes": 95.0,
            "Turbidity": 6.2
        }
        response = self.app.post('/api/what-if', data=json.dumps(sample), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('adjusted_parameters', data)
        self.assertTrue(data['projected_potable'])

if __name__ == '__main__':
    unittest.main()
