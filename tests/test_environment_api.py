
"""
tests/test_environment_api.py

Automated Test Suite for Real-Time Environmental Data Integration (OpenWeather API).
Student ID: IT23166660 - FullStrength.ai TEQS Component.

Covers:
1. Successful weather lookup with temperature, humidity, AQI, and PM2.5 parsing.
2. Verification of OpenWeather categorical AQI scale (1-5) and labels.
3. Missing API key returning HTTP 503.
4. Invalid/unknown city returning HTTP 404.
5. Network / API timeout returning HTTP 504.
6. Graceful handling of missing air pollution data (returns weather with None AQI).
7. In-memory caching with 10-minute TTL and cache indicator flags.
8. Preservation of indoor vs outdoor disclaimers and AQI scale compatibility notices.
"""

import unittest
from unittest.mock import patch, AsyncMock
import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.services.weather_service import weather_service, OPENWEATHER_AQI_LABELS
from app.core.config import settings


class MockHttpxResponse:
    """Mock helper for httpx responses."""

    def __init__(self, status_code: int, json_data: any, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text or str(json_data)

    def json(self):
        return self._json_data


class TestEnvironmentAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # Clear cache before each test
        weather_service.clear_cache()
        weather_service._api_key = "test_mock_api_key_12345"

    def tearDown(self):
        weather_service.clear_cache()
        weather_service._api_key = None

    @patch("httpx.AsyncClient.get", new_callable=AsyncMock)
    def test_successful_weather_and_aqi_lookup(self, mock_get):
        """Verify successful retrieval and parsing of weather and air pollution data."""

        async def side_effect(url, *args, **kwargs):
            url_str = str(url)
            if "geo/1.0/direct" in url_str:
                return MockHttpxResponse(200, [
                    {
                        "name": "Kurunegala",
                        "lat": 7.4863,
                        "lon": 80.3623,
                        "country": "LK",
                    }
                ])
            elif "data/2.5/weather" in url_str:
                return MockHttpxResponse(200, {
                    "name": "Kurunegala",
                    "main": {
                        "temp": 28.5,
                        "humidity": 78.0,
                    },
                    "weather": [
                        {"description": "scattered clouds"}
                    ],
                    "dt": 1728738900,
                })
            elif "data/2.5/air_pollution" in url_str:
                return MockHttpxResponse(200, {
                    "list": [
                        {
                            "main": {"aqi": 2},
                            "components": {"pm2_5": 14.8},
                        }
                    ]
                })
            return MockHttpxResponse(404, {})

        mock_get.side_effect = side_effect

        resp = self.client.get("/api/v1/environment/current?city=Kurunegala&country_code=LK")
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["location"]["city"], "Kurunegala")
        self.assertEqual(data["location"]["country"], "LK")
        self.assertEqual(data["location"]["latitude"], 7.4863)
        self.assertEqual(data["location"]["longitude"], 80.3623)
        self.assertEqual(data["temperature_c"], 28.5)
        self.assertEqual(data["humidity_pct"], 78.0)
        self.assertEqual(data["openweather_aqi_category"], 2)
        self.assertEqual(data["openweather_aqi_label"], "Fair")
        self.assertEqual(data["pm2_5_ug_m3"], 14.8)
        self.assertEqual(data["weather_description"], "scattered clouds")
        self.assertEqual(data["source"], "OPENWEATHER")
        self.assertEqual(data["environment_type"], "OUTDOOR_ESTIMATE")
        self.assertFalse(data["cached"])
        self.assertIn("OUTDOOR_ESTIMATE", data["environment_type"])
        self.assertIn("Outdoor weather estimates do not represent indoor gym ambient conditions", data["scientific_note"])
        self.assertIn("OpenWeather AQI uses a 1-5 categorical index", data["aqi_compatibility_note"])

    @patch("httpx.AsyncClient.get", new_callable=AsyncMock)
    def test_in_memory_caching(self, mock_get):
        """Verify that repeated queries within TTL are served from cache."""

        async def side_effect(url, *args, **kwargs):
            url_str = str(url)
            if "geo/1.0/direct" in url_str:
                return MockHttpxResponse(200, [{"name": "Colombo", "lat": 6.9271, "lon": 79.8612, "country": "LK"}])
            elif "data/2.5/weather" in url_str:
                return MockHttpxResponse(200, {"main": {"temp": 30.0, "humidity": 80.0}, "weather": [{"description": "clear sky"}]})
            elif "data/2.5/air_pollution" in url_str:
                return MockHttpxResponse(200, {"list": [{"main": {"aqi": 3}, "components": {"pm2_5": 25.0}}]})
            return MockHttpxResponse(404, {})

        mock_get.side_effect = side_effect

        # 1. First call: fresh request
        resp1 = self.client.get("/api/v1/environment/current?city=Colombo&country_code=LK")
        self.assertEqual(resp1.status_code, 200)
        self.assertFalse(resp1.json()["cached"])

        # 2. Second call: should be served from in-memory cache without extra HTTP calls
        call_count_before = mock_get.call_count
        resp2 = self.client.get("/api/v1/environment/current?city=Colombo&country_code=LK")
        self.assertEqual(resp2.status_code, 200)
        self.assertTrue(resp2.json()["cached"])
        self.assertIsNotNone(resp2.json()["cached_at_utc"])
        self.assertEqual(mock_get.call_count, call_count_before)

    def test_missing_api_key_returns_503(self):
        """Verify HTTP 503 is returned when API key is empty or unconfigured."""
        weather_service._api_key = ""
        with patch.object(settings, "OPENWEATHER_API_KEY", ""):
            resp = self.client.get("/api/v1/environment/current?city=Kandy")
            self.assertEqual(resp.status_code, 503)
            self.assertIn("not configured", resp.json()["detail"])

    @patch("httpx.AsyncClient.get", new_callable=AsyncMock)
    def test_invalid_city_returns_404(self, mock_get):
        """Verify HTTP 404 is returned when geocoding returns an empty city list."""
        mock_get.return_value = MockHttpxResponse(200, [])

        resp = self.client.get("/api/v1/environment/current?city=NonExistentCityXYZ")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("not found", resp.json()["detail"])

    @patch("httpx.AsyncClient.get", new_callable=AsyncMock)
    def test_api_timeout_returns_504(self, mock_get):
        """Verify HTTP 504 Gateway Timeout when OpenWeather times out."""
        mock_get.side_effect = httpx.TimeoutException("Connection timed out")

        resp = self.client.get("/api/v1/environment/current?city=Galle")
        self.assertEqual(resp.status_code, 504)
        self.assertIn("timed out", resp.json()["detail"])

    @patch("httpx.AsyncClient.get", new_callable=AsyncMock)
    def test_missing_air_pollution_handled_gracefully(self, mock_get):
        """Verify weather still succeeds even if air pollution endpoint fails (AQI fields become None)."""

        async def side_effect(url, *args, **kwargs):
            url_str = str(url)
            if "geo/1.0/direct" in url_str:
                return MockHttpxResponse(200, [{"name": "Jaffna", "lat": 9.6615, "lon": 80.0255, "country": "LK"}])
            elif "data/2.5/weather" in url_str:
                return MockHttpxResponse(200, {"main": {"temp": 31.0, "humidity": 70.0}, "weather": [{"description": "sunny"}]})
            elif "data/2.5/air_pollution" in url_str:
                # Simulate 404 or missing coordinates for air pollution
                return MockHttpxResponse(404, {})
            return MockHttpxResponse(404, {})

        mock_get.side_effect = side_effect

        resp = self.client.get("/api/v1/environment/current?city=Jaffna")
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["temperature_c"], 31.0)
        self.assertEqual(data["humidity_pct"], 70.0)
        self.assertIsNone(data["openweather_aqi_category"])
        self.assertIsNone(data["openweather_aqi_label"])
        self.assertIsNone(data["pm2_5_ug_m3"])

    def test_openweather_aqi_categories_mapping(self):
        """Verify the 1-5 categorical scale dictionary mapping."""
        self.assertEqual(OPENWEATHER_AQI_LABELS[1], "Good")
        self.assertEqual(OPENWEATHER_AQI_LABELS[2], "Fair")
        self.assertEqual(OPENWEATHER_AQI_LABELS[3], "Moderate")
        self.assertEqual(OPENWEATHER_AQI_LABELS[4], "Poor")
        self.assertEqual(OPENWEATHER_AQI_LABELS[5], "Very Poor")


if __name__ == "__main__":
    unittest.main()
