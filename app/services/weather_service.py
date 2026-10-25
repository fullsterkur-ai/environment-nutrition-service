"""
app/services/weather_service.py

Asynchronous OpenWeather API service with in-memory caching and graceful error handling.
Student ID: IT23166660 - FullStrength.ai TEQS Component.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple
import logging
import httpx
from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.environment import LocationInfo, EnvironmentCurrentResponse

logger = logging.getLogger(__name__)

OPENWEATHER_AQI_LABELS = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor",
}


class WeatherService:
    """Service to fetch real-time atmospheric and air quality estimates from OpenWeather API."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._cache: Dict[Tuple[str, str], Tuple[datetime, EnvironmentCurrentResponse]] = {}

    @property
    def api_key(self) -> str:
        return self._api_key or settings.OPENWEATHER_API_KEY

    def is_api_key_configured(self) -> bool:
        key = self.api_key.strip()
        return bool(key) and key != "your_openweather_api_key_here"

    def clear_cache(self) -> None:
        """Clear the in-memory cache (useful for testing)."""
        self._cache.clear()

    async def get_current_environment(
        self, city: str, country_code: Optional[str] = None
    ) -> EnvironmentCurrentResponse:
        """
        Fetches current outdoor weather and air pollution estimates for a specified city.
        Caches responses in-memory for 10 minutes (TTL).
        """
        cleaned_city = city.strip()
        cleaned_country = country_code.strip().upper() if country_code else ""
        cache_key = (cleaned_city.lower(), cleaned_country.lower())

        # Check in-memory cache
        now = datetime.now(timezone.utc)
        if cache_key in self._cache:
            cached_time, cached_response = self._cache[cache_key]
            if now - cached_time < timedelta(seconds=settings.WEATHER_CACHE_TTL_SECONDS):
                # Return cached copy with cached flags updated
                cached_data = cached_response.model_dump()
                cached_data["cached"] = True
                cached_data["cached_at_utc"] = cached_time.isoformat()
                return EnvironmentCurrentResponse(**cached_data)

        # Validate API Key
        if not self.is_api_key_configured():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "OpenWeather API key is not configured. "
                    "Please set OPENWEATHER_API_KEY in your environment or .env file."
                ),
            )

        timeout = httpx.Timeout(settings.WEATHER_REQUEST_TIMEOUT_SECONDS)

        async with httpx.AsyncClient(timeout=timeout) as client:
            # 1. Geocoding: lookup coordinates
            geo_query = f"{cleaned_city},{cleaned_country}" if cleaned_country else cleaned_city
            geo_url = f"{settings.OPENWEATHER_BASE_URL}/geo/1.0/direct"
            geo_params = {"q": geo_query, "limit": 1, "appid": self.api_key}

            try:
                geo_resp = await client.get(geo_url, params=geo_params)
            except httpx.TimeoutException:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="OpenWeather geocoding request timed out.",
                )
            except httpx.RequestError as exc:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Network error contacting OpenWeather geocoding API: {str(exc)}",
                )

            if geo_resp.status_code == 401:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="OpenWeather authentication failed. Please verify OPENWEATHER_API_KEY.",
                )
            elif geo_resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"OpenWeather geocoding API returned HTTP {geo_resp.status_code}.",
                )

            geo_data = geo_resp.json()
            if not geo_data or not isinstance(geo_data, list):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"City '{cleaned_city}' not found in OpenWeather geocoding database.",
                )

            location_entry = geo_data[0]
            lat = location_entry["lat"]
            lon = location_entry["lon"]
            resolved_city = location_entry.get("name", cleaned_city)
            resolved_country = location_entry.get("country", cleaned_country or "Unknown")

            # 2. Weather Conditions
            weather_url = f"{settings.OPENWEATHER_BASE_URL}/data/2.5/weather"
            weather_params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric",
            }

            try:
                weather_resp = await client.get(weather_url, params=weather_params)
            except httpx.TimeoutException:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="OpenWeather weather request timed out.",
                )
            except httpx.RequestError as exc:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Network error contacting OpenWeather weather API: {str(exc)}",
                )

            if weather_resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"OpenWeather weather API returned HTTP {weather_resp.status_code}.",
                )

            weather_data = weather_resp.json()
            main_block = weather_data.get("main", {})
            temp_c = float(main_block.get("temp", 0.0))
            humidity_pct = float(main_block.get("humidity", 0.0))

            weather_desc_list = weather_data.get("weather", [])
            weather_desc = (
                weather_desc_list[0].get("description", "clear sky")
                if weather_desc_list
                else "clear sky"
            )

            # Observation timestamp
            dt_unix = weather_data.get("dt")
            if dt_unix:
                observed_dt = datetime.fromtimestamp(dt_unix, tz=timezone.utc)
            else:
                observed_dt = now
            observed_at_utc = observed_dt.isoformat().replace("+00:00", "Z")

            # 3. Air Pollution (optional / graceful fallback)
            aqi_category: Optional[int] = None
            aqi_label: Optional[str] = None
            pm2_5_val: Optional[float] = None

            pollution_url = f"{settings.OPENWEATHER_BASE_URL}/data/2.5/air_pollution"
            pollution_params = {"lat": lat, "lon": lon, "appid": self.api_key}

            try:
                pollution_resp = await client.get(pollution_url, params=pollution_params)
                if pollution_resp.status_code == 200:
                    pollution_data = pollution_resp.json()
                    p_list = pollution_data.get("list", [])
                    if p_list:
                        first_record = p_list[0]
                        aqi_val = first_record.get("main", {}).get("aqi")
                        if isinstance(aqi_val, int) and 1 <= aqi_val <= 5:
                            aqi_category = aqi_val
                            aqi_label = OPENWEATHER_AQI_LABELS.get(aqi_val, "Unknown")

                        components = first_record.get("components", {})
                        if "pm2_5" in components and components["pm2_5"] is not None:
                            pm2_5_val = float(components["pm2_5"])
            except Exception as e:
                # Log but do not fail the request; air pollution may be unavailable for some coordinates
                logger.warning(f"Could not retrieve air pollution data for ({lat}, {lon}): {e}")

        # Assemble result
        location_info = LocationInfo(
            city=resolved_city,
            country=resolved_country,
            latitude=round(lat, 4),
            longitude=round(lon, 4),
        )

        fresh_response = EnvironmentCurrentResponse(
            location=location_info,
            temperature_c=round(temp_c, 1),
            humidity_pct=round(humidity_pct, 1),
            openweather_aqi_category=aqi_category,
            openweather_aqi_label=aqi_label,
            pm2_5_ug_m3=round(pm2_5_val, 1) if pm2_5_val is not None else None,
            weather_description=weather_desc,
            observed_at_utc=observed_at_utc,
            source="OPENWEATHER",
            environment_type="OUTDOOR_ESTIMATE",
            cached=False,
            cached_at_utc=None,
        )

        # Store in cache
        self._cache[cache_key] = (now, fresh_response)

        return fresh_response


weather_service = WeatherService()
