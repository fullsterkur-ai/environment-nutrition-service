"""
app/api/v1/endpoints/environment.py

API route for real-time environmental data integration via OpenWeather API.
Student ID: IT23166660
"""

from typing import Optional
from fastapi import APIRouter, Query, status

from app.schemas.environment import EnvironmentCurrentResponse
from app.services.weather_service import weather_service

router = APIRouter(prefix="/environment", tags=["Environmental Data"])


@router.get(
    "/current",
    response_model=EnvironmentCurrentResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch current outdoor environmental data via OpenWeather",
    description=(
        "Retrieves real-time outdoor temperature, humidity, weather description, and "
        "OpenWeather air pollution estimates (categorical AQI 1-5 and PM2.5 in ug/m3) for a given city. "
        "Data is cached in-memory for 10 minutes. "
        "NOTE: Outdoor atmospheric data represents an OUTDOOR_ESTIMATE, not a direct indoor gym measurement. "
        "OpenWeather AQI (1-5 scale) is distinct from the TEQS baseline continuous scale (0-500) and is kept separate."
    ),
)
async def get_current_environment(
    city: str = Query(
        ...,
        min_length=1,
        description="Target city name (e.g. 'Kurunegala', 'Colombo', 'London')",
        examples=["Kurunegala"],
    ),
    country_code: Optional[str] = Query(
        None,
        min_length=2,
        max_length=2,
        description="Optional ISO 3166-1 alpha-2 country code (e.g. 'LK', 'GB', 'US')",
        examples=["LK"],
    ),
) -> EnvironmentCurrentResponse:
    """Asynchronously fetches current weather and air pollution from OpenWeather."""
    return await weather_service.get_current_environment(
        city=city, country_code=country_code
    )
