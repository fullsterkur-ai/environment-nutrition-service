"""
app/schemas/environment.py

Pydantic schemas for real-time environmental data integration via OpenWeather API.
Student ID: IT23166660
"""

from typing import Optional
from pydantic import BaseModel, Field


class LocationInfo(BaseModel):
    city: str = Field(..., description="Matched city name", examples=["Kurunegala"])
    country: str = Field(..., description="ISO country code or country name", examples=["LK"])
    latitude: float = Field(..., description="Geographical latitude coordinate", examples=[7.4863])
    longitude: float = Field(..., description="Geographical longitude coordinate", examples=[80.3623])


class EnvironmentCurrentResponse(BaseModel):
    location: LocationInfo = Field(..., description="Geographic location metadata")
    temperature_c: float = Field(
        ...,
        description="Outdoor ambient temperature in Celsius",
        examples=[28.5],
    )
    humidity_pct: float = Field(
        ...,
        description="Outdoor relative humidity percentage",
        examples=[78.0],
    )
    openweather_aqi_category: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="OpenWeather categorical air quality index (1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor)",
        examples=[2],
    )
    openweather_aqi_label: Optional[str] = Field(
        None,
        description="Descriptive label for OpenWeather AQI category ('Good', 'Fair', 'Moderate', 'Poor', 'Very Poor')",
        examples=["Fair"],
    )
    pm2_5_ug_m3: Optional[float] = Field(
        None,
        ge=0.0,
        description="Fine particulate matter (PM2.5) concentration in micrograms per cubic meter (ug/m3)",
        examples=[14.8],
    )
    weather_description: str = Field(
        ...,
        description="Human-readable summary of weather conditions",
        examples=["scattered clouds"],
    )
    observed_at_utc: str = Field(
        ...,
        description="Observation timestamp formatted as ISO 8601 UTC string",
        examples=["2026-10-12T12:45:00Z"],
    )
    source: str = Field(
        default="OPENWEATHER",
        description="Data origin attribution",
    )
    environment_type: str = Field(
        default="OUTDOOR_ESTIMATE",
        description="Environmental measurement classification (outdoor atmospheric estimate, NOT direct indoor sensor)",
    )
    cached: bool = Field(
        default=False,
        description="True if the response was retrieved from in-memory cache",
    )
    cached_at_utc: Optional[str] = Field(
        None,
        description="Timestamp when the cached record was saved",
    )
    scientific_note: str = Field(
        default=(
            "Outdoor weather estimates do not represent indoor gym ambient conditions. "
            "In air-conditioned or enclosed facilities, indoor temperature and humidity "
            "may significantly diverge from outdoor atmospheric data."
        ),
        description="Research caveat on indoor gym vs outdoor atmospheric discrepancies",
    )
    aqi_compatibility_note: str = Field(
        default=(
            "OpenWeather AQI uses a 1-5 categorical index. It must NOT be passed directly "
            "into the exploratory TEQS baseline (which expects continuous US EPA AQI 0-500). "
            "In a future calibration stage, continuous AQI should be derived from raw PM2.5/PM10 "
            "concentrations using official EPA/WHO piecewise linear breakpoint formulas."
        ),
        description="AQI scale incompatibility warning and recommended calibration strategy",
    )
