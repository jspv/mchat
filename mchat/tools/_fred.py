from typing import Annotated

from fredapi import Fred

from config import settings


def fetch_fred_data(
    series: Annotated[str, "FRED series ID"],
    start_date: Annotated[str, "Start date in 'YYYY-MM-DD' format"],
    end_date: Annotated[str, "End date in 'YYYY-MM-DD' format"],
) -> dict:
    """
    Fetch data from the Federal Reserve Economic Data (FRED) API.

    Args:
        series (str): FRED series ID.
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.

    Returns:
        dict: FRED data as a JSON-serializable dictionary.
    """
    # Fetch FRED API key
    api_key = settings.get("fred_api_key", None)

    if not api_key:
        raise ValueError("FRED API key not found")

    # Initialize the FRED API client
    fred = Fred(api_key=api_key)

    # Fetch the FRED series data
    data = fred.get_series(
        series, observation_start=start_date, observation_end=end_date
    )

    # Convert DataFrame to JSON
    data_json = data.to_json(orient="table")

    return data_json
