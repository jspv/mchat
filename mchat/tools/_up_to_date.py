from datetime import datetime
from typing import Annotated, Any, Dict

import requests
from tzlocal import get_localzone

from mchat.tool_utils import BaseTool


class Today(BaseTool):
    name = "today"
    description = "Get the current date and time in the local timezone."

    def run(self) -> Annotated[str, "Current date and time in local timezone"]:
        """
        Get the current date and time in the local timezone.

        Returns:
            str: Current date and time formatted as "YYYY-MM-DD HH:MM:SS TZ+HHMM".
        """
        local_timezone = get_localzone()
        return datetime.now(local_timezone).strftime("%Y-%m-%d %H:%M:%S %Z%z")


class Location(BaseTool):
    name = "get_location"
    description = "Get IP-based geolocation data."

    def run(self) -> Annotated[Dict[str, Any], "IP-based geolocation data"]:
        """
        Get IP-based location data using ipinfo.io.

        Returns:
            dict: JSON response containing location information.
        """
        # Use ipinfo.io to get the IP-based location
        response = requests.get("https://ipinfo.io")
        data = response.json()

        return data

    # Extracting details from the JSON response
    # city = data.get("city")
    # region = data.get("region")
    # country = data.get("country")
    # loc = data.get("loc", "0,0").split(",")

    # # Print the location details
    # print(f"City: {city}")
    # print(f"Region: {region}")
    # print(f"Country: {country}")
    # print(f"Latitude: {loc[0]}")
    # print(f"Longitude: {loc[1]}")
