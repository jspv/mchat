from datetime import datetime

import requests
from tzlocal import get_localzone


def today():
    local_timezone = get_localzone()

    return datetime.now(local_timezone).strftime("%Y-%m-%d %H:%M:%S %Z%z")


def get_location():
    # Use ipinfo.io to get the IP-based location
    response = requests.get("https://ipinfo.io")
    data = response.json()

    # Extracting details from the JSON response
    # city = data.get("city")
    # region = data.get("region")
    # country = data.get("country")
    # loc = data.get("loc", "0,0").split(",")
    return data

    # # Print the location details
    # print(f"City: {city}")
    # print(f"Region: {region}")
    # print(f"Country: {country}")
    # print(f"Latitude: {loc[0]}")
    # print(f"Longitude: {loc[1]}")
