#!/usr/bin/env python3
"""
Script that displays the first launch with these information
"""
import requests
from datetime import datetime


def get_first_launch():
    """
    Retrieves and displays information about the first SpaceX launch.
        Get all launches
        Sort by oldest first
        Get first launch
        Extract IDs
        Get rocket info
        Get launchpad info
        Convert UTC to local time 
    """
    # Step 1: Get all launches
    url = "https://api.spacexdata.com/v4/launches"
    launches = requests.get(url).json()

    # Step 2: Sort by oldest first
    launches.sort(key=lambda l: l.get("date_unix", float('inf')))

    # Step 3: Get first launch
    first = launches[0]

    # Step 4: Extract IDs
    rocket_id = first["rocket"]
    launchpad_id = first["launchpad"]

    # Step 5: Get rocket info
    rocket = requests.get(f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
                          ).json()
    rocket_name = rocket["name"]

    # Step 6: Get launchpad info
    launchpad = requests.get(
        f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
        ).json()
    launchpad_name = launchpad["name"]
    locality = launchpad["locality"]

    # Step 7: Convert UTC to local time (ISO format with offset)
    utc_time = datetime.fromisoformat(first["date_utc"].replace('Z', '+00:00'))
    local_time = utc_time.astimezone()  # local timezone automatically
    formatted_time = local_time.isoformat()

    # Step 8: Final output
    print(f"{first['name']} ({formatted_time}) {rocket_name} -\
          {launchpad_name}({locality})")


if __name__ == '__main__':
    get_first_launch()
