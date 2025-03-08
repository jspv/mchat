from typing import Annotated

from fredapi import Fred

from config import settings
from mchat.tool_utils import BaseTool


class FetchFREDDataTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="fetch_fred_data",
            description="Fetches data from the Federal Reserve Economic Data (FRED) using a series ID and date range.",
        )

    def verify_setup(self):
        api_key = settings.get("fred_api_key", None)
        if not api_key:
            raise ValueError("FRED API key not found.")
        self.fred = Fred(api_key=api_key)

    def run(
        self,
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
        if not self.is_callable:
            raise RuntimeError(
                f"Tool '{self.name}' is not callable due to setup failure: {self.load_error}"
            )

        data = self.fred.get_series(
            series, observation_start=start_date, observation_end=end_date
        )
        return data.to_json(orient="table")
