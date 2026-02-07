import argparse
import os
import random
import string
import time
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm

from utils import int_list_to_str, mean, run_with_timeout, std

TIMEOUT_DURATION = 6


@dataclass
class Submission:
    """Class containing submission details."""
    title: str  # Title.
    sub_id: str  # Paper ID.
    ratings: list[int]  # List of reviewer ratings.
    confidences: list[int]  # List of reviewer confidences.
    final_ratings: list[int]  # List of final reviewer ratings.

    def __repr__(self) -> str:
        return f"Submission({self.sub_id}, {self.title}, {self.ratings}, {self.confidences})"

    def __str__(self) -> str:
        return f"{self.sub_id}, {self.title}, *, {int_list_to_str(self.ratings)}, *, {int_list_to_str(self.final_ratings)}"

    def info(self) -> str:
        return f"ID: {self.sub_id}, {self.title}, " + \
            f"Ratings: {self.ratings}, " + \
            f"Avg: {np.mean(self.ratings):.2f}, " + \
            f"Var: {np.var(self.ratings):.2f}"


class ORAPI:
    conf_to_url = {
        "cvpr_2026": "https://openreview.net/group?id=thecvf.com/CVPR/2026/Conference/Area_Chairs",
        # Add new ones here.
    }

    def __init__(self, conf: str, headless: bool = True):
        """Initializes the OpenReviewAPI.

        Args:
        conf (str): Name of the conference.
        headless (bool): Run without opening a browser window if True.
        skip_reviews (bool): Don't look for paper reviews if True.
        """
        # Create webdriver.
        options = webdriver.FirefoxOptions()
        if headless:
            options.add_argument("--headless")
        self.driver = webdriver.Firefox(options=options)

        self._login(self._get_url(conf))
        self.conf = conf

    def __del__(self):
        self.driver.quit()

    def _get_url(self, conf: str) -> str:
        """Get conference Area Chair URL."""
        if conf in self.conf_to_url:
            return self.conf_to_url[conf]
        else:
            raise ValueError(f"Conf: {conf} not supported.")

    def _login(self, url: str) -> None:
        # Load username and password.
        load_dotenv()
        username = os.environ["USERNAME"]
        password = os.environ["PASSWORD"]

        # Log in and navigate to url.
        print(f"Opening {url}")
        self.driver.get(url)
        time.sleep(1)
        self.driver.find_element(By.ID, "email-input").send_keys(username)
        self.driver.find_element(By.ID, "password-input").send_keys(password)
        self.driver.find_element(By.CLASS_NAME, "btn-login").click()
        print("Logging in.")

        # Wait for page to load, get urls to all papers.
        print("Waiting for page to finish loading...")

        def load_landing_page(driver):
            while True:
                urls = driver.find_elements(By.XPATH, "//div[@class='note']/h4/a")
                urls = [url.get_attribute("href") for url in urls]
                if urls:
                    break
            print("Logged in.")
            print(f"Found {len(urls)} submissions.")
            return urls

        urls = run_with_timeout(
            load_landing_page,
            (self.driver,),
            timeout_duration=TIMEOUT_DURATION,
            default_output=[]
        )
        self.paper_urls = urls

    def _parse_rating(self, reviews):
        """Parse ratings from reviews, based on the conference."""
        ratings, final_ratings, confidences = [], [], []

        for reply in reviews:
            content = reply.text
            rating, final_rating, confidence = None, None, None

            if "iclr" in self.conf:
                rating_start = content.find("Rating: ")
                if rating_start > 0:
                    confidence_start = content.find("Confidence: ")
                    code_start = content.find("Code Of Conduct: ")
                    rating = int(content[rating_start:confidence_start].split(":")[1].strip())
                    confidence = int(content[confidence_start:code_start].split(":")[1].strip())

            elif "cvpr" in self.conf:
                rating_start = content.find("Preliminary Recommendation: ")
                if rating_start > 0:
                    just_start = content.find("Justification For Recommendation And Suggestions For Rebuttal: ")
                    conf_start = content.find("Confidence Level: ")
                    rating = int(content[rating_start:just_start].split(":")[1].strip())
                    confidence = int(content[conf_start:].split(":")[1].strip())
                    final_rating_start = content.find("Final Recommendation:")
                    final_rating_end = content.find("Paper Strengths:")
                    if final_rating_start > 0:
                        final_rating = int(content[final_rating_start:final_rating_end].split(":")[1].strip())

            if rating is not None and confidence is not None:
                ratings.append(rating)
                confidences.append(confidence)
                if final_rating is not None:
                    final_ratings.append(final_rating)

        return ratings, confidences, final_ratings

    def load_submission(self, url: str, skip_reviews: bool = False) -> Submission:
        """Navigate to submission link and parse info.

        Args:
        driver: Instance of Firefox WebDriver.
        url (str): URL to submission.
        skip_reviews (bool): Skip looking for reviews and ratings?

        Returns:
        Instance of Submission().    
        """

        # Open url.
        self.driver.get(url)

        # Get submission title and ID.
        title = self.driver.find_element(By.CLASS_NAME, "citation_title").text
        content = self.driver.find_element(By.XPATH, "//div[@class='forum-note']/div[@class='note-content']").text
        sub_id = content.split("Number:")[1].strip()

        # Get replies.
        def load_reviews(driver):
            while True:
                # Keep trying until page loads...
                replies = driver.find_element(By.ID, "forum-replies").find_elements(By.CLASS_NAME, "depth-odd")
                if replies:
                    break
            return replies

        if skip_reviews:
            reviews = []
        else:
            reviews = run_with_timeout(
                load_reviews,
                (self.driver,),
                timeout_duration=TIMEOUT_DURATION,
                default_output=[]
            )

        # Get ratings and confidences from each valid rating.
        ratings, confidences, final_ratings = self._parse_rating(reviews)

        return Submission(title, sub_id, ratings, confidences, final_ratings)

    def load_all_submissions(self, skip_reviews: bool = False) -> list[Submission]:
        """Get all submission info."""
        subs = [self.load_submission(paper_url, skip_reviews) for paper_url in tqdm(self.paper_urls)]
        return subs


def print_csv(subs: list[Submission]):
    """Basic print as CSV."""
    output = ""
    max_len = 0
    for idx, sub in enumerate(subs):
        line = f"{idx + 1}, {str(sub)}"
        output += line + "\n"
        max_len = max(max_len, len(line))

    left = (max_len - 5) // 2
    right = max_len - 5 - left
    print("-" * left + " CSV " + "-" * right)
    print(output.rstrip("\n"))
    print("-" * max_len)


def print_rich(subs: list[Submission]):
    """Pretty print table."""

    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", justify="right")
    table.add_column("ID", justify="right")
    table.add_column("Title", justify="left")
    table.add_column("Ratings", justify="right")
    table.add_column("Avg.", justify="right")
    table.add_column("Std.", justify="right")
    table.add_column("Confidences", justify="right")
    table.add_column("Final Ratings", justify="right")
    table.add_column("Avg.", justify="right")
    table.add_column("Std.", justify="right")

    for idx, sub in enumerate(subs):
        table.add_row(
            f"{idx + 1}",
            sub.sub_id,
            sub.title,
            int_list_to_str(sub.ratings),
            mean(sub.ratings),
            std(sub.ratings),
            int_list_to_str(sub.confidences),
            int_list_to_str(sub.final_ratings),
            mean(sub.final_ratings),
            std(sub.final_ratings),
        )

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode?")
    parser.add_argument("--skip_reviews", action="store_true",
                        help="Skip reviews? Select if no reviews are in yet.")
    parser.add_argument("--conf", type=str,
                        default="cvpr_2026", choices=list(ORAPI.conf_to_url.keys()))
    parser.add_argument("--simulate", action="store_true",
                        help="Simulate the process.")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    if args.simulate:
        subs = []
        for _ in range(5):
            ratings = [random.choice(range(1, 5 + 1)) for _ in range(random.randint(0, 3))]
            final_ratings = [random.choice(range(1, 5 + 1)) for _ in range(random.randint(0, 3))]
            subs.append(Submission(
                title="Title " + random.choice(string.ascii_uppercase),
                sub_id=str(random.choice(range(1000, 20000))),
                ratings=ratings,
                confidences=[random.choice(range(1, 5 + 1)) for _ in range(len(ratings))],
                final_ratings=final_ratings
            ))
    else:
        # Initialize API object and get all info.
        obj = ORAPI(conf=args.conf, headless=args.headless)
        subs = obj.load_all_submissions(args.skip_reviews)

    # Print info.
    print_rich(subs)
    print_csv(subs)


if __name__ == "__main__":
    main()
