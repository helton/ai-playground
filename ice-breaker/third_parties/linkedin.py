import os
import requests


def cached_scraped_linkedin_profile(linkedin_profile_url: str):
    """scrape information from LinkedIn profile,
    Cached scraped data for sample LinkedIn profiles"""

    # fetch remote profiles map from secret gist
    profiles_url_map = requests.get(
        os.environ.get("SECRET_GIST_PROFILES_MAP_URL")
    ).json()

    # fetch linkedin data from secret gist
    response = requests.get(profiles_url_map[linkedin_profile_url])

    return clean_scraped_data(response.json())


def do_scrape_linkedin_profile(linkedin_profile_url: str):
    """scrape information from LinkedIn profile,
    Manually scrape the information from LinkedIn profile using proxycurl API"""
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    headers_dic = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}
    response = requests.get(
        api_endpoint, params={"url": linkedin_profile_url}, headers=headers_dic
    )
    print(response.json())
    return clean_scraped_data(response.json())


def scrape_linkedin_profile(linkedin_profile_url: str, use_cache: bool = True):
    if use_cache:
        return cached_scraped_linkedin_profile(linkedin_profile_url)
    else:
        return do_scrape_linkedin_profile(linkedin_profile_url)


def clean_scraped_data(data):
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", None) and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")
    return data
