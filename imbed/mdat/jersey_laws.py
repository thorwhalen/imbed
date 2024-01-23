"""To acquire and prepare jersey law data.

MANUAL WORK NEEDED:

1. Go to https://www.jerseylaw.je/laws/current/Pages/search.aspx?size=n_500_n
2. Copy the html from the "inspect" tool in your browser.
3. Save it into an `.html` file in a folder called `law_htmls` in some root directory of your choice.
4. Repeat for each page of results, and save each page to a different file.

--> Tried to scrape this automatically, but wasn't easy, so I did it manually.


"""

import os


laws_list_url = 'https://www.jerseylaw.je/laws/current/Pages/search.aspx?size=n_500_n'


def extract_ref(url):
    import re

    return re.search(r'([\d\.]+)\.aspx', url).group(1)


def extract_info(html):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find("tbody", {"class": "MuiTableBody-root"})
    rows = table.find_all("a", {"class": "MuiTableRow-root resultRow"})

    def gen():
        for row in rows:
            d = {'name': row.text, 'url': row.get('href')}
            ref = extract_ref(d['url'])
            d['ref'] = ref
            if d['name'] == 'Communications (Jersey) Order 2020':
                # Special case: this law has a space in the pdf filename (error in the website)
                d['pdf'] = f"https://www.jerseylaw.je/laws/current/PDFs/{ref} .pdf"
            else:  # Normal case
                d['pdf'] = f"https://www.jerseylaw.je/laws/current/PDFs/{ref}.pdf"
            yield d

    return list(gen())


from dol import TextFiles, filt_iter


@filt_iter(filt=lambda x: x.endswith('html') and x.startswith('jersey_laws_'))
class Htmls(TextFiles):
    """jersey law htmls"""


from itertools import chain
from typing import Union, Mapping

Folderpath = str


def htmls_store(htmls: Union[Folderpath, Mapping[str, str]]) -> Htmls:
    if isinstance(htmls, str):
        if os.path.isdir(htmls):
            folderpath = htmls
            htmls = Htmls(folderpath)
        else:
            raise ValueError(f"Folderpath {htmls} does not exist")
    return htmls


def gather_info(htmls: Union[Folderpath, Mapping[str, str]]):
    htmls = htmls_store(htmls)
    for d in chain.from_iterable(map(extract_info, htmls.values())):
        d['pdf_url'] = d['url'].replace('Pages/Default.aspx', 'Documents')


def get_laws_info(htmls: Union[Folderpath, Mapping[str, str]]):
    htmls = htmls_store(htmls)
    laws_info = list(chain.from_iterable(map(extract_info, htmls.values())))
    assert all(x['pdf'] for x in laws_info)
    assert len(set(x['name'] for x in laws_info)) == len(laws_info)
    return laws_info
