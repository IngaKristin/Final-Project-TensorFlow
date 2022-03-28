"""
This module contains all preprocessing functions

Created: 28.03.22, 15:25

Author: LDankert
"""
import pandas as pd


def duplicate_multiple_styles(data):
    """ This functions duplicates every data entry, that hast more than one style. It
    returns multiple data entries, one for each music style, containing only one style.
    If the data entry just has one style it just return it

    :param
        data: a pandas dataframe entry containing a "stlye" column
    :return:
        duplicated_datas(pd.DataFrame) DataFrame with one or multiple entries
    """
    duplicated_datas = pd.DataFrame()
    styles = data.style.split("/")
    for style in styles:
        dublicated_data = data
        dublicated_data["style"] = style
        duplicated_datas = duplicated_datas.append(dublicated_data, ignore_index=True)
    return duplicated_datas

