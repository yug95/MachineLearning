# -*- coding: utf-8 -*-
"""
This code is just for educational purpose.
Created on Tue Jan  6 2021

@author: Yug Agrawal
"""
import bs4
import pandas as pd
import json
from requests import get
import re
import numpy as np

titles = pd.read_excel("Movie_Name_List.xlsx")


def getIndex(movie_containers, tag):
  for index, container in enumerate(movie_containers):
      try:
        if container.div.text == tag:
          return index
          break
      except:
        pass

def Streaming_date(Moviemeta):
    try:
        Release_Streaming_index = getIndex(Moviemeta, 'Release Date (Streaming):')
        Streaming_Release_date = Moviemeta[Release_Streaming_index].text.strip().replace('\n', '').split(':')[1].strip()
    except:
        Streaming_Release_date = None
    
    return Streaming_Release_date


def Theatre_rlease_date(Moviemeta):
    try:
        Release_Th_index = getIndex(Moviemeta, 'Release Date (Theaters):')
        Theatre_Release_date = Moviemeta[Release_Th_index].text.strip().replace('\n', '').split(':')[1].strip()
    except:
        Theatre_Release_date = None
    
    return Theatre_Release_date

def title_clean(col_name):
    titles[col_name] = titles[col_name].str.replace(":","")
    titles[col_name] = titles[col_name].str.replace(" ","_")
    titles[col_name] = titles[col_name].str.replace("&","and")
    titles[col_name] = titles[col_name].str.replace("'","")
    titles[col_name] = titles[col_name].str.replace("-","")
    titles[col_name] = titles[col_name].str.replace(",","")
    titles[col_name] = titles[col_name].str.replace(".","")
    titles[col_name] = titles[col_name].str.replace("__","_")
    titles[col_name] = titles[col_name].str.replace("/","")
    titles[col_name] = titles[col_name].str.replace("!","")
    titles[col_name] = titles[col_name].str.replace("(","")
    titles[col_name] = titles[col_name].str.replace(")","")
    titles[col_name] = titles[col_name].str.lower()

title_clean('Title')
title_clean('Title_with_year')

titles['RT_Score'] = 0
titles['RT_Score_without_year_flag'] = 0
titles['with_year'] = 0

titles['Theatre_Release_date'] = np.nan
titles['Stream_Release_date'] = np.nan

index_no = 0
for title,Title_with_year in zip(titles.Title,titles.Title_with_year):
    print(title)

    try:
        page_movie = 'https://www.rottentomatoes.com/m/'+Title_with_year
        response = get(page_movie)
        soup = bs4.BeautifulSoup(response.text, 'lxml')
        
        # Score
        score = soup.find_all('span', class_='mop-ratings-wrap__percentage')
        titles.loc[index_no,'RT_Score'] = score[0].text.strip().replace('\n', '').split(' ')[0]
        titles.loc[index_no,'with_year'] = 1
        
        Moviemeta = soup.find_all('li', class_='meta-row clearfix')
        Streaming_Release_date = Streaming_date(Moviemeta)
        Theatre_Release_date = Theatre_rlease_date(Moviemeta)
        
        titles.loc[index_no,'Theatre_Release_date'] = Theatre_Release_date
        titles.loc[index_no,'Stream_Release_date'] = Streaming_Release_date
        
        index_no = index_no + 1
    except:
        try:
            page_movie = 'https://www.rottentomatoes.com/m/'+title
            response = get(page_movie)
            soup = bs4.BeautifulSoup(response.text, 'lxml')
            
            # Score
            score = soup.find_all('span', class_='mop-ratings-wrap__percentage')
            titles.loc[index_no,'RT_Score_without_year_flag'] = score[0].text.strip().replace('\n', '').split(' ')[0]
            
            Moviemeta = soup.find_all('li', class_='meta-row clearfix')
            Streaming_Release_date = Streaming_date(Moviemeta)
            Theatre_Release_date = Theatre_rlease_date(Moviemeta)
            
            titles.loc[index_no,'Theatre_Release_date'] = Theatre_Release_date
            titles.loc[index_no,'Stream_Release_date'] = Streaming_Release_date
            
            index_no = index_no + 1
        except:
            titles.loc[index_no,'RT_Score'] = "Not Available"
            titles.loc[index_no,'RT_Score_without_year_flag'] = "Not Available"
            index_no = index_no + 1
            pass
 

titles.to_excel("Rotten_Tomato_Scarpper_code_output.xlsx")