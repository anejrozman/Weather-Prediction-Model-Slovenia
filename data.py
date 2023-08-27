'''
data.py fetches and parses data from: http://www.arso.gov.si/ which 
is the Slovenian national weather service.

@Author: Anej Rozman
'''

import pandas as pd
import numpy as np
import requests
import time
import datetime as dt
import chompjs
from bs4 import BeautifulSoup


# Set to True if having trouble fetching data
PAUSE = False 

# Id's for fetching data for model training
TRAIN_REGIONS = {
    'NG':'1822',
    'LJ': '1828',
    'MS': '1842', 
    'PO':'1849', 
    'CE':'2471',
    'NM': '1832',
    'JE':'2213',
}

# Column names for training data
TRAIN_COLNAMES = {
    'p0':'temp',
    'p1':'moisture',
    'p2':'rain',
    'p3':'wind_speed',
    'p4':'wind_dir',
    'p5':'pressure',       
}


# Parses historic ARSO data for model training
def fetch_train(url, data_id):
    try:
        headers = {
            "accept": "*/*",
            "accept-language": "sl-SI,sl;q=0.9,en-GB;q=0.8,en;q=0.7,zh-CN;q=0.6,zh;q=0.5",
            "sec-ch-ua": "\"Not.A/Brand\";v=\"8\", \"Chromium\";v=\"114\", \"Google Chrome\";v=\"114\"",
            "sec-ch-ua-mobile": "?1",
            "sec-ch-ua-platform": "\"Android\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin"
        }

        with requests.Session() as s:
            s.headers.update(headers)
            response = s.get(url)
            response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes

            content = response.text.split("CDATA[AcademaPUJS.set(")[1].split(")]]></pujs>")[0]
            parsed_content = chompjs.parse_js_object(content)
            data = {i: parsed_content['points']['_' + data_id][i]
                    for i in parsed_content['points']['_' + data_id]}

            df = pd.DataFrame(data)
            df = df.T
            df.rename(columns=TRAIN_COLNAMES, inplace=True)
            return df
        
    except requests.exceptions.RequestException as error:
        print(f"An error occurred while fetching data: {error}")
        return None
    except Exception as error:
        print(f"An error occurred: {error}")
        return None

# Id's for fetching data for model testing
TEST_REGIONS = {
    'NG':'NOVA-GOR',
    'LJ': 'LJUBL-ANA_BEZIGRAD',
    'MS': 'MURSK-SOB', 
    'PO':'PORTOROZ_SECOVLJE', 
    'CE':'CELJE_MEDLOG',
    'NM': 'NOVO-MES',
    'JE':'LESCE',   
}

# Wind direction comes in png format, so 
# I have to replace it with degrees
WIND_DIRECTION = {
    'lightS': 0,
    'lightSW': 45,
    'lightW': 90,
    'lightNW': 135,
    'lightN': 180,
    'lightNE': 225,
    'lightE': 270,
    'lightSE': 315
}

# column names for testing data
TEST_COLNAMES = {
    0: 'temp', 
    1: 'moisture', 
    2: 'wind_speed',
    3: 'wind_dir',
    4: 'rain',
}

# Parses ARSO data from last two days for model testing 
def fetch_test(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 400 and 500 status codes
        
        soup = BeautifulSoup(response.text, 'html.parser')
        rows = soup.find_all('tr')
        
        class_parameters = ['t', 'rh', 'ffavg_val', 'ddff_icon', 'rr_val']
        
        def td_filter(tag):
            return (
                tag.name == 'td' and  
                len(tag.attrs) == 1 and  
                tag['class'][0] in class_parameters  
            )
        
        data = []
        for row in rows[1:]:
            td_elements = row.find_all(td_filter)
            values = [td.text for td in td_elements]
            try:
                wind_dir = td_elements[3].find('img')['src'].split('/')[-1].split('.')[0]
                values[3] = WIND_DIRECTION.get(wind_dir)
            except Exception as e:
                print(f"Error parsing wind direction: {e}")
                values[3] = None
            data.append(pd.DataFrame(values).T)
        
        data = pd.concat(data)
        data.rename(columns=TEST_COLNAMES, inplace=True)
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Main function
def main():
    train_data = {}
    test_data = {}
    
    try:
        t = dt.datetime.now().date() - dt.timedelta(days=2)
        short_interval = dt.timedelta(days=44)
        yearly_interval = dt.timedelta(days=365)
        
        for i in TRAIN_REGIONS:
            try:
                for j in range(3):
                    url = f"https://meteo.arso.gov.si/webmet/archive/data.xml?lang=si&vars=26,21,15,23,24,18&group=halfhourlyData0&type=halfhourly&id={TRAIN_REGIONS[i]}&d1={str(t - short_interval)}&d2={str(t)}&nocache=ll18z24i2gvfsaiwd1s"
                    train_data[f'{i}{j}'] = fetch_train(url, TRAIN_REGIONS[i])
                    t -= yearly_interval
                    if PAUSE:
                        time.sleep(np.random.uniform(1, 4))
                t = dt.datetime.now().date() - dt.timedelta(days=2)
            except Exception as train_exc:
                print(f"An error occurred while fetching train data for region {i} in loop {j}: {train_exc}")
                
        for i in TEST_REGIONS:
            try:
                url = f'https://meteo.arso.gov.si/uploads/probase/www/observ/surface/text/sl/observationAms_{TEST_REGIONS[i]}_history.html'
                test_data[i] = fetch_test(url)
                if PAUSE:
                    time.sleep(np.random.uniform(1, 4))
            except Exception as test_exc:
                print(f"An error occurred while fetching test data for region {i}: {test_exc}")
        
    except Exception as main_exc:
        print(f"An error occurred in the main function: {main_exc}")

    return train_data, test_data

