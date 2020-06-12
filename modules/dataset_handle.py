import os
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def remove_col_df(dataframe, col_to_remove):
  for c in col_to_remove:
    if c in dataframe.columns:
      dataframe.drop(columns=[c], axis=1,inplace=True)


def build_dataset(dataframe, num_features, norm=False):
  remove_col_df(dataframe, ('ID', '#ID', 'redshiftErr'))
  all_data = dataframe.to_numpy()
  x = all_data[:,0:(num_features)]
  y = all_data[:,-1]

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
  
  scaler = None
  if norm:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

  return x_train, y_train, x_test, y_test, x_val, y_val, scaler


def load_dataframe(dataset_name):
  if dataset_name == 'teddy':
    return pd.read_csv('teddy_train_data', comment='#', delim_whitespace=True, names=['ID','u','g','r','i','z','uErr','gErr','rErr','iErr','zErr','redshift','redshiftErr'])

  if dataset_name == 'happy':
    return pd.read_csv('happy_train_data', comment='#', delim_whitespace=True, names=['ID','u','g','r','i','z','uErr','gErr','rErr','iErr','zErr','redshift','redshiftErr'])

  if dataset_name == 'kaggle' | dataset_name == 'kaggle_bkp':
    return pd.read_csv('kaggle_train_data.csv')

  return None


def download_data(dataset_name):
  if dataset_name == 'teddy':
    download_teddy()
  if dataset_name == 'happy':
    download_happy()
  if dataset_name == 'kaggle':
    download_kaggle()
  if dataset_name == 'kaggle_bkp':
    download_kaggle_alternative()


def download_teddy():
  out = os.system('wget https://raw.githubusercontent.com/COINtoolbox/photoz_catalogues/master/Teddy/forTemplateBased/teddyT_A.cat')
  print(out)
  out = os.system('mv teddyT_A.cat teddy_train_data')
  print(out)


def download_happy():
  out = os.system('https://raw.githubusercontent.com/COINtoolbox/photoz_catalogues/master/Happy/forTemplateBased/happyT_A')
  print(out)
  out = os.system('mv happyT_A happy_train_data')
  print(out)


def download_kaggle():
  out = os.system('wget -O archieve.zip "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/3617/105309/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1589667788&Signature=PqUjFhjNMhS9DEBnk3Yxzo%2BXnKS9c3v1XbmW%2F2dVcromNwDnELmnVSAtGD%2BFi%2FiNay801TWg84P0N6qCq8D5P%2BcOj92zcBB0%2BjPX8yIKvvwud3mA%2FCW%2FLAXTCbgGHrNKUUSIAMhkwro41Nzi%2F4lP0pIpCugQr44bWnCmh57zmt5CNJDb63ZS7CiZyZGsNg6YebXTAzR3yuNbdtMd3pvmQJwv%2FPedtyoR4PhloelwqE3fxVx2pt9AdZzE%2FU%2BRVyoFCnj1EiQBq8oDES2TEoJde9rTCIMWM6OtFPa%2Bq3e%2BqDC9lzgkywTPJ4hjmRA4VmFgPWb57uOX7T2QZuHnnIG87g%3D%3D&response-content-disposition=attachment%3B+filename%3Dphotometric-redshift-estimation-2013.zip"')
  print(out)
  out = os.system('unzip -u archieve.zip')
  print(out)
  out = os.system('unzip -u train.zip')
  print(out)
  out = os.system('mv train.csv kaggle_train_data.csv')
  print(out)


def download_kaggle_alternative():
  def download_file_from_google_drive(id):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
      params = {'id': id, 'confirm': token}
      response = session.get(URL, params=params, stream=True)

    save_response_content(response)


  def get_confirm_token(response):
    for key, value in response.cookies.items():
      if key.startswith('download_warning'):
        return value

    return None


  def save_response_content(response):
    CHUNK_SIZE = 32768

    with open("kaggle_train_data.csv", "wb") as f:
      for chunk in response.iter_content(CHUNK_SIZE):
        if chunk:  # filter out keep-alive new chunks
          f.write(chunk)


  download_file_from_google_drive('1xCQzmusNSt65zQnjsrnddLhuCB5WdYcl')
