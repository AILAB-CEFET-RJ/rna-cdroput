import os
import pandas as pd
import numpy as np
import requests
import gdown

from sklearn.model_selection import train_test_split


def remove_col_df(dataframe, col_to_remove):
  for c in col_to_remove:
    if c in dataframe.columns:
      dataframe.drop(columns=[c], axis=1,inplace=True)


def ugriz_errs_split(x, chunks=2):
  x_splited = np.hsplit(x, chunks)
  if chunks == 2:
    return x_splited[0], x_splited[1], [[]]
  else:
    return x_splited[0], x_splited[1], x_splited[2]


def filter_col(dataframe):
  remove_col_df(dataframe, ('ID', '#ID', 'redshiftErr','objid', 'specobjid', 'class'))


def filter_negative_redshift(dataframe):
  orig = dataframe.shape[0]
  dataframe = dataframe[dataframe.u > 0]
  dataframe = dataframe[dataframe.g > 0]
  dataframe = dataframe[dataframe.r > 0]
  dataframe = dataframe[dataframe.i > 0]
  dataframe = dataframe[dataframe.z > 0]
  clean = dataframe.shape[0]
  print(f"Negative mags removed: {orig-clean}.")

  return dataframe


def build_dataset(dataframe, num_features, scaler):
  all_data = dataframe.to_numpy()
  x = all_data[:,0:(num_features)]
  y = all_data[:,-1]

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

  chunks = 2
  if num_features > 10:
    chunks = 3

  if num_features > 5:
    x_train_ugriz, x_train_errs, x_train_experrs = ugriz_errs_split(x_train, chunks)
    x_val_ugriz, x_val_errs, x_val_experrs = ugriz_errs_split(x_val, chunks)
    x_test_ugriz, x_test_errs, x_test_experrs = ugriz_errs_split(x_test, chunks)

    if scaler != None:
      x_train_ugriz = scaler.fit_transform(x_train_ugriz)
      x_val_ugriz = scaler.transform(x_val_ugriz)
      x_test_ugriz = scaler.transform(x_test_ugriz)

    if chunks == 2:
        x_train = np.hstack((x_train_ugriz, x_train_errs))
        x_val = np.hstack((x_val_ugriz, x_val_errs))
        x_test = np.hstack((x_test_ugriz, x_test_errs))
    else:
        x_train = np.hstack((x_train_ugriz, x_train_errs, x_train_experrs))
        x_val = np.hstack((x_val_ugriz, x_val_errs, x_val_experrs))
        x_test = np.hstack((x_test_ugriz, x_test_errs, x_test_experrs))
  else:
    if scaler != None:
      x_train = scaler.fit_transform(x_train)

  return x_train, y_train, x_test, y_test, x_val, y_val, scaler


def load_dataframe(dataset_name):
  if dataset_name == 'teddy':
    return pd.read_csv('teddy_train_data', comment='#', delim_whitespace=True, names=['ID','u','g','r','i','z','uErr','gErr','rErr','iErr','zErr','redshift','redshiftErr'])

  if dataset_name == 'happy':
    return pd.read_csv('happy_train_data', comment='#', delim_whitespace=True, names=['ID','u','g','r','i','z','uErr','gErr','rErr','iErr','zErr','redshift','redshiftErr'])

  if dataset_name == 'kaggle' or dataset_name == 'kaggle_bkp':
    return pd.read_csv('kaggle_train_data.csv')

  if dataset_name == 'sdss':
    return pd.read_csv('sdss_train_data.csv', comment="#")

  return None


def download_data(dataset_name):
  if dataset_name == 'teddy':
    if os.path.isfile('teddy_train_data'):
      print("Dataset Found!")
    else:
      download_teddy()

  if dataset_name == 'happy':
    if os.path.isfile('happy_train_data'):
      print("Dataset Found!")
    else:
      download_happy()

  if dataset_name == 'kaggle':
    if os.path.isfile('kaggle_train_data.csv'):
      print("Dataset Found!")
    else:
      download_kaggle()

  if dataset_name == 'kaggle_bkp':
    if os.path.isfile('kaggle_train_data.csv'):
      print("Dataset Found!")
    else:
      download_kaggle_alternative()

  if dataset_name == 'sdss':
    if os.path.isfile('sdss_train_data.csv'):
      print("Dataset Found!")
    else:
      download_sdss_alternative()


def download_teddy():
  os.system('wget https://raw.githubusercontent.com/COINtoolbox/photoz_catalogues/master/Teddy/forTemplateBased/teddyT_A.cat')
  os.system('mv teddyT_A.cat teddy_train_data')


def download_happy():
  os.system('wget https://raw.githubusercontent.com/COINtoolbox/photoz_catalogues/master/Happy/forTemplateBased/happyT_A')
  os.system('mv happyT_A happy_train_data')


def download_kaggle():
  os.system('wget -O archieve.zip "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/3617/105309/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1589667788&Signature=PqUjFhjNMhS9DEBnk3Yxzo%2BXnKS9c3v1XbmW%2F2dVcromNwDnELmnVSAtGD%2BFi%2FiNay801TWg84P0N6qCq8D5P%2BcOj92zcBB0%2BjPX8yIKvvwud3mA%2FCW%2FLAXTCbgGHrNKUUSIAMhkwro41Nzi%2F4lP0pIpCugQr44bWnCmh57zmt5CNJDb63ZS7CiZyZGsNg6YebXTAzR3yuNbdtMd3pvmQJwv%2FPedtyoR4PhloelwqE3fxVx2pt9AdZzE%2FU%2BRVyoFCnj1EiQBq8oDES2TEoJde9rTCIMWM6OtFPa%2Bq3e%2BqDC9lzgkywTPJ4hjmRA4VmFgPWb57uOX7T2QZuHnnIG87g%3D%3D&response-content-disposition=attachment%3B+filename%3Dphotometric-redshift-estimation-2013.zip"')
  os.system('unzip -u archieve.zip')
  os.system('unzip -u train.zip')
  os.system('mv train.csv kaggle_train_data.csv')


def download_sdss_alternative():
  id = '1i5fFrVlVkfRvFMBYzYbE_whLfSTz5fYe'
  url = f"https://drive.google.com/uc?id={id}"
  output = 'sdss_train_data.csv'
  gdown.download(url, output, quiet=False)

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
