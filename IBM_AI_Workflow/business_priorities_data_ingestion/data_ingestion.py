#!/usr/bin/env python
# -*- coding:utf8 -*-

import os
import sys
import pandas as pd
import numpy as np
import sqlite3
import getopt


def connect_db(file_name):
    try:
        conn = sqlite3.connect(file_name)
        print("connect to database successful!")
    except Error as e:
        print("connect to database failed!")

    return conn


def ingest_db(conn):
    tables = [t[0] for t in conn.execute(
              "SELECT name FROM sqlite_master WHERE type='table';"
              # "SELECT name FROM sqlite_master;"
              )]
    # tables = ['CUSTOMERS', 'INVOICES', 'INVOICE_ITEMS']
    # database doesn't have 'COUNTRY' table, instead country_name
    # is also stored in 'CUSTOMERS' table

    # query user information from the database
    query = """
    SELECT cu.customer_id, cu.last_name, cu.first_name,
           cu.DOB, cu.city, cu.state, cu.country, cu.gender
    FROM CUSTOMERS cu
    """
    _data = [d for d in conn.execute(query)]
    columns = ['customer_id', 'last_name', 'first_name', 'DOB',
               'city', 'state', 'country', 'gender']
    df_cust = pd.DataFrame(_data, columns=columns)
    print('{} lines of customer information read from database'
          .format(df_cust.shape[0]))

    # Clean data: remove duplicate data from the database
    duplicate = df_cust.duplicated()
    df_cust = df_cust[~duplicate]

    return(df_cust)


def ingest_csv(csv_file):
    # csv file contains customer subscription information
    # check if customer has subscipbed any channel
    df_stream = pd.read_csv(csv_file)
    customer_id = df_stream['customer_id'].values
    unify_id = np.unique(customer_id)
    subscription = df_stream['subscription_stopped'].values
    is_subscriber = [0 if subscription[customer_id == uid].max() > 0
                     else 1 for uid in unify_id]
    cust_subscribe = pd.DataFrame({'customer_id': unify_id,
                                   'is_subscriber': is_subscriber})
    print('find subscription information of {} customers'
          .format(cust_subscribe.shape[0]))

    # clean data
    validate_stream_id = np.isnan(df_stream['stream_id'])
    df_stream = df_stream[~validate_stream_id]

    return(df_stream, cust_subscribe)


def process_data(conn, df_cust, df_stream, cust_subscribe):

    # aggregate all data to a singel DataFrame 'df_clean'
    # df_clean result like below
    #  customer_id is_subscribe country age customer_name subscrib_type_
    # 0     1           1  united_states   21  Kasen Todd  aavail_unlimited
    df_clean = cust_subscribe.copy()
    df_clean = df_clean[np.in1d(df_clean['customer_id'].values,
                                df_cust['customer_id'].values)]
    if not np.equal(df_clean['customer_id'].values.all(),
                    df_cust['customer_id'].values.all()):
        raise Exception('index are out of order or mismatch, need to fix!')

    df_clean['country'] = df_cust['country'].values
    df_clean['age'] = (np.array('today', dtype='datetime64')
                       - df_cust['DOB'].astype('datetime64').values)
    df_clean['age'] = df_clean['age'].astype('timedelta64[Y]').astype(int)
    df_clean['customer_name'] = (df_cust['first_name']
                                 + ' '
                                 + df_cust['last_name'])
    customer_id = df_stream['customer_id'].values
    query = """
    SELECT i.invoice_item_id, i.customer_id
    FROM INVOICES i
    """
    customer_map = {cm[1]: cm[0] for cm in conn.execute(query)}
    df_clean['invoice_item_id'] = [customer_map[uid] for uid
                                   in np.unique(customer_id)]
    query = """
    SELECT i_i.invoice_item_id, i_i.invoice_item
    FROM INVOICE_ITEMS i_i
    """
    invoice_map = {im[0]: im[1] for im in conn.execute(query)}
    df_clean['subscriber_type'] = [invoice_map[iid] for iid
                                   in df_clean['invoice_item_id'].values]
    df_clean['num_streams'] = [df_stream[customer_id == uid].size for uid
                               in np.unique(customer_id)]
    df_clean = df_clean.drop(['invoice_item_id'], axis=1)
    print('Processed data generated! Totally {} lines.'
          .format(df_clean.shape[0]))
    print(df_clean.head())
    return(df_clean)


def update_target(target_file, df_clean, overwrite=False):
    if overwrite or not os.path.exists(target_file):
        df_clean.to_csv(target_file, index=False)
    else:
        df_clean.to_csv(target_file, mode='a', index=False)


if __name__ == "__main__":
    arg_string = "%s -d data_file -s stream_file -t target_file" % sys.argv[0]
    try:
        optlist, _ = getopt.getopt(sys.argv[1:], 'd:s:t:')
    except getopt.GetoptError:
        print(getopt.GetoptError)
        raise Exception(arg_string)

    data_file = ''
    stream_file = ''
    target_file = ''
    for opt in optlist:
        if opt[0] == '-d':
            data_file = opt[1]
        if opt[0] == '-s':
            stream_file = opt[1]
        if opt[0] == '-t':
            target_file = opt[1]

    if data_file == '' or stream_file == '':
        print('data_file & stream_file cannot be empty! Usage: %s',
              arg_string)
        sys.exit()

    conn = connect_db(data_file)
    df_cust = ingest_db(conn)
    df_stream, cust_subscribe = ingest_csv(stream_file)
    df_clean = process_data(conn, df_cust, df_stream, cust_subscribe)

    if not target_file:
        target_file = os.path.join('.', 'data', 'aavail-target.csv')
    update_target(target_file, df_clean, overwrite=False)
    print('done!')
