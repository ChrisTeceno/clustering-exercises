from env import get_db_url
import pandas as pd
import os


def get_zillow_data(use_cache=True):
    """pull from SQL unless zillow.csv exists"""
    filename = "zillow.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading from csv...")
        return pd.read_csv(filename)

    print("reading from sql...")
    url = get_db_url("zillow")
    query = """
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, fips
    FROM properties_2017
    JOIN propertylandusetype using (propertylandusetypeid) 
    JOIN predictions_2017 using (parcelid)
    WHERE propertylandusedesc IN ("Single Family Residential", "Inferred Single Family Residential")
    AND transactiondate like "2017%%";
    """
    df = pd.read_sql(query, url)

    print("Saving to csv in local directory...")
    df.to_csv(filename, index=False)
    return df


# this is used to import different data from sql if needed in the future
def get_zillow_data2(use_cache=True):
    """Same as above but pulls in more squarefoot data, pull from SQL unless zillow.csv exists"""
    filename = "zillow2.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading from csv...")
        return pd.read_csv(filename)

    print("reading from sql...")
    url = get_db_url("zillow")
    query = """
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet,finishedfloor1squarefeet, finishedsquarefeet12,finishedsquarefeet13, finishedsquarefeet15, finishedsquarefeet50, finishedsquarefeet6 ,taxvaluedollarcnt, yearbuilt, fips
    FROM properties_2017
    JOIN propertylandusetype using (propertylandusetypeid) 
    JOIN predictions_2017 using (parcelid)
    WHERE propertylandusedesc IN ("Single Family Residential", "Inferred Single Family Residential")
    AND transactiondate like "2017%%"; 
    """
    df = pd.read_sql(query, url)

    print("Saving to csv in local directory...")
    df.to_csv(filename, index=False)
    return df
