import pandas as pd
import numpy as np

def process(file):
    try:
        df = None
        if(file.content_type=="text/csv"):
            df = pd.read_csv(file,encoding='unicode_escape')
                
        else:
            df = pd.read_excel(file)

        return df.copy(),df.columns
    except Exception as e:
        print(e)
        return None,None
            
    