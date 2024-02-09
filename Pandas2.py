#Demonstration of Pandas library.

import pandas as pd
import xlsxwriter



data = [{'Name':'PPA','Duration':4,'Fees':10500},{'Name':'LB','Duration':3,'Fees':10500},
{'Name':'python','Fees':10500}]
df =  pd.DataFrame(data)
print(df)

writer = pd.ExcelWriter('Marvellouspandas.xlsx',engine='xlsxwriter')

df.to_excel(writer,sheet_name='Sheet1')

writer.save