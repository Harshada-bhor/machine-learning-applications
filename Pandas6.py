#Demonstration of Pandas library.

import os
from sys import *
import fnmatch
import xlsxwriter

 
def ExcelCreate(name):
    workbook = xlsxwriter.Workbook(name)

    worksheet = workbook.add_worksheet()

    worksheet.write('A1','Name')
    worksheet.write('B1', 'College')
    worksheet.write('C1', 'Mail ID')
    worksheet.write('D1', 'Mobile')

    workbook.close()


def main():
    print("------------Demonstration of Pandas library___________")
    print("Application name:" +argv[0])

    if(len(argv) !=2):
        print("Error: Insufficint arguments")
        print("Use -h for help and use -u for usage of the script")
        exit()

    if((argv[1] == "-h" )or (argv[1]=="-H")):
        print("Help:This script used to perform__")
        exit()

    if((argv[1] == "-u") or (argv[1]== "-U")):
        print("Usage: Application_Name of file")
        exit()

    try:
        ExcelCreate(argv[1])

    except Exception :
        print("Error : Invalid input" )

if __name__=="__main__":
    main()