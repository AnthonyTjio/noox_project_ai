# How to use News Dataset Generator:
### Basic Usage: 

Possible Functions:
- **crawl** : Crawl websites based on config file
```
example:
python3 main.py crawl
```
- **scan** : Scan website links and retrieve title and article of the website based on the config file, later on these information will be written into data.csv along with the predetermined tag
```
example:
python3 main.py scan
```
- **compare** : Create 2 csv files based on data.csv which will compare possible similar articles with different tag
```
example:
python3 main.py compare 
```
- **clean** : Clean csv file using regex to certain rows in the csv file with format (csv file) - (regex) - (row numbers)
```
example:
python3 main.py clean dum_data.csv "(CAPTCHA Code*).*" 1 2 3 4
```
- **bagging** : Perform bagging / bootstrap aggregation on the source csv file to multiply datasets (Currently not available)

### Things to Remember
Always check conf.json for crawl and scan configuration