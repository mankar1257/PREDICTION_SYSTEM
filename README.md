# CROWN_AI
  
## Install guide

#####  required Python version Python 3.5 

##### Clone the repo

```bash
$ git clone https://github.com/mankar1257/CROWN_AI.git
$ cd CROWN_AI
```

##### Create the virtualenv
```bash
$ mkvirtualenv CROWN_AI
```

##### Install dependencies
```bash
$ pip install -r requirements.txt
```


## Setting environment variables

#### FOR UBUNTU 

Open environment file 

```
$ sudo -H gedit /etc/environment
```
Add this into environment file 

```
$ DB="mongodb+srv://CROWN_AI:1234@cluster0-eowpu.mongodb.net/<dbname>?retryWrites=true&w=majority"

```


## Run the app
```bash
$ cd API
$ python3 app.py
```


## Test

Using Postman

URL : http://127.0.0.1:5000/find?H=1&I=1
 
Using Python request 

```
import requests

x = requests.get('http://127.0.0.1:5000/find?H=1&I=1')

x = x.json()

print(x)
```
 
Feel  free to change the values of H ( Hospital ) and I (item )

 
