# CROWN_AI
  
## Install guide

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

##### Run the app
```bash
$ cd AI_API
$ python3 app.py
```


## Setting environment variables

### FOR UBUNTU 

Open environment file 

```
$ sudo -H gedit /etc/environment
```
Add this into environment file 

```
$ DB="mongodb+srv://CROWN_AI:1234@cluster0-eowpu.mongodb.net/<dbname>?retryWrites=true&w=majority"

```

## Test

Using Postman

URL : http://127.0.0.1:5000/find?lat=36.336159&lng=-119.645667&radius=50
 
Using Python request 

```
import requests

x = requests.get('http://127.0.0.1:5000/find?lat=36.336159&lng=-119.645667&radius=50')

x = x.json()

print(x)
```
 
Feel  free to change the values of radius, lat, and lng ( within US ) 

 
