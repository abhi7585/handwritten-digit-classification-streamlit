# Handwritten Digit Recognition using OpenCV

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/abhi7585/handwritten-digit-classification-streamlit/main/main.py)

# 1. Introduction

- The handwritten digit recognition is the ability of computers to recognize human handwritten digits. 
- Developing such a system includes a machine to understand and classify the images of handwritten digits as 0-9.
- The handwritten digit recognition uses the image of a digit and recognizes the digit present in the image.

# 2. Objective and Algorithm 

### 2.1 Objective

- The objective of the project is to create a handwritten digit recognition model that can recognise the handwritten digits 
- Our second objective is to create a web application for our model to make it accessible for the users. The web app will take image as an input and will give the predicted results based on the model evaluations


### 2.2 Algorithm 

The data set which I chose for this problem is available on Kaggle. 

* LeNet CNN Model
1) Convolutional Neural Networks is the standard form of neural network architecture for solving tasks associated with images. Solutions for tasks such as object detection, face detection, pose estimation and more all have CNN architecture variants
2) LeNet-5 CNN architecture is made up of 7 layers. The layer composition consists of 3 convolutional layers, 2 subsampling layers and 2 fully connected layers


# 3. Streamlit Deployment

[Streamlit](https://streamlit.io/) is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.
In just a few minutes you can build and deploy powerful data apps - so let's get started!

1. Make sure that you have [Python 3.6 - Python 3.8](https://www.python.org/downloads/release/python-386/) installed.
2. Install Streamlit using [PIP](https://pip.pypa.io/en/stable/installing/) and run the 'hello world' app:

   ```shell
   pip install streamlit
   streamlit hello
   ```

3. That's it! In the next few seconds the sample app will open in a new tab in your default browser.

Now make your own app in just 3 more steps:

1. Open a new Python file, import Streamlit, and write some code

2. Run the file with:

   `streamlit run [filename]`

3. When you're ready, click 'Deploy' from the Streamlit menu to share your app.

#### Create an app

Working with Streamlit is simple. First you sprinkle a few Streamlit commands
into a normal Python script, then you run it with `streamlit run`:

```bash
streamlit run your_script.py [-- script args]
```

As soon as you run the script as shown above, a local Streamlit server will
spin up and your app will open in a new tab your default web browser. The app
is your canvas, where you'll draw charts, text, widgets, tables, and more.

#### Deploy an app

Now that you've created your app, you're ready to share it! Use **Streamlit sharing** to share it with the world completely for free. Streamlit sharing is the perfect solution if your app is hosted in a public GitHub repo and you'd like anyone in the world to be able to access it. If that doesn't sound like your app, then check out [Streamlit for Teams](https://streamlit.io/for-teams) for more information on how to get secure, private sharing for your apps.


##### Get a Streamlit sharing account

To get started, first request an invite at [streamlit.io/sharing](https://streamlit.io/sharing). Streamlit sharing is currently available by invitation only while we ramp things up. Once you have your invite you're ready to deploy! It's really straightforward, just follow the next few steps.

##### Put your Streamlit app on GitHub

Make sure your app is in a public GitHub repo and that you have a requirements.txt file.

- If you need to generate a requirements file, try using `pipreqs`

```bash
pip install pipreqs
pipreqs /home/project/location
```


#### Deploy your app

Click "New app", then fill in your repo, branch, and file path, and click "Deploy". Your app will take a minute or two to deploy and then you'll be ready to share!

If your app has a lot of dependencies it may take some time to deploy the first time. But after that, any change that does not touch your dependencies should show up immediately.

That's it — you're done! Your app can be found at:

```python
https://share.streamlit.io/[user name]/[repo name]/[branch name]/[app path]
```

Most times, your app is also given a shortened URL of the form `https://share.streamlit.io/[user name]/[repo name]`. The only time you need the full URL is when you deployed multiple apps from the same repo. 


# 4. Methodology

The project can be carried out in the following way:
- Import the libraries and load the dataset
- Preprocess the data
- Create the model
- Train the model
- Evaluate the model
- Create an application to predict digits



# 4. Result
The main objective is achieved the web application can successfully detects Handwritten digit.


# 5. Future Work

There is always a scope of improvement. Here are a few things which can be considered to improve. 
* The model can be modified to detect multiple digits in a single frame of reference.
* We can also write a script to access our web camera and take a real time input.


# 6. Conclusion

Results can be achieved by ML algorithms such as KNN, SVM with different parameters and feature scaling vectors but better results are achieved using CNN.
Deploying web application will be useful to the users to access the model more easily and efficiently without considering many requirement factors.
With this project I learnt a lot, especially about the working of CNN and also platform like Streamlit which is not very commonly used but is very helpful for such implementations.


# Bibliography 

* https://docs.streamlit.io/en/stable/
* https://www.murtazahassan.com/
* https://keras.io/getting_started/
* http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
* https://en.wikipedia.org/wiki/LeNet

# Note

* For requirement.txt you can use [pipreqs](https://github.com/bndr/pipreqs)
* Don't forget to add necessary files before deploying on streamlit
* Original Dataset: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
* Dataset used in this project: https://drive.google.com/file/d/1cZr91zzHl93H1cbmIwx_7upfQPPMPi33/view

# Contact me

[Linkedln](https://www.linkedin.com/in/abhi7585/)
[Twitter](https://twitter.com/the_abhi_7585)


# Development and Credits

Want to contribute? Great!
Feel free to add any issues and pull request.

<h3 align="center">Show some &nbsp;❤️&nbsp; by starring this repo! </h3>

