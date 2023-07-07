# README
Check `model development and validation notebooks` folder for detailed explanation.  

Exported Model: RFCloudless_model.pkl
The model is developed by using data satellite image data from dates with less than 15% of cloud cover above training data coordinates. From these dates all feature bands from Sentinel-1 and Sentinel-2 are extracted to train our Random Forest model. For details, read 'Iteration4.ipynb'. For development process, refer to `model_development.ipynb`. 



Data used: 
The data used is extracted from Sentinel-1 and Sentinel-2 data provided by Microsoft Planetary Computer Hub. The extracted data are stored in the form of dataframe in 'Additional Dataset' folder. 

Meanwhile, 'data' and 'submission' folder contain the training data and submission template data respectively. 



Development process:
The two Jupyter notebooks 'data_preparation' and 'model_development' shows the process of how we prepare our data and the development process to our final model. To be able to run the 'model_development.ipynb', 'data', 'model_dev_functions', and 'image_analysis' folders are needed. 

Note: While these notebooks are stored in google drive, certain important EDA images can only be viewed on local system as we have not fixed them on google colab. It is encouraged to download this folder instead of viewing them on google. 


In this model, only Sentinel-1 and Sentinel-2 data from Microsoft Planetary Computer Hub is used. 
