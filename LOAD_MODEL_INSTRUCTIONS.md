For our project to run do the following:
* Ensure you use python 3.11 which can be easily found as an interpreter option when using anaconda
* `pip install -r requirements.txt`
* Carefully go through each cell and refer to their colour:
  - All markdown cells can be run
  - For a section which is coloured <font color="red">red</font>, **NONE** of the cells need to be executed as the existing output can be referred to.
  - For a section which is coloured <font color="green">green</font>, **ALL** of the cells need to be executed as they contain important code needed for inference!
  

Our repo also contains some saved pkl files which ARE ESSENTIAL for the model to do inference.

These are the following files that we have attached:
1. xgb_model.joblib : This is our final trained model which we are loading in the cell to do inference
`
loaded_xgb_model = joblib.load('xgb_model.joblib')
result = loaded_xgb_model.predict(hidden_data)
`

2. svr_model.joblib : This was an experimental model which is also being loaded to test our model metrics.  
3. tfidf_kmeans_model.pkl : This is a utility model we developed that will be used by the final xgb model. 
4. latlong_kmeans_model.pkl: This is a utility model we developed that will be used by the final xgb model.

Wherever needed, our models have already been loaded as we did not need to use any large models which adheres to the scalability and performance judging criteria mentioned.

