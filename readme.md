<p align="center">
  <img src="imgs\banner.jpg" width="1000" >
</p>

# Chronic Kidney Disease Prediction
This is a machine learning project that is trained on a collected data of people who are infected by these disease and some other people.
<br/>
It includes a lot of data about the individual and I used that feature to make an optimal model that will predict if someone is suffering from this disease or not.
<br/><br/>
## What is Chronic Kidney Disease?
Chronic kidney disease (CKD) means your kidneys are damaged and can't filter blood the way they should. The disease is called “chronic” because the damage to your kidneys happens slowly over a long period of time. This damage can cause wastes to build up in your body. CKD can also cause other health problems. [Source](https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/what-is-chronic-kidney-disease)
<br/><br/>
## What causes Chronic Kidney Disease?
The two most common causes of CKD are diabetes and high blood pressure. Diabetes means that your blood sugar is too high, which can damage your kidneys. High blood pressure means that the force of blood in your blood vessels is too strong, which can damage your blood vessels and lead to CKD. [Source](https://www.kidneyfund.org/all-about-kidneys/chronic-kidney-disease-ckd)
<br/><br/>
## My solution approachs in this model
### Data preparation part
- The dataset has very low data records but a little bit better amount of features/columns. So, I added more columns by combining existing one to get many number of columns to work with. Since, we don't have enough data, that was my only option.
- The dataset has some columns with a lot of null values. I removed these columns since they will surely bias my model if I was to impute them.
- I removed some rows which have at least one null record. Because, they will corrupt the model.
<br/>

### Model training part
- I used accuracy, precision, recall and F1 scores as model measuring metrices.
- I tried support vector machine, decision tree and random forest algorithms.
- Support vector machine was severly overfitting the dataset
- Decision tree was not fitting in quite as good as random forest. Because, the accuracy and other metrics were very low relatively to the random forest. Since, random forest is collection of decision trees, that will cover the issue of a single decision tree algorithm.
- RandomForest was fitting in the dataset quite very well. Ofcourse there is a little bit of over fitting but better than support vector machine. So, my final model is Random forest algorithm.
<br/>
<p align="center">
  <img src="imgs\rec.png" width="700" >
</p>
