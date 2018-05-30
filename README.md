# Facebook Footsteps - Predict Facebook checkins

### Project Idea

The idea is to predict where a user could be checking into from their location and timestamp. This project was a [coding challenge](https://www.kaggle.com/c/facebook-v-predicting-check-ins) on kaggle.com, and the dataset we are using is from the website itself.

A system which could classify and analyze social media check-in data to find patterns on how users check-in activity would be beneficial for the customers and businesses. Although, achieving this would never have a hundred percent accuracy, but a good prediction would certainly do wonders for both.

### Applications

1. See the most common places of interests as per the geographical area.
2. Allow businesses to promote themselves as per customer check-ins and find ways to improve and work on customer retention
3. Realize how customer visit places as per the seasons, days, hours, etc. More check-ins in evening could be a sign of restaurant outings and allow to find favorable eating joints.
4. Allow users to find the most visited hangout places around.
5. Allow tourists to better plan their trips highlighting how people go about visiting a city and area as per different time of the day.
6. Customized offers and personalized customer based advertising on social media and other networks.

### Dataset

The dataset contains train and test data files with below columns:

1. row_id -> id of the checkin event
2. x -> x coordinate of checkin
3. y -> y coordinate of checkin
4. accuracy - location accuracy
5. timestamp - timestamp of checkin
6. place_id -> business id

One of the major challenges was to make the data more meaningful by finding patterns on features. Hence, data preprocessing and feature extraction is a critical step in this scenario.

### Exploratory Analysis

To reinforce our understanding of the problem and to visualize the data we start by plotting all the check-ins within a smaller grid of 500 X 500 meters taken at random from the given larger grid. 
We used R script to generate the plot. The script is CreatePlots.R in the repository. A subset of script which generates the same is
```
ggplot(small_trainz, aes(x, y )) +
  geom_point(aes(color = place_id)) + 
  theme_minimal() +
  theme(legend.position = "none") +
  ggtitle("Check-ins for a smaller 500 X 500m grid")
```

We have plotted only place_ids that have more than 100 check-ins to visualize the clusters.

![Exploration](https://github.com/shivamgulati1991/Facebook--Footsteps-Prediction/blob/master/Exploratory_Analysis/Rplot01.png)

To make these clusters separable and more evident we tried using one more feature “hour of the day” as the third dimension for our plot.
Addition of third dimension helps and we can see that our assumption that hour of day affects the check-ins for a place is valid. We tried plotting the same data using “weekday” feature that resulted in similar plot. The same was generated as
```
plot_ly(data = small_trainz, x = small_trainz$x , y = small_trainz$y, z = small_trainz$hour, color = small_trainz$place_id,  type = "scatter3d", mode = "markers", marker=list(size= 5)) %>% layout(title = "Place_ids clustered by x, y and hour of day")
```

These plots confirm our understanding that check-ins from user depends on the different time components like hour of the day and weekday.

![Exploration](https://github.com/shivamgulati1991/Facebook--Footsteps-Prediction/blob/master/Exploratory_Analysis/Rplot03.png)

### Techniques used

Post the processing, wee implemented the below approaches to compare and get the best results.

* K Nearest Neighbour

Once we have a smaller grid of 250 X 250 meters in place first thing that comes to mind is KNN for the classification task. It is very easy to implement and give good results. The only tricky part about applying KNN is finding out the optimal weights for the variables used. We have used hit and trial method to optimize our model. We plan to use data exploratory techniques in future to narrow down on optimal weights for KNN in final version of the report.

* Random Forest

Random Forest was our second choice for the classifier as it is efficient and generally results in more accurate results. The performance factor of random forest is important for us as we are doing classification task on the fly. We chose random forest also because it gives us estimate of the importance of different variables in classification task this helped us in fine-tuning our model for other classifiers as well. We tried different flavors of Random Forest available in Python and achieved best results using [sklearn](http://scikit-learn.org/) random forest classifier.

* Boosted Trees

To improve the accuracy further, we tried boosted trees i.e. tree ensemble model for classification and regression trees (CART). We used [XGBoost library](http://xgboost.readthedocs.io/en/latest/model.html), short for “Extreme Gradient Boosting”, where the term “Gradient Boosting” is proposed in the paper Greedy Function Approximation: A Gradient Boosting Machine, by Friedman.

### Results

We obtained the below accuracies:

| Method        | Accuracy      |
| ------------- |:-------------:|
| KNN           | 0.42          |
| Random Forest | 0.48          |
| Boosted Trees | 0.54          |

Our best results were with the XGBoost library for Boosted trees with an accuracy of 0.54.

![Results](https://github.com/shivamgulati1991/Facebook--Footsteps-Prediction/blob/master/result.PNG)


### Key Learnings

* Spatial data and ways to handle it
* Different classifiers - pros and cons
* Data exploration and feature analysis for Big Data

### Future Enhancements

* Improved processing for better results
* Offline Data Processing for Real-Time Results
* Recommendations based on Check-ins
* Ensemble and Improved Models

### References

1. Problem Statement - https://www.kaggle.com/c/facebook-v-predicting-check-ins
2. Dataset - https://www.kaggle.com/c/facebook-v-predicting-check-ins/data
3. sklearn - http://scikit-learn.org/
4. XGBoost - http://xgboost.readthedocs.io/en
