install.packages("dplyr")
install.packages("plotly")

library(dplyr) #dataframe manipulation
library(ggplot2) #viz
library(plotly) #3D plotting

fb <- fread("train.csv", integer64 = "character", showProgress = FALSE)

fb %>% filter(x >5, x <5.5, y >5, y < 5.5) -> fb
head(fb, 3)

fb$hour = (fb$time/60) %% 24
fb$weekday = (fb$time/(60*24)) %% 7
fb$month = (fb$time/(60*24*30)) %% 12 #month-ish
fb$year = fb$time/(60*24*365)
fb$day = fb$time/(60*24) %% 365


small_train = fb[fb$time < 7.3e5,]
small_val = fb[fb$time >= 7.3e5,] 

ggplot(small_trainz, aes(x, y )) +
  geom_point(aes(color = place_id)) + 
  theme_minimal() +
  theme(legend.position = "none") +
  ggtitle("Check-ins for a smaller 500 X 500m grid")


small_train %>% count(place_id) %>% filter(n > 500) -> ids
small_trainz = small_train[small_train$place_id %in% ids$place_id,]

plot_ly(data = small_trainz, x = small_trainz$x , y = small_trainz$y, z = small_trainz$hour, color = small_trainz$place_id,  type = "scatter3d", mode = "markers", marker=list(size= 5)) %>% layout(title = "Place_ids clustered by x, y and hour of day")
