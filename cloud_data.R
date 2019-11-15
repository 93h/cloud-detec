library(tidyverse)

# Get the data for three images
path <- "data"
image1 <- read.table(paste0('data/', 'image1.txt'), header = F)
image2 <- read.table(paste0('data/', 'image2.txt'), header = F)
image3 <- read.table(paste0('data/', 'image3.txt'), header = F)

# Add informative column names.
collabs <- c('y','x','label','NDAI','SD','CORR','DF','CF','BF','AF','AN')
names(image1) <- collabs
names(image2) <- collabs
names(image3) <- collabs

# take a peek at the data from image1
head(image1)
summary(image1)

# The raw image (red band, from nadir).
ggplot(image1) + 
  geom_point(aes(x = x, y = y, color = NDAI))

# Plot the expert pixel-level classification
ggplot(image1) + geom_point(aes(x = x, y = y, color = factor(label))) +
  scale_color_discrete(name = "Expert label")

# Class conditional densities.
ggplot(image1) + 
  geom_density(aes(x = AN, group = factor(label), fill = factor(label)), 
               alpha = 0.5) +
  scale_fill_discrete(name = "Expert label")

ggplot(image1) + 
  geom_point(aes(x = x, y = y, color = factor(label))) + 
  theme_bw() +
  theme(axis.text = element_text(size = 5),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border= element_blank(),
        legend.text = element_text(size =10),
        legend.title = element_text(size = 15))+
  scale_color_manual(name = "Expert label", 
                     values = c("#3182bd", "#9ecae1", "#deebf7"), 
                     labels = c("Ice", "Unknown", "Clouds"))

ggplot(image2) + 
  geom_point(aes(x = x, y = y, color = factor(label))) + 
  theme_bw() +
  theme(axis.text = element_text(size = 5),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border= element_blank(),
        legend.text = element_text(size =10),
        legend.title = element_text(size = 15))+
  scale_color_manual(name = "Expert label", 
                     values = c("#3182bd", "#9ecae1", "#deebf7"), 
                     labels = c("Ice", "Unknown", "Clouds"))

ggplot(image3) + 
  geom_point(aes(x = x, y = y, color = factor(label))) + 
  theme_bw() +
  theme(axis.text = element_text(size = 5),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border= element_blank(),
        legend.text = element_text(size =10),
        legend.title = element_text(size = 15))+
  scale_color_manual(name = "Expert label", 
                     values = c("#3182bd", "#9ecae1", "#deebf7"), 
                     labels = c("Ice", "Unknown", "Clouds"))


images <- rbind(image1, image2, image3)
# Correlation matrix of radiances for cloud pixels
corr_cloud <- images %>% 
  filter(label == 1) %>% 
  dplyr::select(DF, CF, BF, AF, AN) %>% 
  cor()

# Correlation matrix of radiances for non-cloud pixels
corr_ice <- images %>% 
  filter(label == -1) %>% 
  dplyr::select(DF, CF, BF, AF, AN) %>% 
  cor()

# Correlation matrix of NDAI, SD, CORR for cloud pixels
corr2_cloud <- images %>% 
  filter(label == 1) %>% 
  dplyr::select(NDAI, SD, CORR) %>% 
  cor()

# Correlation matrix of radiances for non-cloud pixels
corr2_ice <- images %>% 
  filter(label == -1) %>% 
  dplyr::select(NDAI, SD, CORR) %>% 
  cor()

#fwd.model = step(polr(as.factor(label) ~ 1, data = images), direction = "forward", scope = (~ SD + NDAI + CORR + DF + CF + BF + AF + AN))

library(leaps)
subsets <- regsubsets(label ~ SD + NDAI + CORR + DF + CF + BF + AF + AN, data = images, nvmax = 3)
summary(subsets)


# Uncertain expert labels are removed from the dataset 
certain_images <- images %>%
  filter(label != 0) %>%
  mutate(label = as.factor(label))

#Split the data into training and test sets
train <- certain_images %>% sample_frac(0.8)
test <- certain_images %>% anti_join(train)


library(mlbench)
library(caret)
library(randomForest)

fit_rf = randomForest(label ~ NDAI + SD + CORR + DF + CF + BF + AF + AN, data = train, importance = TRUE)
fit_rf_limited = randomForest(label ~ NDAI + SD + CORR + AN, data = train, importance = TRUE)

importance(fit_rf)
varImpPlot(fit_rf)
