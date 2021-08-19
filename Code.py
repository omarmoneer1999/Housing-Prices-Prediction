#importing pandas to deal with the dataset
import pandas as pd

#readnig the datasets that we will work on it
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

#show our data to extract information about it and specify the features that we will predict on it
test.head()
test.columns

#specifying the features
y = train.SalePrice
features = ['LotArea', 'YearBuilt', 'OverallQual']

#prepare the data to work on
train_x = train[features].copy()
test_x = test[features].copy()

#importing the split tool to split the data into training and validation data
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(train_x,y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# importing and using the tree that we will use it in our prediction process
from sklearn.ensemble import RandomForestRegressor

# Make a function to find our best MAE
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]
from sklearn.metrics import mean_absolute_error
def mae_model(model,xt = x_train,xv =x_val,yt =y_train,yv =y_val):
    model.fit(xt,yt)
    preds = model.predict(xv)
    return mean_absolute_error(yv, preds)
for i in range(0, len(models)):
    mae = mae_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
    
# Fitting the model to the training data
model_1 = RandomForestRegressor(random_state =1)
model_1.fit(train_x,y)
preds_test = model_1.predict(test_x)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': test_x.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
