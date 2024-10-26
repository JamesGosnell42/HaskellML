module Models.KNN where
{--
K-Nearest Neighbor is a simple classificaiton model that uses no actual learning
The model simply takes an input, sorts known data by which is closest in feature space and classifies input by the highest number of class occurences in k nearest points
E.G: if you have the data set D where a data point has the features x and y and is in the form: ((x, y), classification) and k = 1
    D = {((1,1), dog), ((3,4), cat), ((1,2), dog), ((5,4), cat)} 
    input = (0,0)
    Dsorted = {((1,1), dog), ((1,2), dog), ((3,4), cat), ((5,4), cat)} 
    the nearest point to the input is ((1,1), dog) so input is a dog

The upsides of this model is that the resulting external error is always less then 2 times optimal error
The downsides to this model are that you can never calculate external error and the speed of classification is slow
--}