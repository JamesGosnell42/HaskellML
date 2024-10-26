module Models.LinearModel where
import Data.Matrix as M 
import Data.Vector as V
import Data.List
import Models.Types
import Debug.Trace

-- Function to add bias term to each input vector
addBias :: Matrix Double -> Matrix Double
addBias matrix = M.transpose$fromLists [1.0 : V.toList (getCol i matrix) | i <- [1..ncols matrix]]

addBiases::([Int], [Matrix Double]) -> ([Int], [Matrix Double])
addBiases (labels, matrices) = (labels, Data.List.map addBias matrices)

pseudoInverse :: Matrix Double -> Matrix Double
pseudoInverse matrix =
    case inverse (multStd (M.transpose matrix) matrix) of
        Right invMat -> multStd invMat (M.transpose matrix)
        Left err-> zero (ncols matrix) (nrows matrix)

-- Function to initialize weights using the pseudoinverse algorithm (w = (XTX)-1XTY)
initializeWeights :: ([Int], [Matrix Double]) -> Matrix Double
initializeWeights (labels, matrices) =
    let y = fromLists [[fromIntegral label] | label <- labels]
        xWithBias = fromLists [M.toList matrix | matrix <- matrices]
        xPseudoInv = pseudoInverse xWithBias
    in  multStd xPseudoInv y



-- Function to calculate the classification error
errorCalc :: Matrix Double -> [Matrix Double] -> [Int] -> Double
errorCalc w xs y =
    let predictions = [Data.List.head$M.toList$signum (M.transpose w * x)|x<-xs]
        -- Identify misclassified points
        misclassified = [ if (signum$fromIntegral y_i )/= (signum p) then 1 else 0| (y_i, p) <- Data.List.zip y predictions]

    in fromIntegral (Data.List.sum misclassified) / fromIntegral (Data.List.length y)

newWeights::Matrix Double -> [Matrix Double] -> [Int] -> Matrix Double
newWeights w x y = undefined

-- Linear classification using gradient descent
gradientDescent :: Matrix Double -> Matrix Double -> [Int] -> Int -> Matrix Double
gradientDescent w x y iterations =
    let stepSize = 0.01  -- Learning rate

        -- Calculate the gradient: (X^T * (X * w - y))
        yMatrix = fromLists [[fromIntegral yi] | yi <- y]
        errorMatrix = elementwise (-) (multStd x w) yMatrix
        gradient = multStd (M.transpose x) errorMatrix

        -- Update weights: w = w - stepSize * gradient
        updatedW = iterate (\weights -> elementwise (-) weights (scaleMatrix stepSize gradient)) w !! iterations
    in updatedW

-- Linear Regression using the Pocket Algorithm
linearRegressionPocket :: Matrix Double -> ([Int], [Matrix Double]) -> Int -> Matrix Double
linearRegressionPocket w (y, xs) 0 = w
linearRegressionPocket w (y, xs) iterations =
    let -- Make predictions using the current weight vector
        predictions = [Data.List.head$M.toList$signum (M.transpose w * x)|x<-xs]
        
        -- Identify misclassified points
        misclassified = [(x, y_i) | (x, y_i, p) <- Data.List.zip3 xs y predictions, (signum$fromIntegral y_i )/= p]
        
        
        -- Update weights if there are misclassified points
        newW = case misclassified of
            [] -> w
            ((x_mis, y_mis):rest) -> w + scaleMatrix (fromIntegral y_mis) x_mis
    in let 
        -- Recursive call with updated weights and decreased iteration count
        future = linearRegression newW (y, xs) (iterations - 1)
        --check for errorCalc to return best found line
        in if errorCalc newW xs y < errorCalc future xs y then newW else future 

-- Linear classification using linear regression
linearRegression::Matrix Double -> ([Int], [Matrix Double]) -> Int -> Matrix Double
linearRegression w (y, xs) 0 = w
linearRegression w (y, xs) iterations =
    let -- Make predictions using the current weight vector
        predictions = [Data.List.head$M.toList$signum (M.transpose w * x)|x<-xs]
        
        -- Identify misclassified points
        misclassified = [(x, y_i) | (x, y_i, p) <- Data.List.zip3 xs y predictions, (signum$fromIntegral y_i )/= p]
        
        -- Update weights if there are misclassified points
        newW = case misclassified of
            [] -> w
            ((x_mis, y_mis):rest) -> w + scaleMatrix (fromIntegral y_mis) x_mis
    in case misclassified of
            [] -> w
            _ -> linearRegression newW (y, xs) (iterations - 1)
        
 


    