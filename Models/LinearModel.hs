module Models.LinearModel where
import Data.Matrix as M 
import Data.Vector as V
import Data.List
import Models.Types
import Debug.Trace
import System.Random (randomRIO)

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

errnum :: Matrix Double -> [Int] -> [Matrix Double] -> Matrix Double
errnum w y x = 
    let 
        --logTerms = [Data.List.head $ M.toList $ signum (scaleMatrix (fromIntegral (-yn)) (M.transpose w * xn)) | (yn, xn) <- Data.List.zip y x]
        --Data.List.head $ M.toList $  (scaleMatrix (fromIntegral (yn)) xn) /(1+ exp (
        bottomterms = [Data.List.head $ M.toList (scaleMatrix (fromIntegral (yn)) (multStd (M.transpose w) xn)) | (yn, xn) <- Data.List.zip y x]
        topterms = [(scaleMatrix (fromIntegral (yn)) xn) | (yn, xn) <- Data.List.zip y x]
        logTerms =[scaleMatrix (1/(1+exp b)) t| (t, b) <- Data.List.zip topterms bottomterms]
    in scaleMatrix (-(1 / fromIntegral (Data.List.length y)))  (matrixSum logTerms)

matrixSum:: [Matrix Double] -> Matrix Double
matrixSum ms =  Data.List.foldr (elementwise (+)) (M.zero (nrows$Data.List.head ms) (ncols$Data.List.head ms)) ms

addToMatrix :: Matrix Double -> Double -> Matrix Double
addToMatrix matrix value = M.fromList rows cols updatedList
  where
    rows = nrows matrix
    cols = ncols matrix
    updatedList = Data.List.map (+ value) (M.toList matrix)

-- Linear classification using gradient descent
gradientDescent :: Matrix Double -> ([Int], [Matrix Double]) -> Int ->IO ( Matrix Double)
gradientDescent w _ 0 =return w
gradientDescent w (y, x) iterations =
    let stepSize = 0.1  -- Learning rate
        errn = scaleMatrix (-0.1) (errnum w y x)
        -- Calculate the gradient: (X^T * (X * w - y))
        updatedW = elementwise (+) w errn
    in gradientDescent (updatedW) (y, x) (iterations-1)

-- Linear classification using stochastic gradient descent
stochasticGradientDescent :: Matrix Double -> ([Int], [Matrix Double]) -> Int ->IO ( Matrix Double)
stochasticGradientDescent w _ 0 =return w
stochasticGradientDescent w (y, x) iterations = do
    let predictions = [Data.List.head $ M.toList $ signum (M.transpose w * xn) | xn <- x]
        
        -- Identify misclassified points
        misclassified = [(xn, y_i) | (xn, y_i, p) <- Data.List.zip3 x y predictions, (signum $ fromIntegral y_i) /= p]

    -- Randomly select a misclassified point, if any
    case misclassified of
        [] -> return w
        _  -> do
            -- Generate a random index to pick a misclassified point
            idx <- randomRIO (0, Data.List.length misclassified - 1)
            let (x_mis, y_mis) = misclassified !! idx
            -- Update the weights using the randomly selected point
            let stepSize = 0.1  -- Learning rate
        -- Calculate the gradient: (X^T * (X * w - y))
            let errn = scaleMatrix (-0.1) (errnum w [y_mis] [x_mis] )
            let updatedW = elementwise (+) w errn
            stochasticGradientDescent (updatedW) (y, x) (iterations-1)

-- Linear classification using the pocket algorithm
linearRegressionPocket :: Matrix Double -> ([Int], [Matrix Double]) -> Int -> IO (Matrix Double)
linearRegressionPocket w (y, xs) 0 = return w
linearRegressionPocket w (y, xs) iterations = do
    -- Make predictions using the current weight vector
    let predictions = [Data.List.head $ M.toList $ signum (M.transpose w * x) | x <- xs]
        
        -- Identify misclassified points
        misclassified = [(x, y_i) | (x, y_i, p) <- Data.List.zip3 xs y predictions, (signum $ fromIntegral y_i) /= p]

    -- Randomly select a misclassified point, if any
    newW <- case misclassified of
        [] -> return w
        _  -> do
            -- Generate a random index to pick a misclassified point
            idx <- randomRIO (0, Data.List.length misclassified - 1)
            let (x_mis, y_mis) = misclassified !! idx
            -- Update the weights using the randomly selected point
            return (w + scaleMatrix (fromIntegral y_mis) x_mis)

    -- Recursive call with updated weights and decreased iteration count
    future <- linearRegressionPocket newW (y, xs) (iterations - 1)

    -- Check if the current weights produce a lower error than future weights
    if errorCalc w xs y < errorCalc future xs y
        then return w
        else return future

-- Linear classification using linear regression
linearRegression::Matrix Double -> ([Int], [Matrix Double]) -> Int -> IO (Matrix Double)
linearRegression w (y, xs) 0 = return $w
linearRegression w (y, xs) iterations = do
    let -- Make predictions using the current weight vector
        predictions = [Data.List.head$M.toList$signum (M.transpose w * x)|x<-xs]
        
        -- Identify misclassified points
        misclassified = [(x, y_i) | (x, y_i, p) <- Data.List.zip3 xs y predictions, (signum$fromIntegral y_i )/= p]
        
    -- Randomly select a misclassified point, if any
    newW <- case misclassified of
        [] -> return w
        _  -> do
            -- Generate a random index to pick a misclassified point
            idx <- randomRIO (0, Data.List.length misclassified - 1)
            let (x_mis, y_mis) = misclassified !! idx
            -- Update the weights using the randomly selected point
            return (w + scaleMatrix (fromIntegral y_mis) x_mis)
    case misclassified of
            [] -> return $w
            _ -> linearRegression newW (y, xs) (iterations - 1)


    