module Models.LogisticModel where
import Data.Matrix as M 
import Data.Vector as V
import Data.List as D
import Models.Types
import Debug.Trace
import System.Random (randomRIO)


-- Cross-Entropy Loss Function for logistic error
logisticErr :: Matrix Double -> Matrix Double -> Matrix Double -> Double
logisticErr w x y = 
    let predictions = M.multStd x w
        logPredictions = fmap (\p -> log (1-exp p)) predictions
    in -(D.sum logPredictions) / (fromIntegral (nrows y))

-- Logistic function
logistic :: Double -> Double
logistic n = 1 / (1 + exp (-n))

-- calculate: gradient of Error = -yx Theta(-y wTx) 
errnum :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
errnum w y x = 
    let 
        first = byrow (-y) x 
        xw = M.multStd x w
        neg_y_xw = M.elementwise (*) (-y) xw
        logisticres = fmap (\p -> logistic p) neg_y_xw
        combined = byrow logisticres first
    in (scaleMatrix (-(1 / fromIntegral (nrows y))) (sumColumns combined))


--used with a nx1 and a nxm matrix to multiply each row of the second matrix by the corresponding element of the first matrix
byrow :: Matrix Double -> Matrix Double -> Matrix Double
byrow first second = D.foldl yrow second [1..(nrows second)]
    where
        yrow :: Matrix Double -> Int -> Matrix Double
        yrow acc rowIdx =
            let sval = M.getElem rowIdx 1 first
            in mapRow (\_ x -> sval * x) rowIdx acc

sumColumns :: Matrix Double -> Matrix Double
sumColumns m = 
    let cols = ncols m
        colSums = [D.sum (M.getCol j m) | j <- [1..cols]]  -- List of column sums
    in M.fromList cols 1 colSums 

-- Linear classification using gradient descent
gradientDescent :: Matrix Double -> (Matrix Double, Matrix Double) -> Int -> IO (Matrix Double)
gradientDescent w _ 0 = return w
gradientDescent w (y, x) iterations = do
    let stepSize = 0.01  -- Learning rate
        errn = M.scaleMatrix (-stepSize) (errnum w y x)
        updatedW = M.elementwise (+) w errn
    gradientDescent updatedW (y, x) (iterations - 1)

-- Linear classification using stochastic gradient descent
stochasticGradientDescent :: Matrix Double -> (Matrix Double, Matrix Double) -> Int -> IO (Matrix Double)
stochasticGradientDescent w _ 0 = return w
stochasticGradientDescent w (y, x) iterations = do
    let stepSize = 0.01  -- Learning rate
    let numRows = nrows x
    idx <- randomRIO (1, numRows)
    let x_sel = M.rowVector (M.getRow idx x)
        y_sel = M.rowVector (M.getRow idx y)
        errn = M.scaleMatrix (-stepSize) (errnum w y_sel x_sel)
        updatedW = M.elementwise (+) w errn
    stochasticGradientDescent updatedW (y, x) (iterations - 1)