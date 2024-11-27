module Models.Util where

import Models.Types
import Data.Matrix as M
import Data.Vector as V
import Data.List as D
import Control.Concurrent.Async (mapConcurrently)
import Control.Parallel.Strategies (parMap, rdeepseq)

-- Function to check if a value is NaN and replace it with 0 if it is
safeLog :: Double -> Double
safeLog p = 
    let result = log (1 - exp p)
    in if isNaN result then 0 else result

-- Logistic error
logisticErr :: Matrix Double -> Data -> Double
logisticErr w (y, x) = 
    let predictions = byrow (-y) (x * w)
        log_predictions = parMap rdeepseq (safeLog) (M.toList predictions)
    in -(D.sum log_predictions) / fromIntegral (nrows y)

-- Error calculation for linear regression
errorCalc :: Matrix Double -> Data -> Double
errorCalc w (y, xs) =
    let predictions = M.multStd2 xs w
        incorect = M.elementwise (\a b -> if signum a == signum b then 0 else 1) predictions y
        misclassifiedCount = D.sum (M.toList incorect)
    in (fromIntegral misclassifiedCount / fromIntegral (nrows y)) * 100

--used with a nx1 and a nxm matrix to multiply each row of the second matrix by the corresponding element of the first matrix
byrow :: Matrix Double -> Matrix Double -> Matrix Double
byrow first second = M.fromLists $ parMap rdeepseq (\(row, sval) -> D.map (* sval) row) (D.zip (M.toLists second) (M.toList first))

--sums columns of a nxm matrix and rotates the result to a mx1 matrix
sumColumns :: Matrix Double -> Matrix Double
sumColumns m = 
    let cols = ncols m
        colSums = [D.sum (M.getCol j m) | j <- [1..cols]]  -- List of column sums
    in M.fromList cols 1 colSums 

--cross validation function for finding Ecv and weights of a model
crossValidation :: Data -> Model -> Double -> Int -> IO (Matrix Double, Double)
crossValidation dat@(y, x) model lambda iterations = do
    errors <- mapConcurrently (\i -> runModel dat model lambda iterations i) [1, 11..nrows y]
    finalmodel <- model x dat iterations
    let avgError = (D.sum errors) / (fromIntegral (D.length errors))
    return (finalmodel, avgError)

--helper function for running model wwith cross validation
runModel :: Data -> Model -> Double -> Int -> Int -> IO Double
runModel (y, x) model lambda iterations rowidx = do
    let rowend = if (rowidx + 10 > nrows y) then nrows y else rowidx + 10
    let testy = submatrix rowidx rowend 1 (ncols y) y
        testx = submatrix rowidx rowend 1 (ncols x) x
        trainy = removeRows rowidx rowend y
        trainx = removeRows rowidx rowend x
    weights <- model trainx (trainy, trainx) iterations
    return $ errorCalc weights (trainy, trainx)

-- Helper function for removing the test rows from the training matrix
removeRows :: Int -> Int -> Matrix Double -> Matrix Double
removeRows rowidx rowend mat
    | rowidx == 1 = submatrix (rowend + 1) (nrows mat) 1 (ncols mat) mat
    | rowend >= nrows mat = submatrix 1 (rowidx - 1) 1 (ncols mat) mat
    | otherwise = submatrix 1 (rowidx - 1) 1 (ncols mat) mat <-> submatrix (rowend + 1) (nrows mat) 1 (ncols mat) mat