module Models.Util where
import Models.Types
import Data.Matrix as M
import Data.List as D

import           Data.Maybe

-- Cross-Entropy Loss Function for logistic error
logisticErr :: Matrix Double -> Matrix Double -> Matrix Double -> Double
logisticErr w x y = 
    let predictions = x * w
        logPredictions = fmap (\p -> log (1-exp p)) predictions
    in -(D.sum logPredictions) / (fromIntegral (nrows y))

-- Error calculation for linear regression
errorCalc :: Matrix Double -> Matrix Double -> Matrix Double -> Double
errorCalc w xs y =
    let predictions = M.multStd2 xs w
        incorect = M.elementwise (\a b -> if signum a == signum b then 0 else 1) predictions y
        misclassifiedCount = D.sum incorect
    in (fromIntegral misclassifiedCount / fromIntegral (D.length y)) * 100

--used with a nx1 and a nxm matrix to multiply each row of the second matrix by the corresponding element of the first matrix
byrow :: Matrix Double -> Matrix Double -> Matrix Double
byrow first second = D.foldl yrow second [1..(nrows second)]
    where
        yrow :: Matrix Double -> Int -> Matrix Double
        yrow acc rowIdx =
            let sval = M.getElem rowIdx 1 first
            in mapRow (\_ x -> sval * x) rowIdx acc


--sums columns of a nxm matrix and rotates the result to a mx1 matrix
sumColumns :: Matrix Double -> Matrix Double
sumColumns m = 
    let cols = ncols m
        colSums = [D.sum (M.getCol j m) | j <- [1..cols]]  -- List of column sums
    in M.fromList cols 1 colSums 

--cross validation function for finding Ecv and weights of a model
crossValidation:: Data -> Model -> Double-> Int-> IO (Matrix Double, Double)
crossValidation dat@(y, x) model lambda iterations = do
    errors <- mapM (\i -> runModel dat model lambda iterations i) [1..nrows y]
    finalmodel <- model x dat iterations
    let avgError = (D.sum errors) / (fromIntegral (D.length errors))
    return (finalmodel, avgError)

--helper function for running model wwith cross validation
runModel :: Data -> Model -> Double-> Int -> Int -> IO Double
runModel (y, x) model lambda iterations rowidx = do
    let testy = M.rowVector (getRow rowidx y)
        textx =  M.rowVector (getRow rowidx x )
        trainy = removeRow rowidx y
        trainx = removeRow rowidx x
    weights <- model trainx (trainy, trainx) iterations
    return$errorCalc weights textx testy

--helper function for removing the test row from the training matrix
removeRow :: Int -> Matrix Double -> Matrix Double
removeRow 1 mat = submatrix 2 (nrows mat) 1 (ncols mat) mat
removeRow rowidx mat = if rowidx == nrows mat then submatrix 1 (rowidx - 1) 1 (ncols mat) mat
    else submatrix 1 (rowidx - 1) 1 (ncols mat) mat <-> submatrix (rowidx + 1) (nrows mat) 1 (ncols mat) mat
