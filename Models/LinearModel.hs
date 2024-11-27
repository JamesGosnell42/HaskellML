module Models.LinearModel where
import Models.Util 
import Models.Types

import Data.Matrix as M 
import Data.Vector as V
import Data.List as D

import Debug.Trace
import System.Random (randomRIO)



-- perceptron learning algorithm with pocket
pla :: Model 
pla w (y, xs) 0 = return w
pla w (y, xs) iterations = do
    let predictions = xs * w
    let incorect = M.elementwise (\a b -> signum a /= signum b) predictions y
    let misclassified = [(x, y_i) | (x, y_i, p) <- D.zip3 (M.toLists xs) (M.toList y) (M.toList incorect), p]

    newW <- case misclassified of
        [] -> return w
        _  -> do
            idx <- randomRIO (0, D.length misclassified - 1)
            let (x_mis, y_mis) = misclassified !! idx
            return (w + scaleMatrix y_mis (M.fromList (D.length x_mis) 1 x_mis))
    
    future <- pla newW (y, xs) (iterations - 1)
    
    let currentError = errorCalc w (y, xs)
    let futureError = errorCalc future (y, xs)
    if currentError < futureError
        then return w
        else return future

--linear regresion classification with pocket algorithm
linearRegression:: Model 
linearRegression w (y, xs) 0 = return w
linearRegression w (y, xs) iterations = do
    let predictions = M.multStd2 xs w
    
    let incorect = M.elementwise (-) predictions y
    let gradient = (M.transpose xs) * incorect

    let newW = M.elementwise (-) w (scaleMatrix 0.01 gradient) -- Update weights with learning rate 0.01

    future <- linearRegression newW (y, xs) (iterations - 1)
    
    let currentError = errorCalc w (y, xs)
    let futureError = errorCalc future (y, xs)
    if currentError < futureError
        then return w
        else return future

-- Function to initialize weights using the pseudoinverse algorithm (w = (XTX)-1XTY)
pseudoInverse :: (Matrix Double, Matrix Double) -> Double -> Matrix Double
pseudoInverse (labels, matrix) lambda =
    let  xPseudoInv = case inverse (M.elementwise (+) ((M.transpose matrix) * matrix) (M.scaleMatrix lambda (identity (nrows matrix)))) of 
                        Right invMat -> invMat * (M.transpose matrix)
                        Left err-> zero (ncols matrix) (nrows matrix)
    in   xPseudoInv * labels