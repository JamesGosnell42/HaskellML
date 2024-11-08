module Models.LinearModel where
import Data.Matrix as M 
import Data.Vector as V
import Data.List as D
import Models.Types
import Debug.Trace
import System.Random (randomRIO)
import Control.Monad (when)
import GHC.Enum (Enum(pred))
import GHC.CmmToAsm.AArch64.Instr (x0)
import GHC.Parser.Lexer (xset)



errorCalc :: Matrix Double -> Matrix Double -> Matrix Double -> Double
errorCalc w xs y =
    let predictions = M.multStd xs w
        incorect = M.elementwise (\a b -> if signum a == signum b then 0 else 1) predictions y
        misclassifiedCount = D.sum incorect
    in (fromIntegral misclassifiedCount / fromIntegral (D.length y)) * 100


-- Linear classification using the pocket algorithm
pla :: Matrix Double -> (Matrix Double, Matrix Double) -> Int -> IO (Matrix Double)
pla w (y, xs) 0 = return w
pla w (y, xs) iterations = do
    let predictions = M.multStd xs w
    let incorect = M.elementwise (\a b -> signum a /= signum b) predictions y
    let misclassified = [(x, y_i) | (x, y_i, p) <- D.zip3 (M.toLists xs) (M.toList y) (M.toList incorect), p]

    newW <- case misclassified of
        [] -> return w
        _  -> do
            idx <- randomRIO (0, D.length misclassified - 1)
            let (x_mis, y_mis) = misclassified !! idx
            return (w + scaleMatrix y_mis (M.fromList (D.length x_mis) 1 x_mis))
    
    future <- pla newW (y, xs) (iterations - 1)
    
    -- Debug: Print the current and future error rates
    let currentError = errorCalc w xs y
    let futureError = errorCalc future xs y

    if currentError < futureError
        then return w
        else return future

linearRegression:: Matrix Double -> (Matrix Double, Matrix Double) -> Int -> IO (Matrix Double)
linearRegression w (y, xs) 0 = return w
linearRegression w (y, xs) iterations = do
    let predictions = M.multStd xs w
    let incorect = M.elementwise (\a b -> if signum a == signum b then 0 else 1) predictions y
    let gradient = M.multStd (M.transpose xs) incorect
    let newW = M.elementwise (-) w (scaleMatrix 0.01 gradient) -- Update weights with learning rate 0.01

    future <- linearRegression newW (y, xs) (iterations - 1)
    
    -- Debug: Print the current and future error rates
    let currentError = errorCalc w xs y
    let futureError = errorCalc future xs y
    if currentError < futureError
        then return w
        else return future

