module Models.LogisticModel where
import Models.Util 
import Models.Types

import Data.Matrix as M 
import Data.Vector as V
import Data.List as D

import System.Random (randomRIO)

-- Logistic function
logistic :: Double -> Double
logistic n = 1 / (1 + exp (-n))

-- calculate: gradient of Error = -yx Theta(-y wTx) 
gradient :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
gradient w y x = 
    let 
        first = byrow (-y) x 
        xw = x * w
        neg_y_xw = M.elementwise (*) (-y) xw
        logisticres = fmap logistic neg_y_xw
        combined = byrow logisticres first
    in scaleMatrix (-(1 / fromIntegral (nrows y))) (sumColumns combined)

-- Logistic classification using gradient descent
gradientDescent :: Model 
gradientDescent w _ 0 = return w
gradientDescent w (y, x) iterations = do
    let stepSize = 0.01  -- Learning rate
        errn = M.scaleMatrix (-stepSize) (gradient w y x)
        updatedW = M.elementwise (+) w errn
    gradientDescent updatedW (y, x) (iterations - 1)

-- Logistic classification using stochastic gradient descent
stochasticGradientDescent :: Model 
stochasticGradientDescent w _ 0 = return w
stochasticGradientDescent w (y, x) iterations = do
    let stepSize = 0.01  -- Learning rate
    let numRows = nrows x
    idx <- randomRIO (1, numRows)
    let x_sel = M.rowVector (M.getRow idx x)
        y_sel = M.rowVector (M.getRow idx y)
        errn = M.scaleMatrix (-stepSize) (gradient w y_sel x_sel)
        updatedW = M.elementwise (+) w errn
    stochasticGradientDescent updatedW (y, x) (iterations - 1)